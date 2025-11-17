import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import math
import os
from typing import List, Tuple, Optional
import random
from einops import rearrange

# Hindi Characters (Devanagari script)
HINDI_CHARS = [
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
    'क', 'ख', 'ग', 'घ', 'ड़', 'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',
    'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ',
    'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्',
    'ं', 'ः', '़', 'ँ'
]

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
BLANK_TOKEN = '<BLANK>'

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, BLANK_TOKEN]
ALL_CHARS = SPECIAL_TOKENS + HINDI_CHARS

char2idx = {char: idx for idx, char in enumerate(ALL_CHARS)}
idx2char = {idx: char for char, idx in char2idx.items()}

VOCAB_SIZE = len(ALL_CHARS)
PAD_IDX = char2idx[PAD_TOKEN]
SOS_IDX = char2idx[SOS_TOKEN]
EOS_IDX = char2idx[EOS_TOKEN]
BLANK_IDX = char2idx[BLANK_TOKEN]


def load_ground_truth(gt_file_path: str, base_dir: str) -> Tuple[List[str], List[str]]:
    """Load ground truth labels from text file"""
    image_paths = []
    labels = []
    
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                img_relative_path = ' '.join(parts[:-1])
                label = parts[-1].strip()
                full_path = img_relative_path
                
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                    labels.append(label)
                else:
                    print(f"Warning: Image not found: {full_path}")
    
    return image_paths, labels


def window_partition(x, window_size):
    """
    Partition tensor into non-overlapping windows
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention with relative position bias
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with shifted window mechanism
    """
    def __init__(self, dim, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer - reduces spatial dimensions by 2x and doubles channels
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # Merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinStage(nn.Module):
    """
    A Swin Transformer stage consisting of multiple Swin blocks
    """
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop
            )
            for i in range(depth)
        ])
        
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = H // 2, W // 2
        
        return x, H, W


class SwinTransformerEncoder(nn.Module):
    """
    Swin Transformer Encoder for OCR
    Implements hierarchical vision transformer with shifted windows
    """
    def __init__(self, img_size=(64, 256), patch_size=4, in_chans=1, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=4, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., output_dim=768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_stages = len(depths)
        self.output_dim = output_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        # Calculate dimensions after patch embedding
        self.patches_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        
        # Build stages
        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            stage = SwinStage(
                dim=int(embed_dim * 2 ** i_stage),
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=PatchMerging if i_stage < self.num_stages - 1 else None
            )
            self.stages.append(stage)
        
        # Final normalization
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_stages - 1)))
        
        # Project to output dimension
        final_dim = int(embed_dim * 2 ** (self.num_stages - 1))
        self.projection = nn.Linear(final_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            features: (B, seq_len, output_dim) - sequence features for decoder
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.patch_norm(x)
        
        # Process through stages
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        
        # Final normalization
        x = self.norm(x)  # (B, H*W, C)
        
        # Project to output dimension
        x = self.projection(x)  # (B, seq_len, output_dim)
        
        return x


class MBartDecoderLayer(nn.Module):
    """
    mBART decoder layer with pre-layer normalization
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Pre-LN: normalize before attention
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Self attention with pre-LN
        residual = tgt
        tgt = self.self_attn_layer_norm(tgt)
        tgt, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, 
                                key_padding_mask=tgt_key_padding_mask)
        tgt = self.dropout1(tgt)
        tgt = residual + tgt
        
        # Cross attention with pre-LN
        residual = tgt
        tgt = self.encoder_attn_layer_norm(tgt)
        tgt, _ = self.encoder_attn(tgt, memory, memory)
        tgt = self.dropout2(tgt)
        tgt = residual + tgt
        
        # FFN with pre-LN
        residual = tgt
        tgt = self.final_layer_norm(tgt)
        tgt = self.fc2(self.dropout3(self.activation(self.fc1(tgt))))
        tgt = self.dropout4(tgt)
        tgt = residual + tgt
        
        return tgt


class MBartDecoder(nn.Module):
    """
    mBART-style decoder for multilingual OCR
    Uses pre-layer normalization and denoising capabilities
    """
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=6,
                 dim_feedforward=3072, dropout=0.1, max_seq_len=50):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.embed_scale = math.sqrt(d_model)
        
        # Positional encoding (learned, like mBART)
        self.embed_positions = nn.Embedding(max_seq_len, d_model)
        
        # Decoder layers with pre-LN
        self.layers = nn.ModuleList([
            MBartDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm (pre-LN architecture)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and output projection (like mBART)
        self.output_projection.weight = self.embed_tokens.weight
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            tgt: (B, T) target token indices
            memory: (B, S, D) encoder output
            tgt_mask: (T, T) causal attention mask
            tgt_key_padding_mask: (B, T) padding mask
        Returns:
            logits: (B, T, V)
        """
        # Token embedding with scaling
        positions = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        x = self.embed_tokens(tgt) * self.embed_scale + self.embed_positions(positions)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits


class CTCHead(nn.Module):
    """CTC head for auxiliary training"""
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(input_dim, vocab_size)
        
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=-1)


class ViLanOCR_Hindi(nn.Module):
    """
    Vision Language OCR for Hindi with True Swin Transformer + mBART
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=768, max_seq_len=50,
                 img_height=64, img_width=256, use_ctc=True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_ctc = use_ctc
        
        # True Swin Transformer encoder with shifted window attention
        # CHANGED: window_size from 7 to 4 to match feature map dimensions
        # CHANGED: Using 3 stages instead of 4 (depths=[2,2,6] instead of [2,2,6,2])
        # Reason: With img_height=64, after 3 downsampling stages we get H=4, 
        # which still works with window_size=4. A 4th stage would give H=2 < window_size=4
        self.encoder = SwinTransformerEncoder(
            img_size=(img_height, img_width),
            patch_size=4,
            in_chans=1,
            embed_dim=96,
            depths=[2, 2, 6],  # 3 stages instead of 4
            num_heads=[3, 6, 12],  # Matching number of stages
            window_size=4,  # Changed from 7 to 4
            mlp_ratio=4.,
            output_dim=d_model
        )
        
        # mBART decoder with pre-layer normalization
        self.decoder = MBartDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=8,
            num_layers=6,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            max_seq_len=max_seq_len
        )
        
        # CTC head for auxiliary training
        if use_ctc:
            self.ctc_head = CTCHead(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, images, tgt=None, tgt_padding_mask=None, return_ctc=False):
        # Encode with Swin Transformer
        encoder_output = self.encoder(images)
        
        # CTC prediction
        ctc_logits = None
        if self.use_ctc and return_ctc:
            ctc_logits = self.ctc_head(encoder_output)
        
        # Decoder prediction
        if tgt is not None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            logits = self.decoder(tgt, encoder_output, tgt_mask, tgt_padding_mask)
            
            if return_ctc:
                return logits, ctc_logits
            return logits
        
        return encoder_output
    
    def decode_greedy(self, images, max_len=50):
        """Greedy decoding"""
        self.eval()
        with torch.no_grad():
            B = images.size(0)
            device = images.device
            
            memory = self.encoder(images)
            tgt = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)
            
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            outputs = [[] for _ in range(B)]
            
            for _ in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
                logits = self.decoder(tgt, memory, tgt_mask=tgt_mask)
                preds = logits[:, -1, :].argmax(dim=-1)
                
                for i in range(B):
                    if not finished[i]:
                        if preds[i].item() == EOS_IDX:
                            finished[i] = True
                        else:
                            outputs[i].append(preds[i].item())
                
                if finished.all():
                    break
                
                tgt = torch.cat([tgt, preds.unsqueeze(1)], dim=1)
            
            return outputs
    
    def decode_beam_search(self, images, beam_width=5, max_len=50, length_penalty=0.6):
        """Beam search decoding"""
        self.eval()
        with torch.no_grad():
            B = images.size(0)
            device = images.device
            memory = self.encoder(images)
            results = []
            
            for b in range(B):
                memory_b = memory[b:b+1]
                beams = [([SOS_IDX], 0.0)]
                
                for step in range(max_len):
                    candidates = []
                    
                    for seq, score in beams:
                        if seq[-1] == EOS_IDX:
                            lp = ((5 + len(seq)) / 6) ** length_penalty
                            candidates.append((seq, score / lp))
                            continue
                        
                        tgt = torch.tensor([seq], dtype=torch.long, device=device)
                        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
                        logits = self.decoder(tgt, memory_b, tgt_mask=tgt_mask)
                        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                        top_k_probs, top_k_indices = torch.topk(log_probs[0], beam_width)
                        
                        for prob, idx in zip(top_k_probs, top_k_indices):
                            new_seq = seq + [idx.item()]
                            new_score = score + prob.item()
                            candidates.append((new_seq, new_score))
                    
                    beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                    
                    if all(seq[-1] == EOS_IDX for seq, _ in beams):
                        break
                
                best_seq = beams[0][0][1:]
                if best_seq and best_seq[-1] == EOS_IDX:
                    best_seq = best_seq[:-1]
                
                results.append(best_seq)
            
            return results


class CursiveAugmentedDataset(Dataset):
    """Dataset with augmentations for cursive handwriting"""
    def __init__(self, image_paths, labels, img_height=64, img_width=256, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        
        self.base_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.Grayscale(num_output_channels=1),
        ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def augment_image(self, img):
        """Apply augmentations for cursive text"""
        if random.random() > 0.5:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, fillcolor=255)
        
        if random.random() > 0.5:
            shear = random.uniform(-0.1, 0.1)
            img = img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=255)
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        return img
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            img = Image.open(img_path).convert('L')
            img = self.base_transform(img)
            
            if self.augment:
                img = self.augment_image(img)
            
            img = self.normalize(img)
            
            label = self.labels[idx]
            label_indices = [SOS_IDX] + [char2idx.get(c, char2idx[UNK_TOKEN]) for c in label] + [EOS_IDX]
            label_tensor = torch.tensor(label_indices, dtype=torch.long)
            
            return img, label_tensor
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            img = torch.zeros((1, self.img_height, self.img_width))
            label_tensor = torch.tensor([SOS_IDX, EOS_IDX], dtype=torch.long)
            return img, label_tensor


def collate_fn(batch):
    """Custom collate function to pad sequences"""
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    
    label_lengths = [len(l) for l in labels]
    max_len = max(label_lengths)
    
    padded_labels = torch.full((len(labels), max_len), PAD_IDX, dtype=torch.long)
    padding_mask = torch.ones((len(labels), max_len), dtype=torch.bool)
    
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
        padding_mask[i, :len(label)] = False
    
    return images, padded_labels, padding_mask


class ViLanOCRTrainer:
    """Trainer for ViLanOCR model"""
    def __init__(self, model, device='cuda', label_smoothing=0.1, ctc_weight=0.3):
        self.model = model.to(device)
        self.device = device
        self.ctc_weight = ctc_weight
        
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)
        
        if model.use_ctc:
            self.ctc_criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_ctc_loss = 0
        num_batches = 0
        
        print(f"  Training on {len(dataloader)} batches...")
        
        for batch_idx, (images, labels, padding_mask) in enumerate(dataloader):
            print(f"    Batch {batch_idx+1}/{len(dataloader)} - Processing...", end='\r')
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            padding_mask = padding_mask.to(self.device)
            
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]
            tgt_padding_mask = padding_mask[:, :-1]
            
            if self.model.use_ctc:
                logits, ctc_logits = self.model(images, tgt_input, tgt_padding_mask=tgt_padding_mask, return_ctc=True)
                
                ce_loss = self.ce_criterion(logits.reshape(-1, self.model.vocab_size), tgt_output.reshape(-1))
                
                log_probs = ctc_logits.permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=self.device)
                
                ctc_targets = []
                ctc_target_lengths = []
                for label in labels:
                    target = [idx for idx in label.tolist() if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]]
                    ctc_targets.extend(target)
                    ctc_target_lengths.append(len(target))
                
                ctc_targets = torch.tensor(ctc_targets, dtype=torch.long, device=self.device)
                ctc_target_lengths = torch.tensor(ctc_target_lengths, dtype=torch.long, device=self.device)
                
                try:
                    ctc_loss = self.ctc_criterion(log_probs, ctc_targets, input_lengths, ctc_target_lengths)
                except:
                    ctc_loss = torch.tensor(0.0, device=self.device)
                
                loss = (1 - self.ctc_weight) * ce_loss + self.ctc_weight * ctc_loss
                total_ctc_loss += ctc_loss.item()
            else:
                logits = self.model(images, tgt_input, tgt_padding_mask=tgt_padding_mask)
                loss = self.ce_criterion(logits.reshape(-1, self.model.vocab_size), tgt_output.reshape(-1))
                ce_loss = loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss_so_far = total_loss / num_batches
                print(f"    Batch {batch_idx+1}/{len(dataloader)} - Loss: {avg_loss_so_far:.4f}    ")
        
        print(f"    Batch {num_batches}/{len(dataloader)} - Complete!                    ")
        return (total_loss / num_batches, total_ce_loss / num_batches, 
                total_ctc_loss / num_batches if self.model.use_ctc else 0)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct_chars = 0
        total_chars = 0
        total_cer_distance = 0
        total_cer_chars = 0
        
        with torch.no_grad():
            for images, labels, padding_mask in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                padding_mask = padding_mask.to(self.device)
                
                tgt_input = labels[:, :-1]
                tgt_output = labels[:, 1:]
                tgt_padding_mask = padding_mask[:, :-1]
                
                logits = self.model(images, tgt_input, tgt_padding_mask=tgt_padding_mask)
                loss = self.ce_criterion(logits.reshape(-1, self.model.vocab_size), tgt_output.reshape(-1))
                total_loss += loss.item()
                
                preds = logits.argmax(dim=-1)
                mask = ~padding_mask[:, 1:]
                correct_chars += ((preds == tgt_output) & mask).sum().item()
                total_chars += mask.sum().item()
                
                for i in range(preds.size(0)):
                    pred_seq = preds[i][mask[i]].cpu().tolist()
                    gt_seq = tgt_output[i][mask[i]].cpu().tolist()
                    
                    m, n = len(pred_seq), len(gt_seq)
                    dp = [[0] * (n + 1) for _ in range(m + 1)]
                    
                    for j in range(m + 1):
                        dp[j][0] = j
                    for j in range(n + 1):
                        dp[0][j] = j
                    
                    for j in range(1, m + 1):
                        for k in range(1, n + 1):
                            if pred_seq[j-1] == gt_seq[k-1]:
                                dp[j][k] = dp[j-1][k-1]
                            else:
                                dp[j][k] = 1 + min(dp[j-1][k], dp[j][k-1], dp[j-1][k-1])
                    
                    total_cer_distance += dp[m][n]
                    total_cer_chars += n
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_chars / total_chars if total_chars > 0 else 0
        cer = (total_cer_distance / total_cer_chars * 100) if total_cer_chars > 0 else 0
        
        return avg_loss, accuracy, cer


def decode_prediction(indices):
    """Convert predicted indices to string"""
    chars = [idx2char[idx] for idx in indices if idx not in [PAD_IDX, SOS_IDX, EOS_IDX, BLANK_IDX]]
    return ''.join(chars)


def predict_image(model, image_path, device='cpu', use_beam_search=True):
    """Predict text from a single image"""
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    if use_beam_search:
        predictions = model.decode_beam_search(img_tensor, beam_width=5, max_len=50)
    else:
        predictions = model.decode_greedy(img_tensor, max_len=50)
    
    text = decode_prediction(predictions[0])
    return text


if __name__ == "__main__":
    DATASET_ROOT = "HindiSeg"
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
        print("Using Apple Silicon GPU (MPS)")
    
    print("="*60)
    print("ViLanOCR: Hindi Handwriting Recognition")
    print("Architecture: TRUE Swin Transformer + mBART Decoder + CTC")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_images, train_labels = load_ground_truth(os.path.join(DATASET_ROOT, "train.txt"), DATASET_ROOT)
    print(f"Train samples: {len(train_images)}")
    
    val_images, val_labels = load_ground_truth(os.path.join(DATASET_ROOT, "val.txt"), DATASET_ROOT)
    print(f"Validation samples: {len(val_images)}")
    
    test_images, test_labels = load_ground_truth(os.path.join(DATASET_ROOT, "test.txt"), DATASET_ROOT)
    print(f"Test samples: {len(test_images)}")
    
    # Create datasets
    train_dataset = CursiveAugmentedDataset(train_images, train_labels, augment=True)
    val_dataset = CursiveAugmentedDataset(val_images, val_labels, augment=False)
    test_dataset = CursiveAugmentedDataset(test_images, test_labels, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Initialize model
    print("\nInitializing ViLanOCR with TRUE Swin Transformer + mBART...")
    model = ViLanOCR_Hindi(vocab_size=VOCAB_SIZE, d_model=768, max_seq_len=50, img_height=64, img_width=256, use_ctc=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ViLanOCRTrainer(model, device=DEVICE, label_smoothing=0.1, ctc_weight=0.3)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-9)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE * 10, epochs=NUM_EPOCHS,
                                                     steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos')
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_cer = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-"*60)
        
        train_loss, train_ce_loss, train_ctc_loss = trainer.train_epoch(train_loader, optimizer)
        
        for _ in range(len(train_loader)):
            scheduler.step()
        
        val_loss, val_acc, val_cer = trainer.validate(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        is_best_acc = val_acc > best_val_acc
        is_best_cer = val_cer < best_val_cer
        is_best_loss = val_loss < best_val_loss
        
        if is_best_acc:
            best_val_acc = val_acc
        if is_best_cer:
            best_val_cer = val_cer
        if is_best_loss:
            best_val_loss = val_loss
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary")
        print(f"{'='*60}")
        print(f"Learning Rate:     {current_lr:.6f}")
        print(f"Train Loss:        {train_loss:.4f}")
        print(f"  - CE Loss:       {train_ce_loss:.4f}")
        print(f"  - CTC Loss:      {train_ctc_loss:.4f}")
        print(f"Val Loss:          {val_loss:.4f} {'✓ BEST' if is_best_loss else ''}")
        print(f"Val Accuracy:      {val_acc:.4f} ({val_acc*100:.2f}%) {'✓ BEST' if is_best_acc else ''}")
        print(f"Val CER:           {val_cer:.2f}% {'✓ BEST' if is_best_cer else ''}")
        
        if is_best_acc:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                       'val_acc': val_acc, 'val_loss': val_loss, 'val_cer': val_cer}, 'best_vilanocr_swin_mbart.pth')
            print(f"✓ Model checkpoint saved!")
        
        print(f"{'='*60}")
    
    print("\nTraining Completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best Validation CER: {best_val_cer:.2f}%")