Hindi OCR- Transformer based modal


Overview:
Progressive deep learning architectures for Hindi handwriting recognition using Devanagari script. The project implements three models with increasing complexity, achieving state-of-the-art performance on cursive Hindi text.

Dependencies:

         bashpip install torch torchvision
         pip install pillow numpy
         pip install einops


Requirements:

         - Python 3.8+
         - CUDA-enabled GPU (recommended) or CPU
         - 8GB+ RAM

 Dataset Structure

         HindiSeg/
        ├── train.txt          # Training image paths + labels
        ├── val.txt            # Validation image paths + labels
        ├── test.txt           # Test image paths + labels
        └── images/
                |-test
                |-train    # Image files
                |-val

Models
1. CNN-Transformer Baseline (modal3.py)
         
         Architecture: CNN encoder + Transformer decoder
         Features: Basic attention mechanism, CTC loss


3. Swin-mBART (modal2.py)

         Architecture: Swin Transformer + mBART decoder
         Features: Shifted window attention, hierarchical vision features, pre-layer normalization
         Improvement: Better spatial feature extraction, 5-10% accuracy gain


4. Swin-BiLSTM-mBART (modal1.py) 

         Architecture: Swin Transformer + BiLSTM + mBART decoder
         Features:
                  Shifted window self-attention
                  Bidirectional LSTM for temporal modeling
                  mBART multilingual decoder
                  Mixed precision training (AMP)



Training Configuration

         Batch Size: 8
         Image Size: 64×256 (H×W)
         Vocabulary: 79 characters (Devanagari vowels, consonants, matras + special tokens)
         Loss Function: Combined Cross-Entropy (70%) + CTC (30%)
         Optimizer: AdamW (lr=0.0001, weight_decay=0.01)
         Scheduler: OneCycleLR with cosine annealing
Augmentation:

         Random rotation (±3°)
         Shear transformation
         Brightness/contrast adjustment
         Gaussian blur (simulates poor quality)




Training:
         
         bash# Train best model (Swin-BiLSTM-mBART)
         python modal1.py

# Train Swin-mBART
         python modal2.py

# Train CNN-Transformer baseline
         python modal3.py
         


# Load trained model
         model = ViLanOCR_Hindi()
         checkpoint = torch.load('best_vilanocr_enhanced.pth')
         model.load_state_dict(checkpoint['model_state_dict'])

# Predict
         text = predict_image(model, 'test_image.jpg', device='cuda', use_beam_search=True)
         print(f"Predicted: {text}")
         Results

Inference

         Best Validation Accuracy: 97.57%
         Character Error Rate (CER): 3.21%
         Supports: Vowels, consonants, matras, conjuncts (क्ष, त्र, ज्ञ)

Model Progression
         
         Baseline → Standard CNN-Transformer approach
         Swin-mBART → Hierarchical vision features with shifted windows
         Swin-BiLSTM-mBART → Added temporal context for better cursive recognition

Key Features

         Multi-stage hierarchical feature extraction
         Relative position bias in attention
         Label smoothing for regularization
         Beam search decoding (width=5)
         Gradient clipping for stability
         Support for Apple Silicon (MPS) and CUDA
