import os
 
# Set the directory you want to start from
rootDir = '.'
count = 0
for dirName, subdirList, fileList in os.walk(rootDir):
    if(dirName == '.'):
        continue
    fid = open(dirName+ '//' + dirName[2:] + '.txt','r')
    ss = fid.readlines()
    for line in xrange(0,len(ss)):
        temp = ss[line].strip()
        if(temp == ''):
            print dirName
            print line+1
            count = count +1
    fid.close()
print count
