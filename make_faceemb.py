from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
import torch
from torchvision import transforms
import numpy as np
import os
from pathlib import Path

device='cuda'
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(margin=50, select_largest=False, device='cuda')

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
rootDir = '/disk/data/preprocess_video_train/'
targetDir = '/disk/data/lrs3/faceemb_lrs3_mtcnn_margin50/'

def extract(file):
    while True:
        img = cv2.imread(file)
        data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = Image.fromarray(data)
        name = file.split('/')[-1].replace('.jpg','')
        img_cropped = mtcnn(data)
        '''if len(name.split('_')) > 1:
            dic = '_'.join(name.split('_')[:-1])
        else:
            dic = name.split('_')[-2]'''
        iddir = file.split('/')[-3]#.split('#')[0]
        idsubdir = file.split('/')[-2]

        if img_cropped is None:
            with open(targetDir+'Failed_face_extract.txt','a') as f:
                #print(os.path.join(iddir, idsubdir, name))
                f.write(os.path.join(iddir, idsubdir, name)+'\n')
            break
        else:
            img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
        
        Path(targetDir+iddir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(targetDir, iddir, idsubdir)).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(targetDir, iddir, idsubdir, name), img_embedding.squeeze(0).cpu().detach().numpy())
        break


dirName, subdirList, _ = next(os.walk(rootDir))
for subdir in sorted(subdirList):
    #print(os.path.join(dirName,subdir))
    subdirName, subsubdirList, _ = next(os.walk(os.path.join(dirName,subdir)))
    for subsubdir in sorted(subsubdirList):
        _, _, fileList = next(os.walk(os.path.join(subdirName,subsubdir)))
        for fileName in sorted(fileList):
            #print(os.path.join(subdirName,subsubdir,fileName))
            path = os.path.join(subdirName,subsubdir,fileName)
            extract(path)
            #assert 0
            '''try:
                extract(path)
                with open('vox_train_spk_lst','a') as f:
                    f.write(path+'\n')
            except:
                with open('train_except.txt','a') as fout:
                    fout.write(path+'\n')
                print(path+'\n')'''
