import os
import numpy as np
rootDir = '/disk/data/lrs3/faceemb_lrs3_mtcnn_margin50_500/'
out_dir = '/disk/data/lrs3/faceemb_lrs3_mtcnn_margin50_500_mean/'

if not os.path.exists('/disk/data/lrs3/faceemb_lrs3_mtcnn_margin50_500_mean/'):
        os.makedirs('/disk/data/lrs3/faceemb_lrs3_mtcnn_margin50_500_mean/')

dirName, subdirList, _ = next(os.walk(rootDir))

speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    _, subsubdirList, _ = next(os.walk(os.path.join(dirName,speaker)))
    for subdir in subsubdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker, subdir)))
        embs = []
        for s in sorted(fileList):
            n = np.load(os.path.join(dirName, speaker, subdir, s))
            #print(n.shape)
            embs.append(n)
        if not os.path.exists(os.path.join(out_dir, speaker)):
            os.makedirs(os.path.join(out_dir, speaker))
        np.save(os.path.join(out_dir,speaker, speaker+'-'+subdir+'.npy'), np.mean(embs, axis=0))
    #print(np.mean(embs, axis=0).shape)
    #assert 0