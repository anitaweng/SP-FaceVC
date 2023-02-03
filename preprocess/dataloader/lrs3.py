import os
import logging
import numpy as np
import re
from .base import BaseDataset
from util.transform import segment, random_scale
import pyworld as pw
import soundfile as sf
from glob import glob
import random
from scipy.fft import idctn, dctn
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)

class Dataset(BaseDataset):
    def __init__(self, dset, indexes_path, feat, feat_path, seglen, pitch_path, pitchlen, ap_path, face_path, sp_path, njobs, metadata):
        super().__init__(dset, indexes_path, feat, feat_path, seglen, pitch_path, pitchlen, ap_path, face_path, sp_path, njobs, metadata)
        self.spk_dic = {}
        with open('/home/310505006/data/500_lrs3_shuffle.txt','r') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                self.spk_dic[line.replace('\n','')] = i

    def sub_process(self, each_data, feat, feat_path, pitch_path, ap_path, face_path, sp_path):
        speaker = each_data[0]
        basename = each_data[1]
        #print(each_data, feat, feat_path)
        ret = {}
        for f in feat:
            path = os.path.join(feat_path, f, basename)
            if os.path.isfile(path):
                #print(path)
                ret[f] = np.load(path)
                #ret['pitch'] = np.load(os.path.join(pitch_path, speaker, ''.join(basename.split('.wav'))))
                #data, fs = sf.read(os.path.join(ap_path, speaker, ''.join(basename.split('.npy'))))
                #_, _, ret['ap'] = pw.wav2world(data, fs)
                #ret['ap'] = np.flipud(np.load(os.path.join(ap_path, speaker, ''.join(basename.split('.wav'))))).copy()
                #ret['sp'] = np.flipud(np.load(os.path.join(sp_path, speaker, ''.join(basename.split('.wav'))))).copy()
                m = np.ones(ret[f].shape)
                m[10:, :]=0
                ret['sp'] = idctn(dctn(ret[f])*m)
                #print(ret[f].shape, ret['ap'].shape)
                face_lst = glob(os.path.join(face_path, speaker, basename.split('-')[-1].replace('.wav.npy', ''), '*.npy'))
                #file = random.choice(face_lst)
                '''tmp_face_lst = []
                for file in face_lst:
                    #print(os.path.join(face_path, speaker, basename.split('-')[-1].replace('.wav.npy', ''), file))
                    n = np.load(os.path.join(face_path, speaker, basename.split('-')[-1].replace('.wav.npy', ''), file))
                    tmp_face_lst.append(n)
                if not tmp_face_lst:
                    print(path)
                ret['face'] = np.mean(tmp_face_lst, axis=0)'''
                ret['face'] = np.load(os.path.join(face_path, speaker, basename.replace('.wav', '')))
            else:
                logger.info(f'Skip {path} {f}: invalid file.')
                return
        ret['speaker'] = speaker
        return ret
    
    def __getitem__(self, index):
        #print(self.data[index])
        speaker = self.data[index]['speaker']
        mel = self.data[index]['mel']
        #pitch = self.data[index]['pitch']
        #ap = self.data[index]['ap']
        pitch = None
        ap = None
        face = self.data[index]['face']
        sp = self.data[index]['sp']
        mel, pitch, ap, sp = segment(mel, pitch, ap, sp, return_r=False, seglen=self.seglen, pitchlen=self.pitchlen)
        
        while True:
            indexb = random.randint(0, len(self.data)-1)
            if indexb != index:
                break
                
        '''speakerb = self.data[indexb]['speaker']
        melb = self.data[indexb]['mel']
        pitchb = self.data[indexb]['pitch']
        apb = self.data[indexb]['ap']
        melb, pitchb, apb = segment(melb, pitchb, apb, return_r=False, seglen=self.seglen, pitchlen=self.pitchlen)
        '''
        faceb = self.data[indexb]['face']
        speakerb = self.data[indexb]['speaker']
        
        meta = {
            'speaker' : speaker,
            'mel': mel,
            #'pitch' : pitch,
            'face' : face,
            'faceb' : faceb,
            'sp':sp,
            'speakerb':speakerb,
            'label':F.one_hot(torch.tensor(self.spk_dic[speaker]), num_classes=500),
            'labelb':F.one_hot(torch.tensor(self.spk_dic[speakerb]), num_classes=500),   
        }
        return meta