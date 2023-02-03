import os
import logging
import numpy as np
import re
from .base import BaseDataset
from util.transform import segment, random_scale

logger = logging.getLogger(__name__)

class Dataset(BaseDataset):
    def __init__(self, dset, indexes_path, feat, feat_path, seglen, pitch_path, pitchlen, ap_path, face_path, njobs, metadata):
        super().__init__(dset, indexes_path, feat, feat_path, seglen, pitch_path, pitchlen, ap_path, face_path, njobs, metadata)

    def sub_process(self, each_data, feat, feat_path, pitch_path, ap_path, face_path):
        speaker = each_data[0]
        basename = each_data[1]
        # print(each_data, feat, feat_path)
        ret = {}
        for f in feat:
            path = os.path.join(feat_path, f, basename)
            if os.path.isfile(path):
                ret[f] = np.load(path)
                ret['pitch'] = np.load(os.path.join(pitch_path, speaker, ''.join(basename.split('.wav'))))
                ret['ap'] = np.load(os.path.join(ap_path, speaker, ''.join(basename.split('.wav'))))
            else:
                logger.info(f'Skip {path} {f}: invalid file.')
                return
        ret['speaker'] = speaker
        return ret
    
    def __getitem__(self, index):
        speaker = self.data[index]['speaker']
        mel = self.data[index]['mel']
        pitch = self.data[index]['pitch']
        ap = self.data[index]['ap']
        
        mel, pitch, ap = segment(mel, pitch, ap, return_r=False, seglen=self.seglen, pitchlen=self.pitchlen)#

        meta = {
            'speaker' : speaker,
            'mel': mel,
            'pitch' : pitch,
            'ap' : ap,
        }
        return meta