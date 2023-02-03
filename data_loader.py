from torch.utils import data
import torch
import numpy as np
import pickle 
import os    
from glob import glob
from multiprocessing import Process, Manager   
from scipy.fft import idctn, dctn

# change your directory first
spk_txt = '100_lrs3_shuffle.txt'
mel_path = 'data/lrs3_22feature_500/mel/'
face_path = 'data/faceemb_lrs3_mtcnn_margin50_500_mean/'

class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, len_crop):
        """Initialize and preprocess the Utterances dataset."""
        self.len_crop = len_crop
        
        self.data_dict_lrs3 = self.make_data_dict_lrs3()
        self.speaker_lst_lrs3 = list(self.data_dict_lrs3.keys())
        print(len(self.speaker_lst_lrs3))
        
        print('Finished init the dataset...')

    def make_data_dict_lrs3(self):
        with open(spk_txt, 'r') as f:
            lines = f.readlines()
            data = {}
            for line in lines:
                speaker = line.replace('\n', '' )
                if speaker not in list(data.keys()):
                    data[speaker] = []
                files = glob(os.path.join(mel_path, speaker + '*.npy'))
                for file in files:
                    idx = file.split('/')[-1]
                    data[speaker].append(idx)
                if len(data[speaker]) < 1:
                    print(speaker)

        return data
        
    def __getitem__(self, index):
      # pick a random speaker
      # dataset = self.train_dataset 
      
      root_speaker = face_path
      root_mel = mel_path
      speaker = self.speaker_lst_lrs3[index]

      # pick random uttr with random crop
      a = np.random.randint(0, len(self.data_dict_lrs3[speaker]))
      emb_org = np.load(os.path.join(root_speaker, speaker, self.data_dict_lrs3[speaker][a]).replace('.wav', ''))
      tmp = np.load(os.path.join(root_mel, self.data_dict_lrs3[speaker][a]))
      tmp = tmp.transpose()
      # extract sp
      m = np.ones(tmp.shape)
      m[:, 10:]=0
      sp = dctn(idctn(tmp)*m)

      if tmp.shape[0] < self.len_crop:
          len_pad = self.len_crop - tmp.shape[0]
          uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
          sp = np.pad(sp, ((0,len_pad),(0,0)), 'constant')
      elif tmp.shape[0] > self.len_crop:
          left = np.random.randint(tmp.shape[0]-self.len_crop)
          uttr = tmp[left:left+self.len_crop, :]
          sp = sp[left:left+self.len_crop, :]
      else:
          uttr = tmp
          sp = sp


      while True:
        index_b = np.random.randint(0, len(self.speaker_lst_lrs3))
        if index != index_b:
            break

      speaker_b = self.speaker_lst_lrs3[index_b]
      b = np.random.randint(0, len(self.data_dict_lrs3[speaker_b]))
      emb_tgt = np.load(os.path.join(root_speaker, speaker_b, self.data_dict_lrs3[speaker_b][b].replace('.wav', '')))
      tmpb = np.load(os.path.join(root_mel, self.data_dict_lrs3[speaker_b][b]))
      tmpb = tmpb.transpose()

      if tmpb.shape[0] < self.len_crop:
          len_padb = self.len_crop - tmpb.shape[0]
          uttrb = np.pad(tmpb, ((0,len_padb),(0,0)), 'constant')
      elif tmpb.shape[0] > self.len_crop:
          leftb = np.random.randint(tmpb.shape[0]-self.len_crop)
          uttrb = tmpb[leftb:leftb+self.len_crop, :]
      else:
          uttrb = tmpb

      return uttr, emb_org, emb_tgt, index, index_b, sp, uttrb


    def __len__(self):
        """Return the number of spkrs."""
        return len(self.speaker_lst_lrs3)
    
    
    

def get_loader(batch_size=16, len_crop=128, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(len_crop)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader






