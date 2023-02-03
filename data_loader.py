from torch.utils import data
import torch
import numpy as np
import pickle 
import os    
from glob import glob
from multiprocessing import Process, Manager   
from scipy.fft import idctn, dctn

class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, len_crop, sv2ttsemb):
        """Initialize and preprocess the Utterances dataset."""
        self.len_crop = len_crop
        self.sv2ttsemb = sv2ttsemb
        
        '''self.data_dict = self.make_data_dict()
        self.speaker_lst = list(self.data_dict.keys())
        print(len(self.speaker_lst))'''
        
        self.data_dict_lrs3 = self.make_data_dict_lrs3()
        self.speaker_lst_lrs3 = list(self.data_dict_lrs3.keys())
        print(len(self.speaker_lst_lrs3))
        
        print('Finished init the dataset...')

            
    def make_data_dict(self):
        '''with open('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/vctk_20spk_lst.txt', 'r') as f:
            lines = f.readlines()
            data = {}
            for line in lines:
                file = line.replace('.npy\n', '')
                speaker = file.split('/')[-2]
                idx = file.split('/')[-1]
                
                if speaker not in list(data.keys()):
                    data[speaker] = []
                data[speaker].append(idx)
        return data'''
        with open('/disk/data/VCTK-Corpus/vctk_allspk_lst.txt', 'r') as f:
            lines = f.readlines()
            data = {}
            for i, line in enumerate(lines):
                speaker = line.replace('\n', '' )
                if speaker not in list(data.keys()) and i != len(lines)-1:
                    data[speaker] = []
                elif i == len(lines)-1:
                    break
                files = glob(os.path.join('/disk/data/VCTK-Corpus/spmel16/', speaker, '*.npy'))
                for file in files:
                    idx = file.split('/')[-1]
                    data[speaker].append(idx)

        return data
        
    def make_data_dict_lrs3(self):
        '''with open('../faceVC_ge2egrad_cddiff/20sp_train.txt', 'r') as f:
            lines = f.readlines()
            data = {}
            for line in lines:
                file = line.replace('.mp4\n', '')
                speaker = file.split('-')[0]
                idx = file.split('-')[1]
                
                if speaker not in list(data.keys()):
                    data[speaker] = []
                data[speaker].append(idx)
        return data'''
        with open('/home/310505006/data/100_lrs3_shuffle.txt', 'r') as f:
            lines = f.readlines()
            data = {}
            for line in lines:
                speaker = line.replace('\n', '' )
                if speaker not in list(data.keys()):
                    data[speaker] = []
                files = glob(os.path.join('/home/310505006/data/lrs3_22feature_500/mel/', speaker + '*.npy'))
                for file in files:
                    idx = file.split('/')[-1]
                    data[speaker].append(idx)
                if len(data[speaker]) < 1:
                    print(speaker)

        return data
        
    def __getitem__(self, index):
        # pick a random speaker
        # dataset = self.train_dataset 
        #sv2ttsemb = self.sv2ttsemb
        if self.sv2ttsemb:
            root_speaker = '/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/sv2tts_embeds16/'
            root_mel = '/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spmel16/'
            speaker = self.speaker_lst[index]
            
            # pick random uttr with random crop
            while True:
                try:
                    a = np.random.randint(0, len(self.data_dict[speaker]))
                    emb_org = np.load(root_speaker+'embed-'+self.data_dict[speaker][a]+'.npy')
                    tmp = np.load(root_mel+speaker+'/'+self.data_dict[speaker][a]+'.npy')
                    break
                except:
                    print(root_speaker+'embed-'+self.data_dict[speaker][a]+'.npy')
                    print(root_mel+speaker+'/'+self.data_dict[speaker][a]+'.npy')

            if tmp.shape[0] < self.len_crop:
                len_pad = self.len_crop - tmp.shape[0]
                uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
            elif tmp.shape[0] > self.len_crop:
                left = np.random.randint(tmp.shape[0]-self.len_crop)
                uttr = tmp[left:left+self.len_crop, :]
            else:
                uttr = tmp
            
            return uttr, emb_org

        else:
            dataset = np.random.randint(1, 10)
            if dataset < 0:#vctk
              root_speaker = '/disk/data/VCTK-Corpus/spk_emb16/'
              root_mel = '/disk/data/VCTK-Corpus/spmel16/'
              speaker = self.speaker_lst[index]
              
              # pick random uttr with random crop
              a = np.random.randint(0, len(self.data_dict[speaker]))
              #emb_org = np.load(root_speaker+speaker+'.npy')
              #tmp = np.load(root_mel+speaker+'/'+speaker+'-'+self.data_dict_lrs3[speaker][a]+'.npy')
              #emb_org = np.load(os.path.join(root_speaker, speaker, self.data_dict_lrs3[speaker][a]))
              emb_org = np.load(os.path.join(root_speaker, speaker+'.npy'))
              tmp = np.load(os.path.join(root_mel, speaker, self.data_dict[speaker][a]))
      
              if tmp.shape[0] < self.len_crop:
                  len_pad = self.len_crop - tmp.shape[0]
                  uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
              elif tmp.shape[0] > self.len_crop:
                  left = np.random.randint(tmp.shape[0]-self.len_crop)
                  uttr = tmp[left:left+self.len_crop, :]
              else:
                  uttr = tmp

              while True:
                index_b = np.random.randint(0, len(self.speaker_lst))
                if index != index_b:
                    break

              speaker_b = self.speaker_lst[index_b]
              b = np.random.randint(0, len(self.data_dict[speaker_b]))
              #emb_tgt = np.load(os.path.join(root_speaker, speaker_b, self.data_dict_lrs3[speaker_b][b]))
              emb_tgt = np.load(os.path.join(root_speaker, speaker_b+'.npy'))


              return uttr, emb_org, emb_tgt, index, index_b
              
            else:#lrs3
              #root_speaker = '/disk/data/lrs3/lrs3_speechbrain_spkemb22_500_mean/'
              root_speaker = '/home/310505006/data/faceemb_lrs3_mtcnn_margin50_500_mean/'
              root_mel = '/home/310505006/data/lrs3_22feature_500/mel/'
              speaker = self.speaker_lst_lrs3[index]
              
              # pick random uttr with random crop
              a = np.random.randint(0, len(self.data_dict_lrs3[speaker]))
              #emb_org = np.load(root_speaker+speaker+'.npy')
              #tmp = np.load(root_mel+speaker+'/'+speaker+'-'+self.data_dict_lrs3[speaker][a]+'.npy')
              emb_org = np.load(os.path.join(root_speaker, speaker, self.data_dict_lrs3[speaker][a]).replace('.wav', ''))
              #emb_org = np.load(os.path.join(root_speaker, speaker+'.npy'))
              '''flst = glob(os.path.join(root_speaker, speaker, self.data_dict_lrs3[speaker][a].split('-')[-1].split('.')[0], '*.npy'))
              tmp_flst = []
              for f in flst:
                n = np.load(os.path.join(root_speaker, speaker, self.data_dict_lrs3[speaker][a].split('-')[-1].split('.')[0], f))
                tmp_flst.append(n)
              emb_org = np.mean(tmp_flst, axis=0)'''
              #tmp = np.load(os.path.join(root_mel, speaker, self.data_dict_lrs3[speaker][a]))
              tmp = np.load(os.path.join(root_mel, self.data_dict_lrs3[speaker][a]))
              tmp = tmp.transpose()
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
              #emb_tgt = np.load(os.path.join(root_speaker, speaker_b+'.npy'))
              '''flstb = glob(os.path.join(root_speaker, speaker_b, self.data_dict_lrs3[speaker_b][b].split('-')[-1].split('.')[0], '*.npy'))
              tmp_flstb = []
              for fb in flstb:
                nb = np.load(os.path.join(root_speaker, speaker_b, self.data_dict_lrs3[speaker_b][b].split('-')[-1].split('.')[0], fb))
                tmp_flstb.append(nb)
              emb_tgt = np.mean(tmp_flstb, axis=0)'''
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
        #return len(self.speaker_lst) 
    
    
    

def get_loader(batch_size=16, len_crop=128, sv2ttsemb = False, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(len_crop, sv2ttsemb)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader






