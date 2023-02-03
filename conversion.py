import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from pathlib import Path 

ckpt_path = '/disk/autovc/checkpoint/autovc_100spk_ssim_spkmean_adamw_160/G.ckpt'
save_path = ''
def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def chunking_mel(melspectrogram, base = 128):
    data = []
    num_spectro = (melspectrogram.shape[0]//base)+1
    print('num_spectro: ', num_spectro)
    for index in range(num_spectro):
        if index < num_spectro - 1:
            mel = melspectrogram[index*base:index*base+base,:]
            print('mel: ', mel.shape)
        else:
            mel = melspectrogram[index*base:, :]
            len_pad = base - melspectrogram.shape[0]%base        
            mel = np.pad(mel, ((0, len_pad), (0,0)), 'constant', constant_values=(0,0))
            print('last mel shape: ', mel.shape)
        data.append(mel)

    return torch.tensor(np.array(data)), len_pad


device = 'cuda:0'
G = Generator(32,256,512,32).eval().to(device)

g_checkpoint = torch.load(ckpt_path)
G.load_state_dict(g_checkpoint)

spect_vc = []

src_speaker_lst = ["0af00UcTOSc", "FD5ZKiSmoMU",  "0akiEFwtkyA", "pcNxS2i7SvQ"]
tgt_speaker_lst = ["0af00UcTOSc", "FD5ZKiSmoMU",  "0akiEFwtkyA", "pcNxS2i7SvQ", "0wpCZxiAQzw",  "1gdKrtwBGqY", "11Mq9ZuxZMc", "F2hc2FLOdhI", "oXSyMUeAEec", "Yo5cKRmJaf0"]#['0d6iSvF1UmA', '0C5UQbWzwg8', '0akiEFwtkyA', '0af00UcTOSc']#["2ZviHInGBJQ", "qykSnLkPM7E", "16cMSRFid9U", "zvvZEIGNLo0",  "0nI65jgHG9o", "VStEmE85QDE", "0tqq66zwa7g"]


for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        try:
            src_speaker_mel = np.load(os.path.join('/disk/data/lrs3/lrs3_16k_spmel_all_160/',src_speaker, src_speaker+'-00001.npy'))
        except:
            src_speaker_mel = np.load(os.path.join('/disk/data/lrs3/lrs3_16k_spmel_all_160/',src_speaker, src_speaker+'-00002.npy'))
        try:
            src_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_no_meam/',src_speaker, src_speaker+'-00001.npy'))
        except:
            src_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_no_meam/',src_speaker, src_speaker+'-00002.npy'))
        try:    
            tgt_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_no_meam/', tgt_speaker, tgt_speaker+'-00001.npy')) 
        except:
            tgt_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_no_meam/', tgt_speaker, tgt_speaker+'-00002.npy'))

        print('src_speaker_mel',src_speaker_mel.shape)
        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.to(device)       
        src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    
        print('src_speaker_mel',src_speaker_mel.shape)
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
        x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
        print(x_identic_psnt.shape)            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
        print(uttr_trg.shape)
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))
        
Path(save_path).mkdir(parents=True, exist_ok=True)        
with open(save_path+'/results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)
