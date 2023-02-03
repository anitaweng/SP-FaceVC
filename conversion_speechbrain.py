import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc_gan import Generator
from pathlib import Path 
from scipy.fft import idctn, dctn
#
dir = 'gan_100_mask_neck128'#'spk_200'#'gan_no_dis'#'autovc_waveglow_face_mod1'#
no_repar = False
no_attn = False

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def chunking_mel(melspectrogram, base = 128):
    data = []
    num_spectro = (melspectrogram.shape[0]//base)+1
    #print('num_spectro: ', num_spectro)
    for index in range(num_spectro):
        if index < num_spectro - 1:
            mel = melspectrogram[index*base:index*base+base,:]
            #print('mel: ', mel.shape)
        else:
            mel = melspectrogram[index*base:, :]
            len_pad = base - melspectrogram.shape[0]%base        
            mel = np.pad(mel, ((0, len_pad), (0,0)), 'constant', constant_values=(0,0))
            #print('last mel shape: ', mel.shape)
        data.append(mel)

    return torch.tensor(np.array(data)), len_pad

spk_dir = '/home/310505006/data/faceemb_lrs3_mtcnn_margin50_500_mean/'
device = 'cuda'
G = Generator(128,512,512,32,no_attn, no_repar).eval().to(device)

g_checkpoint = torch.load('checkpoint/'+dir+'/G_1900000.ckpt')
G.load_state_dict(g_checkpoint)
'''model_dict = G.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in g_checkpoint.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
G.load_state_dict(pretrained_dict)'''

spect_vc = []
#############################  LRS3  #################################
mel_dir = '/home/310505006/data/lrs3_22feature_500/mel/'
src_speaker_lst = ["0wpCZxiAQzw", "FD5ZKiSmoMU", "1gdKrtwBGqY", "pcNxS2i7SvQ", "11Mq9ZuxZMc", "F2hc2FLOdhI", "oXSyMUeAEec", "Yo5cKRmJaf0", "weyd0UMdP7g", "LKAhTELkOV8", "2L4BSVpvx1A", "xbagFzcyNiM", "6Af6bSwyiwI", "MgUTzbGakRw", "EZwxKPv1CwA", "5duz42kHqPs"]
#src_speaker_lst = ["0wpCZxiAQzw", "FD5ZKiSmoMU", "1gdKrtwBGqY", "pcNxS2i7SvQ"]
#src_speaker_lst = ["0wpCZxiAQzw-00035"]
tgt_speaker_lst = ["0wpCZxiAQzw", "FD5ZKiSmoMU", "1gdKrtwBGqY", "pcNxS2i7SvQ", "11Mq9ZuxZMc", "F2hc2FLOdhI", "oXSyMUeAEec", "Yo5cKRmJaf0", "weyd0UMdP7g", "LKAhTELkOV8", "2L4BSVpvx1A", "xbagFzcyNiM", "6Af6bSwyiwI", "MgUTzbGakRw", "EZwxKPv1CwA", "5duz42kHqPs"]
'''tgt_speaker_lst = []
with open('/home/310505006/data/100_lrs3_shuffle.txt') as f:
    lines = f.readlines()
    for line in lines:
        ID = line.replace('\n','')
        tgt_speaker_lst.append(ID)
src_speaker_lst = tgt_speaker_lst'''
for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        try:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'-00001.wav.npy'))
            #src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
        except:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'-00002.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
            
        src_speaker_emb = None

        try:
            try:    
                tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00001.npy'))
            except:
                tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00002.npy'))       
        
            src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
            src_speaker_mel = src_speaker_mel.float().to(device)   
            if src_speaker_emb is not None:
                src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
            tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

            with torch.no_grad():
                _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
            x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                      
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
                
            spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))
        except:
            pass  

#############################  VCTK->LRS3  #################################
mel_dir = '/home/310505006/data/vctk_22feature/mel/'
src_speaker_lst = ['p286_001', 'p258_031', 'p266_243', 'p333_027', 'p225_006', 'p226_024', 'p292_041', 'p317_074']
tgt_speaker_lst = ["0wpCZxiAQzw", "FD5ZKiSmoMU", "1gdKrtwBGqY", "pcNxS2i7SvQ", "11Mq9ZuxZMc", "F2hc2FLOdhI", "oXSyMUeAEec", "Yo5cKRmJaf0", "weyd0UMdP7g", "LKAhTELkOV8", "2L4BSVpvx1A", "xbagFzcyNiM", "6Af6bSwyiwI", "MgUTzbGakRw", "EZwxKPv1CwA", "5duz42kHqPs"]

for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        try:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
        except:
            pass

        src_speaker_emb = None

        try:    
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00001.npy'))
        except:
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00002.npy'))
            
        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.float().to(device)   
        if src_speaker_emb is not None:
            src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

        with torch.no_grad():
            _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
        x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                  
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))

#############################  TW->LRS3  #################################
mel_dir = '/home/310505006/data/tw_22feature/mel/'
src_speaker_lst = ['mtw']
for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        try:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
        except:
            pass

        src_speaker_emb = None

        try:    
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00001.npy'))
        except:
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00002.npy'))
            
        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.float().to(device)   
        if src_speaker_emb is not None:
            src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

        with torch.no_grad():
            _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
        x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                  
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))

#############################  JP->LRS3  #################################
mel_dir = '/home/310505006/data/jp_22feature/mel/'
src_speaker_lst = ['jp']
for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        try:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
        except:
            pass

        src_speaker_emb = None

        try:    
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00001.npy'))
        except:
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00002.npy'))
            
        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.float().to(device)   
        if src_speaker_emb is not None:
            src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

        with torch.no_grad():
            _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
        x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                  
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))

#############################  FR->LRS3  #################################
mel_dir = '/home/310505006/data/fr_22feature/mel/'
src_speaker_lst = ['fr']
for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        try:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
        except:
            pass

        src_speaker_emb = None

        try:    
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00001.npy'))
        except:
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00002.npy'))
            
        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.float().to(device)   
        if src_speaker_emb is not None:
            src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

        with torch.no_grad():
            _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
        x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                  
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))

#############################  Ch->LRS3  #################################
mel_dir = '/home/310505006/data/ch_22feature/mel/'
src_speaker_lst = ['fch']
for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        try:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
        except:
            pass

        src_speaker_emb = None

        try:    
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00001.npy'))
        except:
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker, tgt_speaker+'-00002.npy'))
            
        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.float().to(device)   
        if src_speaker_emb is not None:
            src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

        with torch.no_grad():
            _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
        x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                  
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))

#############################  LRS3  inter #################################
'''mel_dir = '/home/310505006/data/lrs3_22feature_500/mel/'
src_speaker_lst = ["1gdKrtwBGqY"]
tgt_speaker_lst = ["xbagFzcyNiM", "EZwxKPv1CwA"]

for i, src_speaker in enumerate(src_speaker_lst):
        try:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'-00001.wav.npy'))
            #src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
        except:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'-00002.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
            
        src_speaker_emb = None

        try:    
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker_lst[0], tgt_speaker_lst[0]+'-00001.npy'))
        except:
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker_lst[0], tgt_speaker_lst[0]+'-00002.npy'))       
        
        try:    
            tgt_speaker_embb = np.load(os.path.join(spk_dir, tgt_speaker_lst[1], tgt_speaker_lst[1]+'-00001.npy'))
        except:
            tgt_speaker_embb = np.load(os.path.join(spk_dir, tgt_speaker_lst[1], tgt_speaker_lst[1]+'-00002.npy'))       
    

        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.float().to(device)   
        if src_speaker_emb is not None:
            src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    
        tgt_speaker_embb = torch.from_numpy(tgt_speaker_embb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

        for alpha in np.arange(0, 1.1, 0.33):
            with torch.no_grad():
                _, x_identic_psnt, _ = G.inter(src_speaker_mel, tgt_speaker_emb, tgt_speaker_embb, alpha)
            x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                      
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
                
            spect_vc.append(('{}x{}_{}_{:.2f}'.format(src_speaker, tgt_speaker_lst[0], tgt_speaker_lst[1], alpha), uttr_trg))

#mel_dir = '/home/310505006/data/vctk_22feature/mel/'
src_speaker_lst = ['0wpCZxiAQzw']
tgt_speaker_lst = ["Yo5cKRmJaf0", "oXSyMUeAEec"]

for i, src_speaker in enumerate(src_speaker_lst):
        try:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'-00001.wav.npy'))
            #src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
        except:
            src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+'-00002.wav.npy'))
            src_speaker_mel = src_speaker_mel.transpose()
            m = np.ones(src_speaker_mel.shape)
            m[:, 10:]=0
            src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
            
        src_speaker_emb = None

        try:    
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker_lst[0], tgt_speaker_lst[0]+'-00001.npy'))
        except:
            tgt_speaker_emb = np.load(os.path.join(spk_dir, tgt_speaker_lst[0], tgt_speaker_lst[0]+'-00002.npy'))       
        
        try:    
            tgt_speaker_embb = np.load(os.path.join(spk_dir, tgt_speaker_lst[1], tgt_speaker_lst[1]+'-00001.npy'))
        except:
            tgt_speaker_embb = np.load(os.path.join(spk_dir, tgt_speaker_lst[1], tgt_speaker_lst[1]+'-00002.npy'))       
    

        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.float().to(device)   
        if src_speaker_emb is not None:
            src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    
        tgt_speaker_embb = torch.from_numpy(tgt_speaker_embb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

        for alpha in np.arange(0, 1.1, 0.33):
            with torch.no_grad():
                _, x_identic_psnt, _ = G.inter(src_speaker_mel, tgt_speaker_emb, tgt_speaker_embb, alpha)
            x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                      
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
                
            spect_vc.append(('{}x{}_{}_{:.2f}'.format(src_speaker, tgt_speaker_lst[0], tgt_speaker_lst[1], alpha), uttr_trg))

'''
Path('result/'+dir+'').mkdir(parents=True, exist_ok=True)        
with open('result/'+dir+'/results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)