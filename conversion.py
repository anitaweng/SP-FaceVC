import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from pathlib import Path 
#
dir = 'autovc_100spk_ssim_spkmean_adamw_160'
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

g_checkpoint = torch.load('/disk/autovc/checkpoint/'+dir+'/G.ckpt')
G.load_state_dict(g_checkpoint)

#g_checkpoint = torch.load('autovc.ckpt')
#G.load_state_dict(g_checkpoint['model'])

p = torch.nn.MaxPool2d((1,3), stride=1, padding=(0,1))

spect_vc = []

#src_speaker_lst = ['p286_001', 'p258_031', 'p266_243', 'p333_027']
#src_speaker_lst = ['38_5845_20170916163416', '5_3099_20170702124152']
src_speaker_lst = ["0af00UcTOSc", "FD5ZKiSmoMU",  "0akiEFwtkyA", "pcNxS2i7SvQ"]
#src_speaker_lst = ["oy25A7vnigg"]
#tgt_speaker_lst = ['p286_001', 'p258_031', 'p266_243', 'p333_027']
#tgt_speaker_lst = ['0af00UcTOSc-00024.npy', '0C5UQbWzwg8-00029.npy', '0akiEFwtkyA-00025.npy', '0d6iSvF1UmA-00030.npy','0FQXicAGy5U-00001.npy', '1bXAkbCyjpo-00032.npy', '0JGarsZE1rk-00002.npy']
#tgt_speaker_lst = ['0af00UcTOSc-00024.npy', '0C5UQbWzwg8-00029.npy', '0akiEFwtkyA-00025.npy', '0d6iSvF1UmA-00030.npy']
#tgt_speaker_lst = ['p225_001.npy']
tgt_speaker_lst = ["0af00UcTOSc", "FD5ZKiSmoMU",  "0akiEFwtkyA", "pcNxS2i7SvQ", "0wpCZxiAQzw",  "1gdKrtwBGqY", "11Mq9ZuxZMc", "F2hc2FLOdhI", "oXSyMUeAEec", "Yo5cKRmJaf0"]#['0d6iSvF1UmA', '0C5UQbWzwg8', '0akiEFwtkyA', '0af00UcTOSc']#["2ZviHInGBJQ", "qykSnLkPM7E", "16cMSRFid9U", "zvvZEIGNLo0",  "0nI65jgHG9o", "VStEmE85QDE", "0tqq66zwa7g"]


for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        #src_speaker_mel = np.load(os.pa_500000th.join('mel_100', src_speaker))
        try:
            src_speaker_mel = np.load(os.path.join('/disk/data/lrs3/lrs3_16k_spmel_all_160/',src_speaker, src_speaker+'-00001.npy'))
            #src_speaker_mel = np.load(os.path.join('/disk/data/VCTK-Corpus/spmel16/',src_speaker.split('_')[0], src_speaker+'.npy')) 
            #src_speaker_mel = np.load(os.path.join('/disk/data/dev_set/chinese_mel','_'.join(src_speaker.split('_')[:2]), src_speaker+'.npy'))
        except:
            src_speaker_mel = np.load(os.path.join('/disk/data/lrs3/lrs3_16k_spmel_all_160/',src_speaker, src_speaker+'-00002.npy'))
            #src_speaker_mel = np.load(os.path.join('/disk/data/VCTK-Corpus/spmel16/',src_speaker.split('_')[0], src_speaker+'.npy')) 
            #src_speaker_mel = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spmel16/p225/', src_speaker))
            #src_speaker_mel = np.load(os.path.join('/mnt/hdd0/hsiaohan/lrs3/spmel16/'+src_speaker.split('-')[0], src_speaker))
            
            #src_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/sv2tts_embeds16', 'embed-'+src_speaker))
            #tgt_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/sv2tts_embeds16', 'embed-'+tgt_speaker))
            
            #src_speaker_emb = np.load(os.path.join('/mnt/ssd0/hsiaohan/', src_speaker))
            #tgt_speaker_emb = np.load(os.path.join('/mnt/ssd0/hsiaohan/', tgt_speaker))
        try:
            src_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_no_meam/',src_speaker, src_speaker+'-00001.npy'))
            #src_speaker_emb = np.load(os.path.join('/disk/data/VCTK-Corpus/spk_emb16_no_mean/', src_speaker.split('_')[0],src_speaker+'.npy'))
            #src_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_meam/',src_speaker, src_speaker+'.npy'))
            #src_speaker_emb = np.load(os.path.join('/disk/data/dev_set/chinese_spk','_'.join(src_speaker.split('_')[:2]), src_speaker+'.npy'))
        except:
            src_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_no_meam/',src_speaker, src_speaker+'-00002.npy'))
            #src_speaker_emb = np.load(os.path.join('/disk/data/dev_set/chinese_spk','_'.join(src_speaker.split('_')[:2]), src_speaker+'.npy'))
            #src_speaker_emb = np.load(os.path.join('/disk/data/VCTK-Corpus/spk_emb16_no_mean/',src_speaker.split('_')[0]+'.npy'))   
            #src_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/lrs3/spk_emb16/', src_speaker.split('-')[0]+'.npy'))
            #tgt_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spk_emb16/', tgt_speaker.split('_')[0]+'.npy'))
        try:    
            tgt_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_no_meam/', tgt_speaker, tgt_speaker+'-00001.npy'))
            #tgt_speaker_emb = np.load(os.path.join('/disk/data/VCTK-Corpus/spk_emb16_no_mean/',tgt_speaker.split('_')[0], tgt_speaker+'.npy'))
            #tgt_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_meam/', tgt_speaker, tgt_speaker+'.npy'))
        except:
            tgt_speaker_emb = np.load(os.path.join('/disk/data/lrs3/autovc_spk_emb16_all_no_meam/', tgt_speaker, tgt_speaker+'-00002.npy'))
            #tgt_speaker_emb = np.load(os.path.join('/disk/data/VCTK-Corpus/spk_emb16_no_mean/',tgt_speaker.split('_')[0]+'.npy'))
            
        
        print('src_speaker_mel',src_speaker_mel.shape)
        src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
        src_speaker_mel = src_speaker_mel.to(device)       
        src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    
        print('src_speaker_mel',src_speaker_mel.shape)

        '''src_speaker_mel, len_pad = pad_seq(src_speaker_mel)
        src_speaker_mel = torch.from_numpy(src_speaker_mel[np.newaxis, :, :]).to(device)
        src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).to(device)
        print('src_speaker_mel',src_speaker_mel.shape)'''

        '''if src_speaker=="0akiEFwtkyA" or src_speaker=="pcNxS2i7SvQ" or src_speaker=="0tqq66zwa7g" or src_speaker=="VStEmE85QDE" or src_speaker=='0d6iSvF1UmA' or src_speaker=='p266_243' or src_speaker=='p333_027' or src_speaker=='5_3099_20170702122314'\
        or src_speaker=="5_3099_20170702124152":
            src_speaker_mel = p(src_speaker_mel.unsqueeze(1)).squeeze(1)
        print('src_speaker_mel',src_speaker_mel.shape)'''
        #src_speaker_mel = p(src_speaker_mel.unsqueeze(1)).squeeze(1)

        

        
        with torch.no_grad():
            #if 
            _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
        x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
        print(x_identic_psnt.shape)            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
        print(uttr_trg.shape)
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))

'''tgt_speaker_lst = ['p286_001.npy', 'p258_031.npy', 'p266_243.npy', 'p333_027.npy']
for i, src_speaker in enumerate(src_speaker_lst):
    for j, tgt_speaker in enumerate(tgt_speaker_lst):
        print(src_speaker + '>' + tgt_speaker)
        #src_speaker_mel = np.load(os.path.join('mel_100', src_speaker))
        src_speaker_mel = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spmel16/'+src_speaker.split('_')[0], src_speaker))
        #src_speaker_mel = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spmel16/p225/', src_speaker))
        #src_speaker_mel = np.load(os.path.join('/mnt/hdd0/hsiaohan/lrs3/spmel16/'+src_speaker.split('-')[0], src_speaker))
        
        #src_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/sv2tts_embeds16', 'embed-'+src_speaker))
        #tgt_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/sv2tts_embeds16', 'embed-'+tgt_speaker))
        
        #src_speaker_emb = np.load(os.path.join('/mnt/ssd0/hsiaohan/', src_speaker))
        #tgt_speaker_emb = np.load(os.path.join('/mnt/ssd0/hsiaohan/', tgt_speaker))

        src_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spk_emb16/', src_speaker.split('_')[0]+'.npy'))
        #src_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/lrs3/spk_emb16/', src_speaker.split('-')[0]+'.npy'))
        tgt_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/vctk/VCTK-Corpus/spk_emb16/', tgt_speaker.split('_')[0]+'.npy'))
        #tgt_speaker_emb = np.load(os.path.join('/mnt/hdd0/hsiaohan/lrs3/spk_emb16/', tgt_speaker.split('-')[0]+'.npy'))
        

        src_speaker_mel, len_pad = pad_seq(src_speaker_mel)
        src_speaker_mel = torch.from_numpy(src_speaker_mel[np.newaxis, :, :]).to(device)
        src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).to(device)
        tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).to(device)

        with torch.no_grad():
            _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)

        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            
        print(uttr_trg.shape)
        spect_vc.append(('{}x{}'.format(src_speaker, tgt_speaker), uttr_trg))'''
        
Path('/disk/autovc/result/'+dir+'').mkdir(parents=True, exist_ok=True)        
with open('/disk/autovc/result/'+dir+'/results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)