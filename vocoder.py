import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
import soundfile as sf

dir = 'spfacevc'
spect_vc = pickle.load(open(dir+'/results.pkl', 'rb'))
device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    plt.figure()
    plt.title('convert_' + name)
    librosa.display.specshow(np.transpose(c, (-1, -2)), x_axis='time', y_axis='mel', sr=16000)
    plt.colorbar(format='%f')
    plt.savefig(os.path.join(dir+'','convert_'+ name + '.png'))
    plt.close()        
    waveform = wavegen(model, c=c)  
    sf.write(dir+'/'+name+'.wav', waveform, 16000)
    
