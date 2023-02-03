import os
import torch
import logging
import numpy as np
from glob import glob
from tqdm import tqdm

from preprocessor.base import preprocess_one
from .base import BaseAgent
from util.dsp import Dsp
import parselmouth
from scipy import signal
import soundfile as sd
import pyworld as pw
from scipy.fft import idctn, dctn

logger = logging.getLogger(__name__)

def gen_wav_list(path):
    if os.path.isdir(path):
        wav_list = glob(os.path.join(path, '*.wav'))
    elif os.path.isfile(path):
        wav_list = [path]
    else:
        raise NotImplementedError(f'{path} is invalid for generating wave file list.')
    return wav_list

class WaveData():
    def __init__(self, path):
        self.path = path
        self.processed = False
        self.data = {}

    def set_processed(self):
        self.processed = True

    def is_processed(self):
        return self.processed
    
    def __getitem__(self, key):
        if type(key) is str:
            return self.data[key]
        else:
            raise NotImplementedError
    
    def __setitem__(self, key, value):
        if type(key) is str:
            self.data[key] = value
        else:
            raise NotImplementedError

class Inferencer(BaseAgent):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.indexes_path = config.dataset.indexes_path
        self.dsp_modules = {}
        for feat in config.dataset.feat:
            if feat in self.dsp_modules.keys():
                module = self.dsp_modules[feat]
                print(feat)
            else:
                module = Dsp(args.dsp_config.feat[feat])
                self.dsp_modules[feat] = module
        self.model_state, self.step_fn = self.build_model(config.build)
        self.model_state = self.load_model(self.model_state, args.load)
        self.config = config.build

    def build_model(self, build_config):
        return super().build_model(build_config, mode='inference', device=self.device)

    def load_wav_data(self, source_path, target_path, out_path):
        # load wavefiles
        sources =gen_wav_list(source_path)
        assert(len(sources) > 0), f'Source path "{source_path}"" should be a wavefile or a directory which contains wavefiles.'
        targets = gen_wav_list(target_path)
        assert(len(targets) > 0), f'Target path "{target_path}" should be a wavefile or a directory which contains wavefiles.'
        if os.path.exists(out_path):
            assert(os.path.isdir(out_path)), f'Output path "{out_path}" should be a directory.'
        else:
            os.makedirs(out_path)
            logger.info(f'Output directory "{out_path}" is created.')
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(os.path.join(out_path, 'wav'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'plt'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'mel'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'npy'), exist_ok=True)

        for i, source in enumerate(sources):
            sources[i] = WaveData(source)
        for i, target in enumerate(targets):
            targets[i] = WaveData(target)

        return sources, targets, out_path

    def process_wave_data(self, wav_data, seglen=None):
        if wav_data.is_processed():
            return
        else:
            wav_path = wav_data.path
            basename = os.path.basename(wav_path)
            for feat, module in self.dsp_modules.items():
                wav_data[feat] = preprocess_one((wav_path, basename), module)
                '''snd = parselmouth.Sound(wav_path)
                pitch = snd.to_pitch()
                pitch_values = pitch.selected_array['frequency']
                wav_data['pitch'] = signal.resample(pitch_values,wav_data[feat].shape[1])'''
                '''data, fs = sd.read(wav_path)
                f0, sp, ap = pw.wav2world(data, fs)
                x = signal.resample(ap, wav_data[feat], axis=0)
                y = signal.resample(x, 256, axis=1)
                wav_data['ap'] = np.transpose(np.array(y))'''
                '''try:
                    wav_data['ap'] = np.load(os.path.join('/home/310505006/data/lrs3_ap160_500/','/'.join(wav_path.split('/')[-2:]).replace('.wav','.npy')))
                    wav_data['sp'] = np.load(os.path.join('/home/310505006/data/lrs3_sp160_500/','/'.join(wav_path.split('/')[-2:]).replace('.wav','.npy')))
                except:
                    wav_data['ap'] = np.load(os.path.join('/home/310505006/data/vctk_ap/','/'.join(wav_path.split('/')[-2:]).replace('.wav','.npy')))
                '''
                data, fs = sd.read(wav_path)
                f0, sp, ap = pw.wav2world(data, fs)
                '''sp = np.log(sp+1e-9)
                sp = signal.resample(sp, wav_data[feat].shape[1], axis=0)
                sp = signal.resample(sp, wav_data[feat].shape[0], axis=1)
                sp = np.rot90(sp)'''
                m = np.ones(wav_data[feat].shape)
                m[10:, :]=0
                wav_data['sp'] = idctn(dctn(wav_data[feat])*m)
                #print(ap.shape, wav_data[feat].shape)
                ap = signal.resample(ap, wav_data[feat].shape[1], axis=0)
                ap = signal.resample(ap, 256, axis=1)
                ap = ap.transpose(-1,-2)
                wav_data['ap'] = ap
                #print(ap.shape)
                '''wav_data['ap'] = np.load('/disk/data/VCTK-Corpus/vctk_ap/p225/p225_001.npy')
                wav_data['ap'] = signal.resample(wav_data['ap'], wav_data[feat].shape[1], axis=1)
                print(wav_data['ap'].shape)'''

                if seglen is not None:
                    wav_data[feat] = wav_data[feat][:,:seglen]
                    #wav_data['pitch'] = wav_data['pitch'][:,:seglen]
                    wav_data['ap'] = wav_data['ap'][:,:seglen]
                    wav_data['sp'] = wav_data['sp'][:,:seglen]

                try:
                    wav_data['face'] = np.load(os.path.join('/home/310505006/data/faceemb_lrs3_mtcnn_margin50_500/', wav_path.split('/')[-2], wav_path.split('/')[-1].split('-')[-1].split('.')[0], '00000.npy'))
                except:
                    pass
                wav_data.set_processed()
            return

    # ====================================================
    #  inference
    # ====================================================
    def inference(self, source_path, target_path, out_path, seglen):
        sources, targets, out_path = self.load_wav_data(source_path, target_path, out_path)
        with torch.no_grad():
            for i, source in enumerate(sources):
                logger.info(f'Source: {source.path}')
                for j, target in enumerate(targets):
                    logger.info(f'Target: {target.path}')
                    source_basename = os.path.basename(source.path).split('.wav')[0]
                    target_basename = os.path.basename(target.path).split('.wav')[0]
                    output_basename = f'{source_basename}_to_{target_basename}'
                    output_wav = os.path.join(out_path, 'wav', output_basename+'.wav')
                    output_plt = os.path.join(out_path, 'plt', output_basename+'.png')
                    self.process_wave_data(source, seglen=seglen)
                    # These two line is for generating the wav using melgan.
                    # self.dsp.mel2wav(source['melgan'], os.path.join('data/tmp/mos_melgan/', source_basename+'.wav'))
                    # continue
                    self.process_wave_data(target, seglen=seglen)
                    data = {
                        'source': source,
                        'target': target,
                    }
                    meta = self.step_fn(self.model_state, data, self.config)
                    dec = meta['dec']
                    self.mel2wav(dec, output_wav)
                    Dsp.plot_spectrogram(dec.squeeze().cpu().numpy(), output_plt)

                    source_plt = os.path.join(out_path, 'plt', f'{source_basename}.png')
                    Dsp.plot_spectrogram(source['mel'], source_plt)
                    np.save(os.path.join(out_path, 'mel', f'{source_basename}.npy'), source['mel'])
                    np.save(os.path.join(out_path, 'npy', f'{output_basename}.npy'), dec.cpu().numpy())
        logger.info(f'The generated files are saved to {out_path}.')