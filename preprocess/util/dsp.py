import numpy as np
import torch

import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import logging

from util.config import Config
from util.vocoder import get_vocoder
from util.waveglow_mel import TacotronSTFT

logger = logging.getLogger(__name__)


class Dsp():
    def __init__(self, config=None):
        self.load_config(config)
        self.build_mel_basis()

        # another modules
        self.vocoder = None
        self.s3prl = None
        self.resemblyzer = None

    def load_config(self, config):
        default = {
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'sample_rate': 22050,
            'n_mels': 80,
            'f_min': 0,
            'f_max': 11025,
            'trim': 20,
        }
        if config is None:
            logger.info('Dsp config is None, use default config.')
            config = default
        self.config = Config(config)
        logger.info(self.config)
        for k, v in default.items():
            if k not in self.config.keys():
                self.config[k] = v

        self.stft = TacotronSTFT(
            self.config.n_fft, self.config.hop_length, self.config.win_length,
            self.config.n_mels, self.config.sample_rate, self.config.f_min,
            self.config.f_max)

    def load_wav(self, path):
        y, sr = librosa.load(path, sr=self.config.sample_rate)
        if type(self.config.trim) is int:
            y, _ = librosa.effects.trim(y, top_db=self.config.trim)
        y = np.clip(y, -1.0, 1.0)
        return y

    def save_wav(self, y, path):
        sf.write(file=path, data=y, samplerate=self.config.sample_rate)

    def build_mel_basis(self):
        self.mel_basis = librosa.filters.mel(sr=self.config.sample_rate, n_fft=self.config.n_fft,
                fmin=self.config.f_min, fmax=self.config.f_max,
                n_mels=self.config.n_mels)

    def wav2mel(self, y):
        D = np.abs(librosa.stft(y, n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, win_length=self.config.win_length)**2)
        D = np.sqrt(D)
        S = np.dot(self.mel_basis, D)
        #log_S = np.log10(S)
        #log_S = np.log10(S+1e-10)
        log_S = np.log10(np.clip(S, 1e-4, np.max(S)))
        assert not np.any(np.isfinite(log_S)==False)
        return log_S

    def wav2mel_waveglow(self, y):
        #max_wav_value=32768.0
        y = torch.FloatTensor(y.astype(np.float32))
        #audio_norm = y / max_wav_value
        audio_norm = y.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        assert not torch.any(torch.isfinite(melspec)==False)
        return np.array(melspec)


    def mel2wav(self, mel, save=''):
        if self.vocoder is None:
            self.build_vocoder()
        return self.vocoder.mel2wav(mel, save=save)

    def build_vocoder(self):
        if torch.cuda.is_available:
            device = 'cuda'
        else:
            device = 'cpu'
        self.vocoder = get_vocoder(device=device)

    def wav2s3prl_spec(self, wav):
        if self.s3prl is None:
            from s3prl import S3prl
            self.s3prl = S3prl()
        ret = self.s3prl.wav2spec(wav)
        return ret

    def wav2resemblyzer(self, wav):
        if self.resemblyzer is None:
            from resemblyzer import VoiceEncoder
            self.resemblyzer = VoiceEncoder()
        ret = self.resemblyzer.embed_utterance(wav)
        return ret

    @staticmethod
    def plot_spectrogram(mag, save=''):
        librosa.display.specshow(mag, x_axis='off', cmap='viridis')
        plt.title('spectrogram')
        if save != '':
            plt.savefig(save, format='jpg')
            plt.close()
        else:
            plt.show()
