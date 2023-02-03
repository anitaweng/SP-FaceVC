import numpy as np
import cv2
from scipy import signal

def pad(x, seglen, mode='wrap'):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0,0), (0,pad_len)), mode=mode)
    return y

def pad_pitch(x, pitchlen, mode='wrap'):
    pad_len = pitchlen - x.shape[0]
    y = np.pad(x, ((0), (pad_len)), mode=mode)
    return y

def pad_ap(x, seglen, mode='wrap'):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0,0), (0,pad_len)), mode=mode)
    return y

def segment(x, pitch, ap, sp, seglen=128, r=None, return_r=False, pitchlen=147):
    if pitch is not None:
        pitch_y = signal.resample(pitch,x.shape[1])
    if x.shape[1] < seglen:
        y = pad(x, seglen)
        if pitch is not None:
            pitch_y = pad_pitch(pitch_y, seglen)
        if ap is not None:
            ap = pad_ap(ap, seglen)
        sp = pad_ap(sp, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        if r is None:
            r = np.random.randint(x.shape[1] - seglen)
        '''lp, rp = r/x.shape[1], (r+seglen)/x.shape[1]
        tot_len = pitch.shape[0]
        start = int(np.rint(tot_len*lp))
        try:
            pitch_y = pitch[start:start+pitchlen]
        except:
            pitch_y = pitch[start-1:start+pitchlen-1]'''
        y = x[:,r:r+seglen]
        if pitch is not None:
            pitch_y = pitch_y[r:r+seglen]
        if ap is not None:
            ap = ap[:,r:r+seglen]
        sp = sp[:,r:r+seglen]

    if pitch is None:
        pitch_y = None
    
    if return_r:
        return y, r, pitch_y, ap, sp
    else:
        return y, pitch_y, ap, sp

def resize(x, dim):
    return cv2.resize(x, dim, interpolation=cv2.INTER_AREA)

def random_scale(mel, allow_flip=False, r=None, return_r=False):
    if r is None:
        r = np.random.random(3)
    rate = r[0] * 0.6 + 0.7 # 0.7-1.3
    dim = (int(mel.shape[1] * rate), mel.shape[0])
    r_mel = resize(mel, dim)

    rate = r[1] * 0.4 + 0.3 # 0.3-0.7
    trans_point = int(dim[0] * rate)
    dim = (mel.shape[1]-trans_point, mel.shape[0])
    if r_mel.shape[1] < mel.shape[1]:
        r_mel = pad(r_mel, mel.shape[1])
    # r_mel[:,trans_point:mel.shape[1]] = cv2.resize(r_mel[:,trans_point:], dim, interpolation=cv2.INTER_AREA)
    r_mel[:,trans_point:mel.shape[1]] = resize(r_mel[:,trans_point:], dim)
    if r[2] > 0.5 and allow_flip:
        ret = r_mel[:,:mel.shape[1]][:,::-1].copy()
    else:
        ret = r_mel[:,:mel.shape[1]]
    if return_r:
        return ret, r
    else:
        return ret