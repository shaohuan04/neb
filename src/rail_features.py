import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

SAMPLE_RATE_HZ = 10000

def channel_features(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0: return [0.0]*7
    y = x - x.mean()
    spec = np.abs(rfft(y)); freqs = rfftfreq(len(y), 1/SAMPLE_RATE_HZ)
    if len(spec): spec[0] = 0
    dom = float(freqs[np.argmax(spec)]) if len(spec) else 0.0
    return [float(np.sqrt(np.mean(x*x))), float(np.std(x)), float(np.max(np.abs(x))), float(np.ptp(x)), float(kurtosis(x, fisher=False)) if len(x)>3 else 0.0, float(skew(x)) if len(x)>2 else 0.0, dom]

def extract_rail_features(df):
    if df.shape[1] < 129:
        raise ValueError('Expected rotational speed plus 128 vibration/shock channels.')
    out = {}
    signals = df.iloc[:,1:129].apply(pd.to_numeric, errors='coerce')
    names = ['rms','std','peak','ptp','kurtosis','skew','dominant_freq']
    for i,col in enumerate(signals.columns,1):
        vals = channel_features(signals[col].to_numpy())
        for n,v in zip(names,vals): out['ch%d_%s'%(i,n)] = v
    # Channel pairs are vibration/shock for axle positions 1..8, repeated for each car.
    # Odd axle positions = Side I; even axle positions = Side II.
    side1=[]; side2=[]
    for ch in range(1,129):
        axle_pos = ((ch-1)//2)%8 + 1
        (side1 if axle_pos%2 else side2).append(out['ch%d_rms'%ch])
    out['side1_mean_rms']=float(np.mean(side1)); out['side2_mean_rms']=float(np.mean(side2))
    out['side_rms_ratio']=out['side1_mean_rms']/max(out['side2_mean_rms'],1e-12)
    out['speed_signal_mean']=float(pd.to_numeric(df.iloc[:,0],errors='coerce').mean())
    return pd.DataFrame([out])
