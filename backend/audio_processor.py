import numpy as np
import librosa
import io
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

TARGET_SR = 16000
BANDPASS_LOW = 200
BANDPASS_HIGH = 2000
BANDPASS_ORDER = 5
N_MFCC = 13
FEAT_LEN = 78

def bandpass_filter(audio, sr, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, order=BANDPASS_ORDER):
    """Apply bandpass filter to audio"""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1:
        return audio
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)
    return filtered

def preprocess_audio(audio, sr, target_sr=TARGET_SR):
    """Preprocess audio: resample, bandpass filter, normalize"""
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = bandpass_filter(audio, sr)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio, sr

def extract_features_from_file(file_path):
    """Extract 78-dim feature vector from audio file"""
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        audio, sr = preprocess_audio(audio, sr)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        n_frames = mfcc.shape[1]
        
        if n_frames < 3:
            print(f"    ⚠️ Too few frames: {n_frames}")
            return None
        
        # Calculate delta width based on available frames
        delta_width = min(9, n_frames)
        if delta_width % 2 == 0:
            delta_width -= 1
        if delta_width < 3:
            delta_width = 3
        
        # Compute delta and delta-delta
        delta = librosa.feature.delta(mfcc, width=delta_width, mode='nearest')
        delta2 = librosa.feature.delta(mfcc, order=2, width=delta_width, mode='nearest')
        
        # Concatenate features: mean and std of mfcc, delta, delta2
        feature_vec = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1)
        ])
        
        # Handle NaN/Inf values
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure exact feature length (78)
        if len(feature_vec) < FEAT_LEN:
            feature_vec = np.pad(feature_vec, (0, FEAT_LEN - len(feature_vec)), mode='constant')
        elif len(feature_vec) > FEAT_LEN:
            feature_vec = feature_vec[:FEAT_LEN]
        
        return feature_vec.reshape(1, -1)
        
    except Exception as e:
        print(f"    Feature extraction error: {e}")
        return None

def extract_features_from_bytes(audio_bytes):
    """Extract features from audio bytes"""
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        y, sr = preprocess_audio(y, sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        n_frames = mfcc.shape[1]
        
        if n_frames < 3:
            return None
        
        delta_width = min(9, n_frames)
        if delta_width % 2 == 0:
            delta_width -= 1
        if delta_width < 3:
            delta_width = 3
        
        delta = librosa.feature.delta(mfcc, width=delta_width, mode='nearest')
        delta2 = librosa.feature.delta(mfcc, order=2, width=delta_width, mode='nearest')
        
        feature_vec = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1)
        ])
        
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(feature_vec) < FEAT_LEN:
            feature_vec = np.pad(feature_vec, (0, FEAT_LEN - len(feature_vec)), mode='constant')
        elif len(feature_vec) > FEAT_LEN:
            feature_vec = feature_vec[:FEAT_LEN]
        
        return feature_vec.reshape(1, -1)
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None