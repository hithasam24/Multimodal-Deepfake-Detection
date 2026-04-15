from pathlib import Path
import librosa
import numpy as np


def mel_cache_path(cache_dir: str | Path, video_rel_path: str):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = video_rel_path.replace("/", "__").replace(".mp4", ".npy")
    return cache_dir / safe_name


def load_audio_mel_cached(
    video_path: str | Path,
    video_rel_path: str,
    cache_dir: str | Path = "cache/mel_cache",
    sr: int = 16000,
    duration: float = 3.0,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
    max_time_steps: int = 300,
):
    cache_path = mel_cache_path(cache_dir, video_rel_path)
    if cache_path.exists():
        return np.load(cache_path)

    y, _ = librosa.load(str(video_path), sr=sr, mono=True, duration=duration)
    if len(y) == 0:
        y = np.zeros(int(sr * duration), dtype=np.float32)

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    t = mel_db.shape[1]
    if t < max_time_steps:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_time_steps - t)))
    else:
        mel_db = mel_db[:, :max_time_steps]

    np.save(cache_path, mel_db)
    return mel_db
