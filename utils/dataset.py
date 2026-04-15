from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

from .video import read_video_frames
from .audio import load_audio_mel_cached


class LAVDFMultimodalDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        num_frames: int = 6,
        img_size: int = 160,
        audio_sr: int = 16000,
        audio_duration: float = 3.0,
        n_mels: int = 128,
        n_fft: int = 400,
        hop_length: int = 160,
        max_audio_time_steps: int = 300,
        cache_dir: str | Path = "cache/mel_cache",
    ):
        df = pd.read_csv(csv_path)
        self.files = df["file"].tolist()
        self.labels = df["label"].astype("float32").tolist()
        self.data_root = Path(data_root)

        self.num_frames = num_frames
        self.img_size = img_size
        self.audio_sr = audio_sr
        self.audio_duration = audio_duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_time_steps = max_audio_time_steps
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel_path = self.files[idx]
        label = self.labels[idx]
        video_path = self.data_root / rel_path

        frames = read_video_frames(
            video_path, num_frames=self.num_frames, img_size=self.img_size
        )
        mel = load_audio_mel_cached(
            video_path,
            rel_path,
            cache_dir=self.cache_dir,
            sr=self.audio_sr,
            duration=self.audio_duration,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            max_time_steps=self.max_audio_time_steps,
        )

        return {
            "frames": torch.tensor(frames, dtype=torch.float32),
            "mel": torch.tensor(mel, dtype=torch.float32).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }
