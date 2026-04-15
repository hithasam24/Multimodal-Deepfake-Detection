from pathlib import Path
import cv2
import numpy as np


def sample_frame_indices(total_frames: int, num_frames: int):
    if total_frames <= 0:
        return [0] * num_frames
    idxs = np.linspace(0, max(total_frames - 1, 0), num_frames)
    return idxs.astype(int).tolist()


def read_video_frames(video_path: str | Path, num_frames: int = 6, img_size: int = 160):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(total_frames, num_frames)
    idxs_set = set(idxs)

    frames_dict = {}
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id in idxs_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames_dict[frame_id] = frame
            if len(frames_dict) == len(idxs):
                break
        frame_id += 1

    cap.release()

    frames = []
    for idx in idxs:
        if idx in frames_dict:
            frames.append(frames_dict[idx])
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))

    frames = np.stack(frames).astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2))  # (T,C,H,W)
    return frames
