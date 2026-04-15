# Multimodal Deepfake Detection
Description: Multimodal deepfake detection using PyTorch, SAFF, CM-GAN, FastAPI, and Gradio.
Topics: pytorch, deep-learning, multimodal, deepfake-detection, fastapi, gradio, audio-visual

A multimodal deepfake detection system that combines visual and audio signals for robust fake video detection. The project implements:

- Visual encoder using EfficientNet + temporal modeling
- Audio encoder using residual CNN blocks with squeeze-and-excitation
- SAFF: Synchronization-Aware Feature Fusion
- CM-GAN: Cross-Modal Graph Attention Network
- Deployment via FastAPI and Gradio
- Dataset used: https://www.kaggle.com/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df

## Features
- Audio-video deepfake detection
- Multimodal temporal modeling
- Research-inspired SAFF + CM-GAN architecture
- Training and evaluation pipelines
- API and demo UI support

## Project Structure
```text
models/      -> model definitions
utils/       -> preprocessing and dataset loading
training/    -> training scripts
api/         -> FastAPI backend
demo/        -> Gradio demo
data/        -> local dataset storage (ignored in git)
outputs/     -> manifests and checkpoints
cache/       -> feature caching
