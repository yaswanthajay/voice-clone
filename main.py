# voice_cloning_app/main.py

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import numpy as np
import os
import sys

# Ensure current folder is in import path
sys.path.append(".")

from model import SpeakerEncoder, SimpleTTS, SimpleVocoder
from utils import text_to_sequence, load_wav

# Check if sample audio file exists
if os.path.exists("sample_audio.wav"):
    print("✅ sample_audio.wav found!")
else:
    print("❌ sample_audio.wav not found. Please place a .wav file in this folder.")
    exit()

def main():
    # === STEP 1: Load audio and extract voice embedding ===
    wav, sr = load_wav("sample_audio.wav")  # Load using librosa
    wav_tensor = torch.tensor(wav).float().unsqueeze(0)

    # Extract mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=40,
        n_fft=1024,
        hop_length=256
    )
    mel_spec = mel_transform(wav_tensor)

    speaker_encoder = SpeakerEncoder()
    speaker_embedding = speaker_encoder(mel_spec).detach()

    # === STEP 2: Convert text to mel-spectrogram ===
    text = "Hello, this is my cloned voice"
    tts = SimpleTTS()
    text_seq = text_to_sequence(text)
    text_tensor = torch.tensor(text_seq).unsqueeze(0)

    mel_out = tts(text_tensor, speaker_embedding.unsqueeze(0)).detach()

    # === STEP 3: Vocoder - Convert mel to waveform ===
    vocoder = SimpleVocoder()
    waveform = vocoder(mel_out).squeeze().detach().numpy()

    # Save to file
    sf.write("output.wav", waveform, 22050)
    print("✅ Cloned voice saved to output.wav")

if __name__ == "__main__":
    main()
