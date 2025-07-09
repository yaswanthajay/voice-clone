# voice_cloning_app/main.py

import torch
import torch.nn as nn
import torchaudio
import librosa
import soundfile as sf
import numpy as np
import sys
import os

if os.path.exists("sample_audio.wav"):
    print("✅ sample_audio.wav found!")
else:
    print("❌ sample_audio.wav not found.")




def main():
    # === STEP 1: Load audio and extract voice embedding ===
    wav, sr = load_wav("sample_audio.wav")
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=40)(torch.tensor(wav).unsqueeze(0))

    speaker_encoder = SpeakerEncoder()
    speaker_embedding = speaker_encoder(mel_spec).detach()

    # === STEP 2: Convert text to mel-spectrogram ===
    text = "Hello, this is my cloned voice"
    tts = SimpleTTS()
    text_seq = text_to_sequence(text)
    mel_out = tts(torch.tensor(text_seq).unsqueeze(0), speaker_embedding.unsqueeze(0))

    # === STEP 3: Vocoder - Convert mel to waveform ===
    vocoder = SimpleVocoder()
    waveform = vocoder(mel_out).squeeze().detach().numpy()
    sf.write("output.wav", waveform, 22050)

    print("✅ Cloned voice saved to output.wav")

if __name__ == "__main__":
    main()
