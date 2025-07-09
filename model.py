# voice_cloning_app/model.py

import torch.nn as nn
import torch

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=2, batch_first=True)
        self.proj = nn.Linear(128, 256)

    def forward(self, mel):
        _, (h, _) = self.lstm(mel)
        return self.proj(h[-1])

class SimpleTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 64)
        self.lstm = nn.LSTM(64 + 256, 128, batch_first=True)
        self.linear = nn.Linear(128, 80)  # mel bands

    def forward(self, x, speaker_embed):
        x = self.embed(x)
        speaker_embed = speaker_embed.expand(-1, x.shape[1], -1)
        x = torch.cat([x, speaker_embed], dim=2)
        out, _ = self.lstm(x)
        return self.linear(out)

class SimpleVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 1, kernel_size=5, padding=2)
        )

    def forward(self, mel):
        mel = mel.transpose(1, 2)  # (B, mel, T)
        return self.conv(mel)
