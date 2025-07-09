import soundfile as sf
import re

def load_wav(path):
    wav, sr = sf.read(path)
    return wav, sr

def text_to_sequence(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    char_to_id = {ch: idx for idx, ch in enumerate(vocab)}
    return [char_to_id[ch] for ch in text if ch in char_to_id]
