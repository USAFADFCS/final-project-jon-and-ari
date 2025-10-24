import torchaudio

def simple_vad_chunks(wav_path, min_speech_len=0.6):
    wav, sr = torchaudio.load(wav_path)
    wav = torchaudio.functional.resample(wav, sr, 16000)
    vad = torchaudio.transforms.Vad(sample_rate=16000)
    voiced = vad(wav.squeeze(0))
    # Fallback: if overly aggressive, just return original path
    if voiced.numel() < 16000 * min_speech_len:
        return [wav_path]
    # For brevity, write voiced chunk to temp file; in production, slice windows
    out = "/tmp/voiced.wav"
    torchaudio.save(out, voiced.unsqueeze(0), 16000)
    return [out]
