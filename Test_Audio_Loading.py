import os
import torchaudio
import soundfile as sf

# Your audio file
audio_file = "EnglishTriageTest.wav"  # Use the WAV version

print(f"Testing audio file: {audio_file}")
print(f"File exists: {os.path.exists(audio_file)}")
print(f"Absolute path: {os.path.abspath(audio_file)}")

if not os.path.exists(audio_file):
    print("\n❌ File not found!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files here: {[f for f in os.listdir('.') if f.endswith(('.wav', '.mp3'))]}")
else:
    print(f"File size: {os.path.getsize(audio_file)} bytes")
    
    # Test 1: Try with soundfile
    print("\n--- Test 1: soundfile ---")
    try:
        data, samplerate = sf.read(audio_file)
        print(f"✓ soundfile SUCCESS")
        print(f"  Shape: {data.shape}")
        print(f"  Sample rate: {samplerate}")
    except Exception as e:
        print(f"❌ soundfile FAILED: {e}")
    
    # Test 2: Try with torchaudio
    print("\n--- Test 2: torchaudio ---")
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        print(f"✓ torchaudio SUCCESS")
        print(f"  Shape: {waveform.shape}")
        print(f"  Sample rate: {sample_rate}")
    except Exception as e:
        print(f"❌ torchaudio FAILED: {e}")
    
    # Test 3: Try with Whisper pipeline directly
    print("\n--- Test 3: Whisper pipeline ---")
    try:
        from transformers import pipeline
        
        print("Loading pipeline...")
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
        
        print(f"Transcribing {audio_file}...")
        result = asr(audio_file)
        
        print(f"✓ Pipeline SUCCESS")
        print(f"  Transcription: {result['text']}")
        
    except Exception as e:
        print(f"❌ Pipeline FAILED: {type(e).__name__}")
        print(f"  Error: {e}")