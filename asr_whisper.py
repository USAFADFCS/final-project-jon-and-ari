import os, torch
from transformers import pipeline

MODEL_ID = "openai/whisper-large-v3-turbo"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Build one reusable pipeline
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=DEVICE,
    torch_dtype=DTYPE,
    chunk_length_s=30,           # robust for long audio
    stride_length_s=5,           # overlap for context
    return_timestamps=True
)

MEDICAL_LEXICON = (
    "tourniquet, hemorrhage, capillary refill, obey commands, airway, "
    "respirations, pulse, radial pulse, naloxone, unresponsive, shock"
)

def transcribe(path: str) -> dict:
    return asr(
        path,
        generate_kwargs={
            "task": "transcribe",      # or "translate" if needed
            "temperature": 0.0,
            "num_beams": 5
        },
        # primes decoding with triage vocabulary
        prompt=MEDICAL_LEXICON,
        return_timestamps=True
    )
