import torch
import librosa
import numpy as np
from transformers import AutoProcessor

# 1. Load processor
print("Loading processor...")
processor = AutoProcessor.from_pretrained("google/gemma-4-12b-it", trust_remote_code=True)

# 2. Create a dummy audio array
audio_array = np.zeros(97920, dtype=np.float32)

print("\n--- TEST 1: apply_chat_template tokenize=True with audio_array ---")
try:
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": "Transcribe the audio."},
            ],
        },
    ]
    inputs2 = processor.apply_chat_template(conv, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt")
    audio_tokens2 = (inputs2["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<audio>")).sum()
    print("Test 1 tokens:", audio_tokens2.item())
    if "input_features" in inputs2 or "audio_features" in inputs2 or "pixel_values" in inputs2 or "audio" in inputs2 or "pixel_values_audio" in inputs2:
        print("Test 1 Success! Features present.")
    else:
        print("Test 1 Dropped Audio silently.")
except Exception as e:
    print("Test 1 failed:", type(e).__name__, e)

print("\n--- TEST 2: processor(text, audio=...) ---")
try:
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    print("Formatted text:", repr(text))
    inputs1 = processor(text=text, audio=[audio_array], sampling_rate=16000, return_tensors="pt")
    
    # We must explicitly check how many tokens are in input_ids now.
    audio_tokens1 = (inputs1["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<audio>")).sum()
    print("Test 2 Audio tokens found:", audio_tokens1.item())
except Exception as e:
    print("Test 2 failed:", type(e).__name__, e)

