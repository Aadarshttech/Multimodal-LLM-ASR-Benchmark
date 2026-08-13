"""
Build script to generate all 5 Kaggle-optimized Jupyter Notebooks
for the Nepali ASR Benchmarking Pipeline.

Each notebook is self-contained with:
- Correct pip install cells (version-pinned to avoid conflicts)
- Kaggle dataset paths
- Official HuggingFace model loading patterns
- Full batch inference pipeline
- Metrics evaluation
"""
import json
import os

KAGGLE_AUDIO_BASE = "/kaggle/input/datasets/panditaadarsh/llm-bechmarking-audio"
KAGGLE_WORKING = "/kaggle/working"

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [
                    {"sourceId": 0, "sourceType": "datasetVersion", "datasetSlug": "llm-bechmarking-audio"}
                ],
                "isGpuEnabled": True,
                "isInternetEnabled": True
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md(text):
    lines = text.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def code(text):
    lines = text.split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {"trusted": True}, "outputs": [], "source": [l + "\n" for l in lines]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTEBOOK 01: Gemma 4 12B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_nb01():
    cells = []
    cells.append(md("# 01 — Gemma 4 12B: Prompt Development for Nepali ASR Benchmarking\n\n**Model:** `google/gemma-4-12b-it`  \n**Architecture:** Encoder-free multimodal (text + image + audio)  \n**Audio support:** Native, up to 30s, 16 kHz mono, 25 tokens/sec  \n**Class:** `AutoModelForImageTextToText`"))
    
    cells.append(md("## 0. Install Dependencies\n\n> ⚠️ Gemma 4 requires `transformers >= 4.54` and the latest `accelerate`. Run this cell first and **restart the runtime** if prompted."))
    
    cells.append(code("""import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", pkg])

# Core — pinned versions that work together for Gemma 4
pip("transformers>=4.54")
pip("accelerate>=1.6")

pip("bitsandbytes>=0.45")
pip("sentencepiece")
pip("protobuf")

# Audio processing
pip("librosa>=0.10")
pip("soundfile>=0.12")
pip("scipy")

# Metrics
pip("jiwer>=3.1")
pip("jsonlines")

# Data
pip("pandas")
pip("tqdm")

print("\\n✅ All dependencies installed.")"""))

    cells.append(md("## A. Experiment Configuration"))
    cells.append(code(f"""import os, json, torch, gc
from datetime import datetime

MODEL_ID   = "google/gemma-4-12b-it"
MODEL_REV  = "main"
QUANT      = "bfloat16"   # Gemma 4 12B fits in bf16 on T4/P100 with device_map=auto

experiment_config = {{
    "model_id":       MODEL_ID,
    "model_revision": MODEL_REV,
    "quantization":   QUANT,
    "random_seed":    42,
    "audio_sr":       16000,
    "batch_size":     1,       # process one at a time for reliability
    "max_new_tokens": 256,
    "temperature":    0.0,
    "do_sample":      False,
    "timestamp":      datetime.now().isoformat(),
}}

AUDIO_BASE = "{KAGGLE_AUDIO_BASE}"
RESULTS_DIR = "{KAGGLE_WORKING}/results/prompt_dev/gemma4_12b"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(f"{{RESULTS_DIR}}/run_config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)

print("Config saved →", RESULTS_DIR)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("VRAM:", f"{{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB" if torch.cuda.is_available() else "N/A")"""))

    cells.append(md("## B. Model Loading"))
    cells.append(code("""from transformers import AutoProcessor, AutoModelForImageTextToText

print(f"Loading {MODEL_ID} ...")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    revision=MODEL_REV,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("✅ Model loaded successfully.")
print(f"   dtype: {model.dtype}")
print(f"   device_map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")"""))

    cells.append(md("## B.1 — Quick Sanity Check (one audio file)"))
    cells.append(code(f"""import glob

# Pick the first .wav from the clean set
audio_files = sorted(glob.glob(f"{{AUDIO_BASE}}/clean_nepali_200_flat/*.wav"))
if not audio_files:
    audio_files = sorted(glob.glob(f"{{AUDIO_BASE}}/clean_nepali_200_flat/*.mp3"))
print(f"Found {{len(audio_files)}} audio files in clean set")

test_audio_path = audio_files[0]
print(f"Testing with: {{test_audio_path}}")

# Build conversation — pass the FILE PATH, not a numpy array.
# The processor will load it internally.
conversation = [
    {{
        "role": "user",
        "content": [
            {{"type": "audio", "audio": test_audio_path}},
            {{"type": "text", "text": "Transcribe the following Nepali speech verbatim in Devanagari script. Output ONLY the transcription, nothing else."}},
        ],
    }},
]

# Use tokenize=True so the processor handles EVERYTHING end-to-end.
# Use load_audio_backend='soundfile' to bypass Kaggle's broken torchcodec.
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={{"load_audio_backend": "librosa"}},
).to(model.device)

print(f"input_ids shape: {{inputs['input_ids'].shape}}")
print(f"input_features present: {{'input_features' in inputs}}")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

# Decode only newly generated tokens
new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
result = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
print(f"\\n📝 Model output:\\n{{result}}")"""))

    cells.append(md("## C. Prompt Templates\n\nDefining prompt levels 0-2 per `todo_bef_benchmark.md`."))
    cells.append(code("""PROMPTS = {
    "L0_a": "Transcribe the following speech segment in its original language. Only output the transcription.",
    
    "L1_a": (
        "You are a speech transcription system. "
        "Transcribe the following Nepali audio into Nepali text using Devanagari script. "
        "Produce a verbatim transcription. Do not translate. "
        "Return only the transcription, nothing else."
    ),
    "L1_b": (
        "Task: verbatim Nepali speech transcription.\\n"
        "Language: Nepali (Devanagari script).\\n"
        "Instructions: transcribe exactly what is spoken. Do not translate. "
        "Output only the transcription."
    ),
    
    "L2_a": (
        "Transcribe the spoken Nepali audio verbatim in Devanagari script. "
        "Preserve any English words in Latin script. "
        "Maintain the order of Nepali–English code-switching as spoken. "
        "Do not translate between languages. "
        "Keep fillers, repetitions, corrections, and incomplete words. "
        "Do not correct grammar. Do not infer inaudible words. "
        "Do not add timestamps, speaker labels, explanations, or confidence scores. "
        "Return only the transcription."
    ),
    "L2_b": (
        "You are a verbatim transcription system for Nepali speech.\\n"
        "Rules:\\n"
        "1. Write Nepali in Devanagari.\\n"
        "2. Write English words in Latin script.\\n"
        "3. Preserve code-switching order.\\n"
        "4. Do not translate.\\n"
        "5. Keep fillers, repetitions, corrections, incomplete words.\\n"
        "6. Do not correct grammar.\\n"
        "7. Do not guess inaudible words.\\n"
        "8. No timestamps, no speaker labels, no explanations.\\n"
        "9. Output only the transcription."
    ),
}

print(f"Defined {len(PROMPTS)} prompt variants.")
for pid, text in PROMPTS.items():
    print(f"  {pid}: {text[:60]}...")"""))

    cells.append(md("## D. Build Manifest"))
    cells.append(code(f"""import pandas as pd

def build_manifest(audio_dir, condition, max_files=None):
    '''Scan audio directory and build a manifest DataFrame, loading references from CSV if available.'''
    import glob, os
    
    # Try to load metadata
    metadata_df = None
    for meta_name in ["metadata.csv", "noisy_metadata.csv"]:
        meta_path = os.path.join(audio_dir, meta_name)
        if os.path.exists(meta_path):
            metadata_df = pd.read_csv(meta_path)
            # Ensure we have a consistent identifier to join on
            if "file" in metadata_df.columns:
                metadata_df["utterance_id"] = metadata_df["file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            break
            
    files = sorted(glob.glob(f"{{audio_dir}}/**/*.wav", recursive=True)) + \
            sorted(glob.glob(f"{{audio_dir}}/**/*.mp3", recursive=True))
            
    if max_files:
        files = files[:max_files]
        
    records = []
    for fp in files:
        uid = os.path.splitext(os.path.basename(fp))[0]
        
        # Look up reference
        ref_text = ""
        if metadata_df is not None and "utterance_id" in metadata_df.columns:
            match = metadata_df[metadata_df["utterance_id"] == uid]
            if not match.empty:
                # Use label_normalized if available, else reference
                if "label_normalized" in match.columns:
                    ref_text = str(match.iloc[0]["label_normalized"])
                elif "reference" in match.columns:
                    ref_text = str(match.iloc[0]["reference"])
                    
        records.append({{
            "utterance_id": uid,
            "audio_path": fp,
            "speech_condition": condition,
            "reference_raw": ref_text,
        }})
    return pd.DataFrame(records)


manifest_clean = build_manifest(f"{{AUDIO_BASE}}/clean_nepali_200_flat", "clean", max_files=5)
manifest_noisy = build_manifest(f"{{AUDIO_BASE}}/noisy_nepali_200", "noisy", max_files=5)
manifest_cs    = build_manifest(f"{{AUDIO_BASE}}/codeswitched_nepali_200_flat", "codeswitched", max_files=5)

manifest = pd.concat([manifest_clean, manifest_noisy, manifest_cs], ignore_index=True)
print(f"Pilot manifest: {{len(manifest)}} utterances")
print(manifest.speech_condition.value_counts().to_dict())"""))

    cells.append(md("## E. Batch Inference Pipeline"))
    cells.append(code("""import time, traceback
import jsonlines
from tqdm.auto import tqdm

def transcribe_one(audio_path, prompt_text):
    \"\"\"Run inference for a single audio + prompt pair.\"\"\"
    # Pass the FILE PATH directly — the processor loads audio internally.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    
    # tokenize=True lets the processor handle everything end-to-end.
    # load_audio_backend='soundfile' bypasses Kaggle's broken torchcodec.
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"load_audio_backend": "librosa"},
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=experiment_config["max_new_tokens"],
            do_sample=experiment_config["do_sample"],
        )
    
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def clean_output(raw_output):
    \"\"\"Basic output cleaning — remove markdown wrappers, labels, whitespace.\"\"\"
    text = raw_output.strip()
    # Remove common wrappers
    for prefix in ["Transcription:", "Output:", "```", "**"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip("`*\\n ")
    return text


def run_pipeline(manifest_df, prompts_dict, output_file):
    \"\"\"Run batch inference with checkpointing.\"\"\"
    output_path = f"{RESULTS_DIR}/{output_file}"
    
    # Load existing results for resume support
    completed = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            for obj in reader:
                completed.add((obj["utterance_id"], obj["prompt_id"]))
        print(f"Resuming: {len(completed)} already completed.")
    
    total = len(manifest_df) * len(prompts_dict)
    pbar = tqdm(total=total, desc="Inference")
    
    for _, row in manifest_df.iterrows():
        for prompt_id, prompt_text in prompts_dict.items():
            key = (row["utterance_id"], prompt_id)
            if key in completed:
                pbar.update(1)
                continue
            
            record = {
                "model_id": MODEL_ID,
                "utterance_id": row["utterance_id"],
                "prompt_id": prompt_id,
                "prompt_level": prompt_id.split("_")[0],
                "prompt_variant": prompt_id.split("_")[1] if "_" in prompt_id else "a",
                "audio_path": row["audio_path"],
                "speech_condition": row["speech_condition"],
                "reference_raw": row.get("reference_raw", ""),
                "status": "success",
                "raw_output": "",
                "cleaned_prediction": "",
                "inference_seconds": 0,
                "timestamp": datetime.now().isoformat(),
            }
            
            try:
                t0 = time.time()
                raw = transcribe_one(row["audio_path"], prompt_text)
                record["inference_seconds"] = round(time.time() - t0, 2)
                record["raw_output"] = raw
                record["cleaned_prediction"] = clean_output(raw)
                
                if not record["cleaned_prediction"]:
                    record["status"] = "empty_output"
                    
            except torch.cuda.OutOfMemoryError:
                record["status"] = "out_of_memory"
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                record["status"] = "inference_error"
                record["raw_output"] = str(e)
            
            # Append immediately for checkpoint
            with jsonlines.open(output_path, mode="a") as writer:
                writer.write(record)
            
            completed.add(key)
            pbar.update(1)
    
    pbar.close()
    print(f"\\n✅ Pipeline complete. {len(completed)} results saved to {output_path}")
    return output_path"""))

    cells.append(md("## F. Run Inference"))
    cells.append(code("""results_file = run_pipeline(manifest, PROMPTS, "raw_predictions.jsonl")"""))

    cells.append(md("## G. Compute Metrics"))
    cells.append(code("""from jiwer import wer, cer

def compute_metrics(ref, hyp):
    \"\"\"Compute WER and CER. Returns dict.\"\"\"
    if not ref or not hyp:
        return {"wer": 1.0, "cer": 1.0}
    try:
        w = wer(ref, hyp)
        c = cer(ref, hyp)
    except Exception:
        w, c = 1.0, 1.0
    return {"wer": round(w, 4), "cer": round(c, 4)}

# Load results and compute metrics
results = []
with jsonlines.open(f"{RESULTS_DIR}/raw_predictions.jsonl") as reader:
    for obj in reader:
        if obj["status"] == "success" and obj.get("reference_raw"):
            m = compute_metrics(obj["reference_raw"], obj["cleaned_prediction"])
            obj.update(m)
        results.append(obj)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/utterance_metrics.csv", index=False)

# Prompt-level summary
if "wer" in df.columns:
    summary = df[df["status"]=="success"].groupby(["prompt_id"]).agg(
        avg_wer=("wer", "mean"),
        avg_cer=("cer", "mean"),
        count=("utterance_id", "count"),
    ).reset_index().sort_values("avg_wer")
    summary.to_csv(f"{RESULTS_DIR}/prompt_summary.csv", index=False)
    print("\\n📊 Prompt Summary:")
    display(summary)
else:
    print("No reference transcriptions available — skipping WER/CER.")
    print("Status distribution:")
    print(df["status"].value_counts())"""))

    return nb(cells)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTEBOOK 02: Qwen2.5-Omni-3B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_nb02():
    cells = []
    cells.append(md("# 02 — Qwen2.5-Omni-3B: Prompt Development for Nepali ASR Benchmarking\n\n**Model:** `Qwen/Qwen2.5-Omni-3B`  \n**Architecture:** End-to-end multimodal (text + audio + image + video)  \n**Class:** `Qwen2_5OmniForConditionalGeneration` + `Qwen2_5OmniProcessor`  \n**Special:** Requires a specific `transformers` preview branch."))

    cells.append(md("## 0. Install Dependencies\n\n> ⚠️ Qwen2.5-Omni requires a **specific preview branch** of transformers. This cell installs it. **Restart runtime** after running."))
    cells.append(code("""import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", pkg])

# Qwen2.5-Omni requires this specific transformers preview
pip("git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview")
pip("accelerate>=1.6")

pip("bitsandbytes>=0.45")
pip("sentencepiece")
pip("protobuf")

# Qwen-specific audio utilities
pip("qwen-omni-utils[decord]")
pip("soundfile>=0.12")
pip("librosa>=0.10")
pip("scipy")
pip("einops")
pip("timm")

# Metrics
pip("jiwer>=3.1")
pip("jsonlines")

# Data
pip("pandas")
pip("tqdm")

print("\\n✅ All dependencies installed.")"""))

    cells.append(md("## A. Experiment Configuration"))
    cells.append(code(f"""import os, json, torch, gc
from datetime import datetime

MODEL_ID   = "Qwen/Qwen2.5-Omni-3B"
MODEL_REV  = "main"
QUANT      = "auto"

experiment_config = {{
    "model_id":       MODEL_ID,
    "model_revision": MODEL_REV,
    "quantization":   QUANT,
    "random_seed":    42,
    "audio_sr":       16000,
    "batch_size":     1,
    "max_new_tokens": 256,
    "temperature":    0.0,
    "do_sample":      False,
    "timestamp":      datetime.now().isoformat(),
}}

AUDIO_BASE = "{KAGGLE_AUDIO_BASE}"
RESULTS_DIR = "{KAGGLE_WORKING}/results/prompt_dev/qwen2_5_omni"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(f"{{RESULTS_DIR}}/run_config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)

print("Config saved →", RESULTS_DIR)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")"""))

    cells.append(md("## B. Model Loading"))
    cells.append(code("""from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

print(f"Loading {MODEL_ID} ...")

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",  # Use SDPA instead of flash_attention_2 for broader GPU support
)
model.disable_talker()  # We only need text output, not audio generation

print("✅ Model loaded successfully.")"""))

    cells.append(md("## B.1 — Quick Sanity Check"))
    cells.append(code(f"""import librosa, glob

audio_files = sorted(glob.glob(f"{{AUDIO_BASE}}/clean_nepali_200_flat/*.wav"))
if not audio_files:
    audio_files = sorted(glob.glob(f"{{AUDIO_BASE}}/clean_nepali_200_flat/*.mp3"))
test_audio_path = audio_files[0]
print(f"Testing: {{test_audio_path}}")

# Qwen2.5-Omni uses chat messages with audio_url
conversation = [
    {{
        "role": "system",
        "content": [
            {{"type": "text", "text": "You are a helpful assistant."}}
        ]
    }},
    {{
        "role": "user",
        "content": [
            {{"type": "audio", "audio": test_audio_path}},
            {{"type": "text", "text": "Transcribe the following Nepali speech verbatim in Devanagari script. Output ONLY the transcription, nothing else."}},
        ],
    }},
]

# Process multimodal info
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device).to(model.dtype)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
result = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
print(f"\\n📝 Model output:\\n{{result}}")"""))

    # Prompts, Manifest, Pipeline, Metrics — same structure as NB01 but adapted for Qwen
    cells.append(md("## C. Prompt Templates"))
    cells.append(code("""PROMPTS = {
    "L0_a": "Transcribe the following speech segment in its original language. Only output the transcription.",
    "L1_a": (
        "You are a speech transcription system. "
        "Transcribe the following Nepali audio into Nepali text using Devanagari script. "
        "Produce a verbatim transcription. Do not translate. "
        "Return only the transcription, nothing else."
    ),
    "L1_b": (
        "Task: verbatim Nepali speech transcription.\\n"
        "Language: Nepali (Devanagari script).\\n"
        "Instructions: transcribe exactly what is spoken. Do not translate. "
        "Output only the transcription."
    ),
    "L2_a": (
        "Transcribe the spoken Nepali audio verbatim in Devanagari script. "
        "Preserve any English words in Latin script. "
        "Maintain the order of Nepali–English code-switching as spoken. "
        "Do not translate between languages. "
        "Keep fillers, repetitions, corrections, and incomplete words. "
        "Do not correct grammar. Do not infer inaudible words. "
        "Do not add timestamps, speaker labels, explanations, or confidence scores. "
        "Return only the transcription."
    ),
    "L2_b": (
        "You are a verbatim transcription system for Nepali speech.\\n"
        "Rules:\\n"
        "1. Write Nepali in Devanagari.\\n"
        "2. Write English words in Latin script.\\n"
        "3. Preserve code-switching order.\\n"
        "4. Do not translate.\\n"
        "5. Keep fillers, repetitions, corrections, incomplete words.\\n"
        "6. Do not correct grammar.\\n"
        "7. Do not guess inaudible words.\\n"
        "8. No timestamps, no speaker labels, no explanations.\\n"
        "9. Output only the transcription."
    ),
}
print(f"Defined {len(PROMPTS)} prompt variants.")"""))

    cells.append(md("## D. Build Manifest"))
    cells.append(code(f"""import pandas as pd

def build_manifest(audio_dir, condition, max_files=None):
    '''Scan audio directory and build a manifest DataFrame, loading references from CSV if available.'''
    import glob, os
    
    # Try to load metadata
    metadata_df = None
    for meta_name in ["metadata.csv", "noisy_metadata.csv"]:
        meta_path = os.path.join(audio_dir, meta_name)
        if os.path.exists(meta_path):
            metadata_df = pd.read_csv(meta_path)
            # Ensure we have a consistent identifier to join on
            if "file" in metadata_df.columns:
                metadata_df["utterance_id"] = metadata_df["file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            break
            
    files = sorted(glob.glob(f"{{audio_dir}}/**/*.wav", recursive=True)) + \
            sorted(glob.glob(f"{{audio_dir}}/**/*.mp3", recursive=True))
            
    if max_files:
        files = files[:max_files]
        
    records = []
    for fp in files:
        uid = os.path.splitext(os.path.basename(fp))[0]
        
        # Look up reference
        ref_text = ""
        if metadata_df is not None and "utterance_id" in metadata_df.columns:
            match = metadata_df[metadata_df["utterance_id"] == uid]
            if not match.empty:
                # Use label_normalized if available, else reference
                if "label_normalized" in match.columns:
                    ref_text = str(match.iloc[0]["label_normalized"])
                elif "reference" in match.columns:
                    ref_text = str(match.iloc[0]["reference"])
                    
        records.append({{
            "utterance_id": uid,
            "audio_path": fp,
            "speech_condition": condition,
            "reference_raw": ref_text,
        }})
    return pd.DataFrame(records)


manifest_clean = build_manifest(f"{{AUDIO_BASE}}/clean_nepali_200_flat", "clean", max_files=5)
manifest_noisy = build_manifest(f"{{AUDIO_BASE}}/noisy_nepali_200", "noisy", max_files=5)
manifest_cs    = build_manifest(f"{{AUDIO_BASE}}/codeswitched_nepali_200_flat", "codeswitched", max_files=5)
manifest = pd.concat([manifest_clean, manifest_noisy, manifest_cs], ignore_index=True)
print(f"Pilot manifest: {{len(manifest)}} utterances")"""))

    cells.append(md("## E. Batch Inference Pipeline"))
    cells.append(code("""import time, traceback
import jsonlines
from tqdm.auto import tqdm

def transcribe_one(audio_path, prompt_text):
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    
    inputs = processor(text=text, audio=audios, images=images, videos=videos, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device).to(model.dtype)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=experiment_config["max_new_tokens"], do_sample=False)
    
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

def clean_output(raw):
    text = raw.strip()
    for prefix in ["Transcription:", "Output:", "```", "**"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip("`*\\n ")

def run_pipeline(manifest_df, prompts_dict, output_file):
    output_path = f"{RESULTS_DIR}/{output_file}"
    completed = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            for obj in reader:
                completed.add((obj["utterance_id"], obj["prompt_id"]))
        print(f"Resuming: {len(completed)} already completed.")
    
    total = len(manifest_df) * len(prompts_dict)
    pbar = tqdm(total=total, desc="Inference")
    
    for _, row in manifest_df.iterrows():
        for prompt_id, prompt_text in prompts_dict.items():
            key = (row["utterance_id"], prompt_id)
            if key in completed:
                pbar.update(1)
                continue
            record = {
                "model_id": MODEL_ID, "utterance_id": row["utterance_id"],
                "prompt_id": prompt_id, "prompt_level": prompt_id.split("_")[0],
                "audio_path": row["audio_path"], "speech_condition": row["speech_condition"],
                "reference_raw": row.get("reference_raw", ""),
                "status": "success", "raw_output": "", "cleaned_prediction": "",
                "inference_seconds": 0, "timestamp": datetime.now().isoformat(),
            }
            try:
                t0 = time.time()
                raw = transcribe_one(row["audio_path"], prompt_text)
                record["inference_seconds"] = round(time.time() - t0, 2)
                record["raw_output"] = raw
                record["cleaned_prediction"] = clean_output(raw)
                if not record["cleaned_prediction"]:
                    record["status"] = "empty_output"
            except torch.cuda.OutOfMemoryError:
                record["status"] = "out_of_memory"
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                record["status"] = "inference_error"
                record["raw_output"] = str(e)
            
            with jsonlines.open(output_path, mode="a") as writer:
                writer.write(record)
            completed.add(key)
            pbar.update(1)
    pbar.close()
    print(f"\\n✅ Done. {len(completed)} results → {output_path}")
    return output_path"""))

    cells.append(md("## F. Run Inference"))
    cells.append(code("""results_file = run_pipeline(manifest, PROMPTS, "raw_predictions.jsonl")"""))

    cells.append(md("## G. Compute Metrics"))
    cells.append(code("""from jiwer import wer, cer
import pandas as pd

def compute_metrics(ref, hyp):
    if not ref or not hyp:
        return {"wer": 1.0, "cer": 1.0}
    try:
        return {"wer": round(wer(ref, hyp), 4), "cer": round(cer(ref, hyp), 4)}
    except:
        return {"wer": 1.0, "cer": 1.0}

results = []
with jsonlines.open(f"{RESULTS_DIR}/raw_predictions.jsonl") as reader:
    for obj in reader:
        if obj["status"] == "success" and obj.get("reference_raw"):
            obj.update(compute_metrics(obj["reference_raw"], obj["cleaned_prediction"]))
        results.append(obj)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/utterance_metrics.csv", index=False)

if "wer" in df.columns:
    summary = df[df["status"]=="success"].groupby("prompt_id").agg(
        avg_wer=("wer","mean"), avg_cer=("cer","mean"), count=("utterance_id","count")
    ).reset_index().sort_values("avg_wer")
    summary.to_csv(f"{RESULTS_DIR}/prompt_summary.csv", index=False)
    display(summary)
else:
    print("Status distribution:"); print(df["status"].value_counts())"""))

    return nb(cells)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTEBOOK 03: Voxtral Mini 3B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_nb03():
    cells = []
    cells.append(md("# 03 — Voxtral Mini 3B: Prompt Development for Nepali ASR Benchmarking\n\n**Model:** `mistralai/Voxtral-Mini-3B-2507`  \n**Architecture:** Audio-language model (transcription + understanding)  \n**Class:** `VoxtralForConditionalGeneration` + `AutoProcessor`  \n**Special:** Has a dedicated `apply_transcription_request` method.  \n**Note:** Requires `transformers >= 4.54` and `mistral_common[audio]`"))

    cells.append(md("## 0. Install Dependencies"))
    cells.append(code("""import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", pkg])

pip("transformers>=4.54")
pip("accelerate>=1.6")

pip("mistral_common[audio]")  # Required for Voxtral audio processing
pip("bitsandbytes>=0.45")
pip("sentencepiece")
pip("protobuf")

pip("librosa>=0.10")
pip("soundfile>=0.12")
pip("scipy")

pip("jiwer>=3.1")
pip("jsonlines")
pip("pandas")
pip("tqdm")

print("\\n✅ All dependencies installed.")"""))

    cells.append(md("## A. Experiment Configuration"))
    cells.append(code(f"""import os, json, torch, gc
from datetime import datetime

MODEL_ID   = "mistralai/Voxtral-Mini-3B-2507"
MODEL_REV  = "main"
QUANT      = "bfloat16"

experiment_config = {{
    "model_id":       MODEL_ID,
    "model_revision": MODEL_REV,
    "quantization":   QUANT,
    "random_seed":    42,
    "audio_sr":       16000,
    "batch_size":     1,
    "max_new_tokens": 256,
    "temperature":    0.0,
    "do_sample":      False,
    "timestamp":      datetime.now().isoformat(),
}}

AUDIO_BASE = "{KAGGLE_AUDIO_BASE}"
RESULTS_DIR = "{KAGGLE_WORKING}/results/prompt_dev/voxtral_mini"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(f"{{RESULTS_DIR}}/run_config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)

print("Config saved →", RESULTS_DIR)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")"""))

    cells.append(md("## B. Model Loading"))
    cells.append(code("""from transformers import VoxtralForConditionalGeneration, AutoProcessor

print(f"Loading {MODEL_ID} ...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = VoxtralForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("✅ Model loaded successfully.")"""))

    cells.append(md("## B.1 — Sanity Check (Transcription Mode)"))
    cells.append(code(f"""import glob

audio_files = sorted(glob.glob(f"{{AUDIO_BASE}}/clean_nepali_200_flat/*.wav"))
if not audio_files:
    audio_files = sorted(glob.glob(f"{{AUDIO_BASE}}/clean_nepali_200_flat/*.mp3"))
test_audio_path = audio_files[0]
print(f"Testing: {{test_audio_path}}")

# Voxtral has a dedicated transcription method
inputs = processor.apply_transcription_request(
    language="ne",  # Nepali ISO code
    audio=test_audio_path,
    model_id=MODEL_ID,
)
inputs = inputs.to(model.device, dtype=torch.bfloat16)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256)

new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
result = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
print(f"\\n📝 Transcription mode output:\\n{{result}}")

# Also test with chat-based prompt for comparison
conversation = [
    {{
        "role": "user",
        "content": [
            {{"type": "audio_url", "audio_url": test_audio_path}},
            {{"type": "text", "text": "Transcribe this Nepali audio in Devanagari script."}},
        ],
    }},
]

inputs2 = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

with torch.no_grad():
    output_ids2 = model.generate(**inputs2, max_new_tokens=256)
new_tokens2 = output_ids2[:, inputs2["input_ids"].shape[1]:]
result2 = processor.batch_decode(new_tokens2, skip_special_tokens=True)[0]
print(f"\\n📝 Chat mode output:\\n{{result2}}")"""))

    # Prompts, Manifest, Pipeline — same structure as NB01
    cells.append(md("## C. Prompt Templates"))
    cells.append(code("""PROMPTS = {
    "L0_a": "Transcribe the following speech segment in its original language. Only output the transcription.",
    "L1_a": (
        "You are a speech transcription system. "
        "Transcribe the following Nepali audio into Nepali text using Devanagari script. "
        "Produce a verbatim transcription. Do not translate. "
        "Return only the transcription, nothing else."
    ),
    "L1_b": (
        "Task: verbatim Nepali speech transcription.\\n"
        "Language: Nepali (Devanagari script).\\n"
        "Instructions: transcribe exactly what is spoken. Do not translate. "
        "Output only the transcription."
    ),
    "L2_a": (
        "Transcribe the spoken Nepali audio verbatim in Devanagari script. "
        "Preserve any English words in Latin script. "
        "Maintain the order of Nepali–English code-switching as spoken. "
        "Do not translate between languages. "
        "Keep fillers, repetitions, corrections, and incomplete words. "
        "Do not correct grammar. Do not infer inaudible words. "
        "Do not add timestamps, speaker labels, explanations, or confidence scores. "
        "Return only the transcription."
    ),
    "L2_b": (
        "You are a verbatim transcription system for Nepali speech.\\n"
        "Rules:\\n"
        "1. Write Nepali in Devanagari.\\n"
        "2. Write English words in Latin script.\\n"
        "3. Preserve code-switching order.\\n"
        "4. Do not translate.\\n"
        "5. Keep fillers, repetitions, corrections, incomplete words.\\n"
        "6. Do not correct grammar.\\n"
        "7. Do not guess inaudible words.\\n"
        "8. No timestamps, no speaker labels, no explanations.\\n"
        "9. Output only the transcription."
    ),
}
print(f"Defined {len(PROMPTS)} prompt variants.")"""))

    cells.append(md("## D. Build Manifest"))
    cells.append(code(f"""import pandas as pd

def build_manifest(audio_dir, condition, max_files=None):
    '''Scan audio directory and build a manifest DataFrame, loading references from CSV if available.'''
    import glob, os
    
    # Try to load metadata
    metadata_df = None
    for meta_name in ["metadata.csv", "noisy_metadata.csv"]:
        meta_path = os.path.join(audio_dir, meta_name)
        if os.path.exists(meta_path):
            metadata_df = pd.read_csv(meta_path)
            # Ensure we have a consistent identifier to join on
            if "file" in metadata_df.columns:
                metadata_df["utterance_id"] = metadata_df["file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            break
            
    files = sorted(glob.glob(f"{{audio_dir}}/**/*.wav", recursive=True)) + \
            sorted(glob.glob(f"{{audio_dir}}/**/*.mp3", recursive=True))
            
    if max_files:
        files = files[:max_files]
        
    records = []
    for fp in files:
        uid = os.path.splitext(os.path.basename(fp))[0]
        
        # Look up reference
        ref_text = ""
        if metadata_df is not None and "utterance_id" in metadata_df.columns:
            match = metadata_df[metadata_df["utterance_id"] == uid]
            if not match.empty:
                # Use label_normalized if available, else reference
                if "label_normalized" in match.columns:
                    ref_text = str(match.iloc[0]["label_normalized"])
                elif "reference" in match.columns:
                    ref_text = str(match.iloc[0]["reference"])
                    
        records.append({{
            "utterance_id": uid,
            "audio_path": fp,
            "speech_condition": condition,
            "reference_raw": ref_text,
        }})
    return pd.DataFrame(records)


manifest = pd.concat([
    build_manifest(f"{{AUDIO_BASE}}/clean_nepali_200_flat", "clean", 5),
    build_manifest(f"{{AUDIO_BASE}}/noisy_nepali_200", "noisy", 5),
    build_manifest(f"{{AUDIO_BASE}}/codeswitched_nepali_200_flat", "codeswitched", 5),
], ignore_index=True)
print(f"Pilot manifest: {{len(manifest)}} utterances")"""))

    cells.append(md("## E. Batch Inference Pipeline"))
    cells.append(code("""import time, jsonlines
from tqdm.auto import tqdm

def transcribe_one(audio_path, prompt_text):
    \"\"\"Use Voxtral chat mode for custom prompts.\"\"\"
    conversation = [
        {"role": "user", "content": [
            {"type": "audio_url", "audio_url": audio_path},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=experiment_config["max_new_tokens"], do_sample=False)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

def clean_output(raw):
    text = raw.strip()
    for prefix in ["Transcription:", "Output:", "```", "**"]:
        if text.startswith(prefix): text = text[len(prefix):]
    return text.strip("`*\\n ")

def run_pipeline(manifest_df, prompts_dict, output_file):
    output_path = f"{RESULTS_DIR}/{output_file}"
    completed = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            for obj in reader:
                completed.add((obj["utterance_id"], obj["prompt_id"]))
    
    pbar = tqdm(total=len(manifest_df)*len(prompts_dict), initial=len(completed), desc="Inference")
    for _, row in manifest_df.iterrows():
        for pid, ptxt in prompts_dict.items():
            if (row["utterance_id"], pid) in completed:
                pbar.update(1); continue
            rec = {"model_id": MODEL_ID, "utterance_id": row["utterance_id"], "prompt_id": pid,
                   "prompt_level": pid.split("_")[0], "audio_path": row["audio_path"],
                   "speech_condition": row["speech_condition"], "reference_raw": row.get("reference_raw",""),
                   "status": "success", "raw_output": "", "cleaned_prediction": "",
                   "inference_seconds": 0, "timestamp": datetime.now().isoformat()}
            try:
                t0 = time.time()
                raw = transcribe_one(row["audio_path"], ptxt)
                rec["inference_seconds"] = round(time.time()-t0, 2)
                rec["raw_output"] = raw
                rec["cleaned_prediction"] = clean_output(raw)
                if not rec["cleaned_prediction"]: rec["status"] = "empty_output"
            except torch.cuda.OutOfMemoryError:
                rec["status"] = "out_of_memory"; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                rec["status"] = "inference_error"; rec["raw_output"] = str(e)
            with jsonlines.open(output_path, mode="a") as w: w.write(rec)
            completed.add((row["utterance_id"], pid)); pbar.update(1)
    pbar.close()
    print(f"\\n✅ Done. {len(completed)} results → {output_path}")"""))

    cells.append(md("## F. Run Inference"))
    cells.append(code("""run_pipeline(manifest, PROMPTS, "raw_predictions.jsonl")"""))

    cells.append(md("## G. Compute Metrics"))
    cells.append(code("""from jiwer import wer, cer
import pandas as pd

results = []
with jsonlines.open(f"{RESULTS_DIR}/raw_predictions.jsonl") as reader:
    for obj in reader:
        if obj["status"]=="success" and obj.get("reference_raw"):
            try: obj.update({"wer": round(wer(obj["reference_raw"], obj["cleaned_prediction"]),4), "cer": round(cer(obj["reference_raw"], obj["cleaned_prediction"]),4)})
            except: obj.update({"wer":1.0,"cer":1.0})
        results.append(obj)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/utterance_metrics.csv", index=False)
if "wer" in df.columns:
    summary = df[df["status"]=="success"].groupby("prompt_id").agg(avg_wer=("wer","mean"),avg_cer=("cer","mean"),count=("utterance_id","count")).reset_index().sort_values("avg_wer")
    summary.to_csv(f"{RESULTS_DIR}/prompt_summary.csv", index=False)
    display(summary)
else:
    print(df["status"].value_counts())"""))

    return nb(cells)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTEBOOK 04: Phi-4 Multimodal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_nb04():
    cells = []
    cells.append(md("# 04 — Phi-4 Multimodal: Prompt Development for Nepali ASR Benchmarking\n\n**Model:** `microsoft/Phi-4-multimodal-instruct`  \n**Architecture:** Multimodal (text + image + audio), max 40s audio  \n**Class:** `AutoModelForCausalLM` + `AutoProcessor` (trust_remote_code)  \n**Audio input:** Uses `<|audio_1|>` placeholder token"))

    cells.append(md("## 0. Install Dependencies"))
    cells.append(code("""import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", pkg])

pip("transformers==4.48.2") # MUST be 4.48.2 for Phi-4 remote code!
pip("accelerate>=1.6")

pip("peft")          # Required by Phi-4 multimodal
pip("torchao>=0.16.0") # Required by latest PEFT
pip("soundfile>=0.12")
pip("scipy")
pip("librosa>=0.10")
pip("bitsandbytes>=0.45")
pip("sentencepiece")
pip("protobuf")
pip("backoff")  # Required by Phi-4 remote code



pip("jiwer>=3.1")
pip("jsonlines")
pip("pandas")
pip("tqdm")

print("\\n✅ All dependencies installed.")"""))

    cells.append(md("## A. Experiment Configuration"))
    cells.append(code(f"""import os, json, torch, gc
from datetime import datetime

MODEL_ID   = "microsoft/Phi-4-multimodal-instruct"
MODEL_REV  = "main"
QUANT      = "auto"

experiment_config = {{
    "model_id":       MODEL_ID,
    "model_revision": MODEL_REV,
    "quantization":   QUANT,
    "random_seed":    42,
    "audio_sr":       16000,
    "batch_size":     1,
    "max_new_tokens": 256,
    "temperature":    0.0,
    "do_sample":      False,
    "timestamp":      datetime.now().isoformat(),
}}

AUDIO_BASE = "{KAGGLE_AUDIO_BASE}"
RESULTS_DIR = "{KAGGLE_WORKING}/results/prompt_dev/phi4_multimodal"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(f"{{RESULTS_DIR}}/run_config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)

print("Config saved →", RESULTS_DIR)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")"""))

    cells.append(md("## B. Model Loading"))
    cells.append(code("""from transformers import AutoModelForCausalLM, AutoProcessor

print(f"Loading {MODEL_ID} ...")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    _attn_implementation="eager",
)

# Phi-4 Multimodal uses a mixture of LoRAs. We MUST load the speech adapter!
print("Loading speech adapter...")
from huggingface_hub import snapshot_download
import os
adapter_path = snapshot_download(MODEL_ID, allow_patterns=["speech-lora/*"])
adapter_path = os.path.join(adapter_path, "speech-lora")
model.load_adapter(adapter_path, adapter_name="speech")
model.set_adapter("speech")

print("✅ Model loaded successfully.")"""))

    cells.append(md("## B.1 — Sanity Check"))
    cells.append(code(f"""import soundfile as sf
import glob

audio_files = sorted(glob.glob(f"{{AUDIO_BASE}}/clean_nepali_200_flat/*.wav"))
if not audio_files:
    audio_files = sorted(glob.glob(f"{{AUDIO_BASE}}/clean_nepali_200_flat/*.mp3"))
test_audio_path = audio_files[0]
print(f"Testing: {{test_audio_path}}")

# Load audio
audio_data, samplerate = sf.read(test_audio_path)
print(f"Audio: {{len(audio_data)/samplerate:.2f}}s at {{samplerate}} Hz")

# Phi-4 strictly requires the full system + user + assistant structure
prompt = "<|system|>You are a helpful assistant.<|end|><|user|><|audio_1|>Transcribe the following Nepali audio into Nepali text.<|end|><|assistant|>"

    inputs = processor(
        text=prompt,
        audios=[(audio_data, samplerate)],
        return_tensors="pt"
    ).to(model.device)

from transformers import GenerationConfig
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        generation_config=generation_config,
    )

result = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(f"\\n📝 Model output:\\n{{result}}")"""))

    cells.append(md("## C. Prompt Templates\n\nNote: Phi-4 uses `<|system|>...<|end|><|user|><|audio_1|>...<|end|><|assistant|>` as the chat format."))
    cells.append(code("""user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

PROMPTS = {
    "L0_a": f"{SYS}<|user|><|audio_1|>Transcribe the following Nepali audio into Nepali text.<|end|><|assistant|>",
    "L1_a": (
        f"{SYS}<|user|><|audio_1|>You are a speech transcription system. "
        "Transcribe the following Nepali audio into Nepali text using Devanagari script. "
        "Produce a verbatim transcription. Do not translate. "
        "Return only the transcription, nothing else.<|end|><|assistant|>"
    ),
    "L1_b": (
        f"{SYS}<|user|><|audio_1|>Task: verbatim Nepali speech transcription.\\n"
        "Language: Nepali (Devanagari script).\\n"
        "Instructions: transcribe exactly what is spoken. Do not translate. "
        "Output only the transcription.<|end|><|assistant|>"
    ),
    "L2_a": (
        f"{SYS}<|user|><|audio_1|>Transcribe the spoken Nepali audio verbatim in Devanagari script. "
        "Preserve any English words in Latin script. "
        "Maintain the order of Nepali–English code-switching as spoken. "
        "Do not translate between languages. "
        "Keep fillers, repetitions, corrections, and incomplete words. "
        "Do not correct grammar. Do not infer inaudible words. "
        "Do not add timestamps, speaker labels, explanations, or confidence scores. "
        "Return only the transcription.<|end|><|assistant|>"
    ),
    "L2_b": (
        f"{SYS}<|user|><|audio_1|>You are a verbatim transcription system for Nepali speech.\\n"
        "Rules:\\n"
        "1. Write Nepali in Devanagari.\\n"
        "2. Write English words in Latin script.\\n"
        "3. Preserve code-switching order.\\n"
        "4. Do not translate.\\n"
        "5. Keep fillers, repetitions, corrections, incomplete words.\\n"
        "6. Do not correct grammar.\\n"
        "7. Do not guess inaudible words.\\n"
        "8. No timestamps, no speaker labels, no explanations.\\n"
        "9. Output only the transcription.<|end|><|assistant|>"
    ),
}
print(f"Defined {len(PROMPTS)} prompt variants.")"""))

    cells.append(md("## D. Build Manifest"))
    cells.append(code(f"""import pandas as pd

def build_manifest(audio_dir, condition, max_files=None):
    '''Scan audio directory and build a manifest DataFrame, loading references from CSV if available.'''
    import glob, os
    
    # Try to load metadata
    metadata_df = None
    for meta_name in ["metadata.csv", "noisy_metadata.csv"]:
        meta_path = os.path.join(audio_dir, meta_name)
        if os.path.exists(meta_path):
            metadata_df = pd.read_csv(meta_path)
            # Ensure we have a consistent identifier to join on
            if "file" in metadata_df.columns:
                metadata_df["utterance_id"] = metadata_df["file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            break
            
    files = sorted(glob.glob(f"{{audio_dir}}/**/*.wav", recursive=True)) + \
            sorted(glob.glob(f"{{audio_dir}}/**/*.mp3", recursive=True))
            
    if max_files:
        files = files[:max_files]
        
    records = []
    for fp in files:
        uid = os.path.splitext(os.path.basename(fp))[0]
        
        # Look up reference
        ref_text = ""
        if metadata_df is not None and "utterance_id" in metadata_df.columns:
            match = metadata_df[metadata_df["utterance_id"] == uid]
            if not match.empty:
                # Use label_normalized if available, else reference
                if "label_normalized" in match.columns:
                    ref_text = str(match.iloc[0]["label_normalized"])
                elif "reference" in match.columns:
                    ref_text = str(match.iloc[0]["reference"])
                    
        records.append({{
            "utterance_id": uid,
            "audio_path": fp,
            "speech_condition": condition,
            "reference_raw": ref_text,
        }})
    return pd.DataFrame(records)


manifest = pd.concat([
    build_manifest(f"{{AUDIO_BASE}}/clean_nepali_200_flat", "clean", 5),
    build_manifest(f"{{AUDIO_BASE}}/noisy_nepali_200", "noisy", 5),
    build_manifest(f"{{AUDIO_BASE}}/codeswitched_nepali_200_flat", "codeswitched", 5),
], ignore_index=True)
print(f"Pilot manifest: {{len(manifest)}} utterances")"""))

    cells.append(md("## E. Batch Inference Pipeline"))
    cells.append(code("""import time, jsonlines, soundfile as sf
from tqdm.auto import tqdm

def transcribe_one(audio_path, prompt_text):
    audio_data, samplerate = sf.read(audio_path)
    inputs = processor(text=prompt_text, audios=[(audio_data, samplerate)], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=experiment_config["max_new_tokens"], generation_config=generation_config)
    return processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

def clean_output(raw):
    text = raw.strip()
    for prefix in ["Transcription:", "Output:", "```", "**"]:
        if text.startswith(prefix): text = text[len(prefix):]
    return text.strip("`*\\n ")

def run_pipeline(manifest_df, prompts_dict, output_file):
    output_path = f"{RESULTS_DIR}/{output_file}"
    completed = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            for obj in reader: completed.add((obj["utterance_id"], obj["prompt_id"]))
    
    pbar = tqdm(total=len(manifest_df)*len(prompts_dict), initial=len(completed), desc="Inference")
    for _, row in manifest_df.iterrows():
        for pid, ptxt in prompts_dict.items():
            if (row["utterance_id"], pid) in completed: pbar.update(1); continue
            rec = {"model_id": MODEL_ID, "utterance_id": row["utterance_id"], "prompt_id": pid,
                   "prompt_level": pid.split("_")[0], "audio_path": row["audio_path"],
                   "speech_condition": row["speech_condition"], "reference_raw": row.get("reference_raw",""),
                   "status": "success", "raw_output": "", "cleaned_prediction": "",
                   "inference_seconds": 0, "timestamp": datetime.now().isoformat()}
            try:
                t0 = time.time()
                raw = transcribe_one(row["audio_path"], ptxt)
                rec["inference_seconds"] = round(time.time()-t0, 2)
                rec["raw_output"] = raw; rec["cleaned_prediction"] = clean_output(raw)
                if not rec["cleaned_prediction"]: rec["status"] = "empty_output"
            except torch.cuda.OutOfMemoryError:
                rec["status"] = "out_of_memory"; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                rec["status"] = "inference_error"; rec["raw_output"] = str(e)
            with jsonlines.open(output_path, mode="a") as w: w.write(rec)
            completed.add((row["utterance_id"], pid)); pbar.update(1)
    pbar.close()
    print(f"\\n✅ Done. {len(completed)} results → {output_path}")"""))

    cells.append(md("## F. Run Inference"))
    cells.append(code("""run_pipeline(manifest, PROMPTS, "raw_predictions.jsonl")"""))

    cells.append(md("## G. Compute Metrics"))
    cells.append(code("""from jiwer import wer, cer

results = []
with jsonlines.open(f"{RESULTS_DIR}/raw_predictions.jsonl") as reader:
    for obj in reader:
        if obj["status"]=="success" and obj.get("reference_raw"):
            try: obj.update({"wer": round(wer(obj["reference_raw"], obj["cleaned_prediction"]),4), "cer": round(cer(obj["reference_raw"], obj["cleaned_prediction"]),4)})
            except: obj.update({"wer":1.0,"cer":1.0})
        results.append(obj)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/utterance_metrics.csv", index=False)
if "wer" in df.columns:
    summary = df[df["status"]=="success"].groupby("prompt_id").agg(avg_wer=("wer","mean"),avg_cer=("cer","mean"),count=("utterance_id","count")).reset_index().sort_values("avg_wer")
    summary.to_csv(f"{RESULTS_DIR}/prompt_summary.csv", index=False)
    display(summary)
else:
    print(df["status"].value_counts())"""))

    return nb(cells)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTEBOOK 05: Prompt Comparison & Freeze
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_nb05():
    cells = []
    cells.append(md("# 05 — Prompt Comparison & Freezing\n\nThis notebook aggregates the prompt-development results from all 4 models and selects the best prompt per level per model."))

    cells.append(code("""import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "jsonlines", "jiwer"])

print("✅ Dependencies ready.")"""))

    cells.append(code(f"""import pandas as pd
import os, json

RESULTS_BASE = "{KAGGLE_WORKING}/results/prompt_dev"
MODELS = ["gemma4_12b", "qwen2_5_omni", "voxtral_mini", "phi4_multimodal"]

all_summaries = []
for model in MODELS:
    path = f"{{RESULTS_BASE}}/{{model}}/prompt_summary.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["model"] = model
        all_summaries.append(df)
        print(f"✅ Loaded {{model}}: {{len(df)}} prompt variants")
    else:
        print(f"⚠️  Missing: {{path}}")

if all_summaries:
    combined = pd.concat(all_summaries, ignore_index=True)
    print(f"\\n📊 Combined results: {{len(combined)}} rows")
    display(combined.sort_values(["model", "prompt_id"]))
else:
    print("No results found. Run notebooks 01-04 first.")
    combined = pd.DataFrame()"""))

    cells.append(md("## Select Best Prompt Per Level Per Model"))
    cells.append(code(f"""if len(combined) > 0:
    # Extract level from prompt_id
    combined["level"] = combined["prompt_id"].str.split("_").str[0]
    
    # For each model+level, pick the prompt with lowest avg_wer
    best = combined.sort_values("avg_wer").groupby(["model", "level"]).first().reset_index()
    print("\\n🏆 Best prompt per level per model:")
    display(best[["model", "level", "prompt_id", "avg_wer", "avg_cer", "count"]])
    
    # Build frozen registry
    registry = {{"models": {{}}}}
    for _, row in best.iterrows():
        if row["model"] not in registry["models"]:
            registry["models"][row["model"]] = {{}}
        registry["models"][row["model"]][row["level"]] = row["prompt_id"]
    
    os.makedirs("{KAGGLE_WORKING}/configs", exist_ok=True)
    with open("{KAGGLE_WORKING}/configs/frozen_prompt_registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print("\\n📁 Frozen registry saved to configs/frozen_prompt_registry.json")
    print(json.dumps(registry, indent=2))
else:
    print("No data to process.")"""))

    cells.append(md("## Status & Failure Summary"))
    cells.append(code(f"""for model in MODELS:
    found = False
    for search_base in SEARCH_PATHS:
        for base in glob.glob(search_base):
            path = f"{{base}}/{{model}}/utterance_metrics.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"\\n=== {{model}} ===")
                print(df["status"].value_counts())
                found = True
                break
        if found: break
    if not found:
        print(f"{{model}}: no data")"""))

    return nb(cells)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERATE ALL NOTEBOOKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.makedirs("notebooks", exist_ok=True)

notebooks = [
    ("notebooks/01_gemma4_12b_prompt_dev.ipynb",      build_nb01),
    ("notebooks/02_qwen2_5_omni_prompt_dev.ipynb",     build_nb02),
    ("notebooks/03_voxtral_mini_prompt_dev.ipynb",      build_nb03),
    ("notebooks/04_phi4_multimodal_prompt_dev.ipynb",   build_nb04),
    ("notebooks/05_prompt_comparison_and_freeze.ipynb",  build_nb05),
]

for path, builder in notebooks:
    notebook = builder()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"✅ {path}")

print("\n🎉 All 5 notebooks generated successfully!")
