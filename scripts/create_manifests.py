import os
import pandas as pd
import numpy as np

# Determine base directory
kaggle_path = "/kaggle/input/datasets/panditaadarsh/llm-bechmarking-audio/"
if os.path.exists(kaggle_path):
    base_dir = kaggle_path
else:
    # Try to get from environment, else use a relative path
    base_dir = os.getenv("DATASET_BASE_DIR", "./drive-download-20260731T192318Z-1-001")

# Define the sets
sets = [
    {
        "name": "clean",
        "dir": os.path.join(base_dir, "clean_nepali_200_flat"),
        "metadata": os.path.join(base_dir, "clean_nepali_200_flat", "metadata.csv"),
        "audio_subdir": "",  # audio files directly in the dir
        "condition": "clean",
        "has_snr": False,
        "has_noise_type": False
    },
    {
        "name": "codeswitched",
        "dir": os.path.join(base_dir, "codeswitched_nepali_200_flat"),
        "metadata": os.path.join(base_dir, "codeswitched_nepali_200_flat", "metadata.csv"),
        "audio_subdir": "",
        "condition": "codeswitched",
        "has_snr": False,
        "has_noise_type": False
    },
    {
        "name": "noisy",
        "dir": os.path.join(base_dir, "noisy_nepali_200"),
        "metadata": os.path.join(base_dir, "noisy_nepali_200", "noisy_metadata.csv"),
        "audio_subdir": "audio",  # audio files in audio subdir
        "condition": "noisy",
        "has_snr": True,
        "has_noise_type": True
    }
]

for s in sets:
    print(f"Processing {s['name']} set...")
    df = pd.read_csv(s["metadata"])

    # Determine audio path
    if s["audio_subdir"]:
        df["audio_path"] = df["file"].apply(lambda x: os.path.join(s["audio_subdir"], x))
    else:
        df["audio_path"] = df["file"]

    # Make audio_path absolute
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.join(s["dir"], x) if not os.path.isabs(x) else x)

    # utterance_id: filename without extension
    df["utterance_id"] = df["file"].apply(lambda x: os.path.splitext(x)[0])

    # reference: use label_normalized
    df["reference"] = df["label_normalized"]

    # duration: from duration column
    df["duration"] = df["duration"]

    # source_dataset: from source column
    df["source_dataset"] = df["source"]

    # gender: from gender column
    df["gender"] = df["gender"]

    # duration_bin: from duration_bin column
    df["duration_bin"] = df["duration_bin"]

    # condition: constant
    df["condition"] = s["condition"]

    # snr_db: if available
    if s["has_snr"]:
        df["snr_db"] = df["snr_db"]
    else:
        df["snr_db"] = np.nan

    # noise_type: if available
    if s["has_noise_type"]:
        df["noise_type"] = df["noise_type"]
    else:
        df["noise_type"] = np.nan

    # cmi: from cmi column (if exists)
    if "cmi" in df.columns:
        df["cmi"] = df["cmi"]
    else:
        df["cmi"] = np.nan

    # Select columns in the desired order
    manifest_cols = [
        "audio_path",
        "utterance_id",
        "reference",
        "duration",
        "source_dataset",
        "gender",
        "duration_bin",
        "condition",
        "snr_db",
        "noise_type",
        "cmi"
    ]

    manifest_df = df[manifest_cols]

    # Save to CSV
    output_path = os.path.join(".", f"{s['name']}_manifest.csv")
    manifest_df.to_csv(output_path, index=False)
    print(f"Saved manifest to {output_path}")
    print(f"Number of utterances: {len(manifest_df)}")
    print()

# Stage 1 prompt development set
print("Processing stage1 prompt development set...")
stage1_metadata = os.path.join(base_dir, "prompt_development_sets", "stage1", "metadata.csv")
stage1_audio_dir = os.path.join(base_dir, "prompt_development_sets", "stage1")
df_stage1 = pd.read_csv(stage1_metadata)

# Determine audio path for stage1
df_stage1["audio_path"] = df_stage1["file"].apply(lambda x: os.path.join(stage1_audio_dir, x))

# utterance_id
df_stage1["utterance_id"] = df_stage1["file"].apply(lambda x: os.path.splitext(x)[0])

# reference
df_stage1["reference"] = df_stage1["label_normalized"]

# duration
df_stage1["duration"] = df_stage1["duration"]

# source_dataset
df_stage1["source_dataset"] = df_stage1["source"]

# gender
df_stage1["gender"] = df_stage1["gender"]

# duration_bin
df_stage1["duration_bin"] = df_stage1["duration_bin"]

# condition: we'll set based on the original set? Or leave as unknown? We'll set to 'prompt_dev_stage1'
df_stage1["condition"] = "prompt_dev_stage1"

# snr_db and noise_type: not available in stage1 metadata? We'll set to NaN
df_stage1["snr_db"] = np.nan
df_stage1["noise_type"] = np.nan

# cmi: if available
if "cmi" in df_stage1.columns:
    df_stage1["cmi"] = df_stage1["cmi"]
else:
    df_stage1["cmi"] = np.nan

# Select columns
stage1_cols = [
    "audio_path",
    "utterance_id",
    "reference",
    "duration",
    "source_dataset",
    "gender",
    "duration_bin",
    "condition",
    "snr_db",
    "noise_type",
    "cmi"
]

stage1_manifest = df_stage1[stage1_cols]
stage1_manifest.to_csv(os.path.join(".", "stage1_manifest.csv"), index=False)
print(f"Saved stage1 manifest to ./stage1_manifest.csv")
print(f"Number of utterances: {len(stage1_manifest)}")
print()

# Stage 2 prompt development set
print("Processing stage2 prompt development set...")
stage2_metadata = os.path.join(base_dir, "prompt_development_sets", "stage2", "metadata.csv")
stage2_audio_dir = os.path.join(base_dir, "prompt_development_sets", "stage2")
df_stage2 = pd.read_csv(stage2_metadata)

# Determine audio path for stage2
df_stage2["audio_path"] = df_stage2["file"].apply(lambda x: os.path.join(stage2_audio_dir, x))

# utterance_id
df_stage2["utterance_id"] = df_stage2["file"].apply(lambda x: os.path.splitext(x)[0])

# reference
df_stage2["reference"] = df_stage2["label_normalized"]

# duration
df_stage2["duration"] = df_stage2["duration"]

# source_dataset
df_stage2["source_dataset"] = df_stage2["source"]

# gender
df_stage2["gender"] = df_stage2["gender"]

# duration_bin
df_stage2["duration_bin"] = df_stage2["duration_bin"]

# condition
df_stage2["condition"] = "prompt_dev_stage2"

# snr_db and noise_type: not available in stage2 metadata? We'll set to NaN
df_stage2["snr_db"] = np.nan
df_stage2["noise_type"] = np.nan

# cmi: if available
if "cmi" in df_stage2.columns:
    df_stage2["cmi"] = df_stage2["cmi"]
else:
    df_stage2["cmi"] = np.nan

# Select columns
stage2_cols = [
    "audio_path",
    "utterance_id",
    "reference",
    "duration",
    "source_dataset",
    "gender",
    "duration_bin",
    "condition",
    "snr_db",
    "noise_type",
    "cmi"
]

stage2_manifest = df_stage2[stage2_cols]
stage2_manifest.to_csv(os.path.join(".", "stage2_manifest.csv"), index=False)
print(f"Saved stage2 manifest to ./stage2_manifest.csv")
print(f"Number of utterances: {len(stage2_manifest)}")
print()

print("Manifest creation complete.")