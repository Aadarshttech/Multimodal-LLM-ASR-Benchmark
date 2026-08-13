# Pre-Benchmark Prompt Development and Inference Pipeline Guide

## Objective

Before benchmarking, the goal is to:

1. Make all four audio-LLM models run reliably.
2. Build a resumable batch-inference pipeline.
3. Define equivalent prompt-complexity levels.
4. Select the best prompt wording within every level.
5. Validate zero-shot and few-shot feasibility.
6. Freeze all prompts, decoding settings, preprocessing, and output-cleaning rules.

The final 500-utterance set must not be used to modify prompts.

---

## 1. Notebook Structure

Create five notebooks:

```text
01_gemma4_12b_prompt_dev.ipynb
02_qwen2_5_omni_prompt_dev.ipynb
03_voxtral_mini_prompt_dev.ipynb
04_phi4_multimodal_prompt_dev.ipynb
05_prompt_comparison_and_freeze.ipynb
```

The four model notebooks should follow the same structure and produce the same output schema. Model loading and chat formatting may differ, but the experiment logic must remain consistent.

---

## 2. Structure of Each Model Notebook

### A. Experiment configuration

Record:

* Model name and exact revision
* Quantization method
* Transformers and dependency versions
* GPU type
* Random seed
* Audio sampling rate
* Batch size
* Maximum output tokens
* Temperature and sampling settings
* Chat template
* Prompt-development manifest version

Prefer deterministic decoding, such as temperature zero or disabled sampling.

### B. Model-loading test

Verify:

* The model loads without errors.
* One audio file can be processed.
* The model produces Nepali text.
* Audio duration and format are accepted.
* GPU memory remains within the Kaggle limit.

Test at least:

* One clean utterance
* One noisy utterance
* One code-switched utterance

### C. Batch-inference pipeline

The pipeline should:

* Accept a manifest and prompt configuration
* Run audio in small batches
* Save results after every batch
* Skip already completed combinations
* Continue after individual failures
* Record out-of-memory and decoding errors
* Support restarting the notebook without losing completed work

A unique result should be identified by:

```text
model_id + utterance_id + prompt_id
```

Do not calculate metrics before saving the original model output.

---

## 3. Prompt Levels

The levels should represent increasing amounts of task guidance.

### Level 0 — Minimal zero-shot

Purpose: measure performance with almost no guidance.

Example structure:

> Transcribe the audio.

Use one candidate unless the model clearly requires the language to be named.

### Level 1 — Structured zero-shot

Include:

* The task is speech transcription.
* The primary language is Nepali.
* Nepali should be written in Devanagari.
* The transcription should be verbatim.
* The model must not translate.
* Only the transcription should be returned.

Create two wording variants.

The variants should contain the same information but differ in brevity and instruction structure.

### Level 2 — Linguistically aware zero-shot

Build this prompt from Level 1 and add rules connected to measurable Nepali ASR behavior:

1. Write spoken Nepali in Devanagari.
2. Preserve spoken English words in Latin script.
3. Preserve Nepali–English code-switching in the order spoken.
4. Do not translate English into Nepali or Nepali into English.
5. Preserve audible fillers, repetitions, corrections, and incomplete words.
6. Do not correct the speaker’s grammar.
7. Do not infer words that are not clearly audible.
8. Follow the dataset’s number and punctuation conventions.
9. Do not add timestamps, speaker labels, explanations, or confidence statements.
10. Return only the transcription.

Create two variants:

* A concise linguistically aware prompt
* A more explicit rule-based prompt

Do not add general descriptions of Nepali grammar that cannot be evaluated.

The treatment of numbers, punctuation, abbreviations, and English terms must match the reference-transcript policy. Inspect the dataset annotations before finalizing these instructions.

### Level 3 — One-shot

Use the selected Level 2 instructions and add one fixed audio-transcript demonstration.

A true ASR demonstration should include:

* One demonstration audio file
* Its correct reference transcription
* The new query audio

The demonstration should contain:

* Nepali speech
* At least one English code-switched word
* A relevant annotation feature such as a number, name, filler, or repetition

Test two possible demonstration utterances and retain the more stable one.

The demonstration must:

* Come from outside the prompt-development and final benchmark evaluation sets
* Remain identical for every benchmark utterance
* Never be selected based on the current test audio

A text-only example is not equivalent to a true audio-transcript few-shot example.

### Level 4 — Three-shot, optional

Test three-shot only when:

* The model supports multiple audio inputs correctly.
* Context length is sufficient.
* GPU memory remains stable.
* One-shot does not cause example copying.
* The format can be implemented equivalently across models.

Use three fixed demonstrations covering:

1. Standard Nepali speech
2. Nepali–English code-switching
3. Numbers, names, fillers, repetitions, or difficult audio

Do not use five-shot during the main study unless three-shot produces a clear and consistent improvement.

If one or more models cannot support true multi-audio few-shot prompting, either exclude three-shot from the main cross-model comparison or report it as an auxiliary model-specific experiment.

---

## 4. Reduced-Cost Prompt Development

To avoid testing every candidate extensively on every model, use two phases.

### Phase A — Prompt discovery

Use Gemma 4 12B and Qwen2.5-Omni-3B.

Test:

* One Level 0 prompt
* Two Level 1 variants
* Two Level 2 variants
* Two candidate one-shot demonstrations
* Three-shot only after one-shot succeeds

Start with approximately 10–15 representative pilot utterances containing:

* Clean speech
* Noisy speech
* Code-switched speech
* Short and long utterances
* Different dataset sources

Remove prompts that frequently:

* Produce explanations
* Translate the audio
* Add speaker labels
* Return empty output
* Refuse the task
* Copy demonstration text
* Generate clearly unrelated content

Run the surviving variants on the complete prompt-selection portion of the development split.

### Phase B — Cross-model verification

Transfer the strongest one or two candidates per level to Voxtral and Phi-4.

Run them first on the pilot subset. If both are stable, evaluate the finalists on the complete prompt-selection split.

This reduces unnecessary generations while still allowing the final selected wording to differ between models when necessary.

---

## 5. Output Processing

Save both the original and extracted output:

```text
raw_output
cleaned_prediction
```

The raw output must never be overwritten.

The output-cleaning stage may remove:

* Markdown wrappers
* Labels such as “Transcription:”
* Leading or trailing whitespace
* Repeated formatting tokens

It should not silently:

* Correct Nepali spelling
* Translate words
* Remove English tokens
* Replace the model’s chosen words
* Delete genuine repetitions

Use one shared normalization policy for every model.

Save:

```text
reference_raw
reference_normalized
prediction_raw
prediction_normalized
```

---

## 6. Metrics and Behavioral Checks

Each model notebook should calculate development-set metrics for prompt selection.

### Core metrics

* Word Error Rate
* Character Error Rate
* Word insertions
* Word deletions
* Word substitutions
* Character-level errors

Calculate results:

* Per utterance
* Per prompt variant
* Per prompt level
* By clean, noisy, and code-switched condition

### Failure statuses

Assign exactly one main status to every result:

```text
success
empty_output
refusal
malformed_output
inference_error
out_of_memory
```

### Additional flags

A successful output may still receive additional flags:

```text
hallucination_candidate
translation_detected
wrong_script
commentary_added
demonstration_copying
repetition_loop
```

### Hallucination detection

Automatic hallucination detection should be treated as candidate flagging, not absolute ground truth.

Flag outputs when they show indicators such as:

* Extremely high insertion count
* Output much longer than the reference
* Complete unrelated sentences
* Explanations or invented context
* Repeated phrases or loops
* Confident transcription for silent or non-speech audio
* Copying from few-shot demonstrations

Manually review:

* Every automatically flagged result
* Every refusal or malformed result
* A random sample of otherwise successful outputs

Optionally include a few silence or non-speech control files for pipeline testing. Do not include them in normal WER and CER averages.

---

## 7. Saved Result Schema

Every model notebook should save one record per utterance and prompt configuration:

```text
run_id
model_id
model_revision
quantization
utterance_id
prompt_level
prompt_variant
prompt_id
demonstration_ids
audio_path
speech_condition
reference_raw
reference_normalized
raw_output
cleaned_prediction
prediction_normalized
wer
cer
word_insertions
word_deletions
word_substitutions
status
hallucination_candidate
translation_detected
wrong_script
commentary_added
demonstration_copying
inference_seconds
audio_duration_seconds
generation_parameters
timestamp
```

Recommended outputs:

```text
results/prompt_dev/<model>/raw_predictions.jsonl
results/prompt_dev/<model>/utterance_metrics.csv
results/prompt_dev/<model>/prompt_summary.csv
results/prompt_dev/<model>/errors.jsonl
results/prompt_dev/<model>/run_config.json
```

---

## 8. Selecting the Best Prompt Within Each Level

Do not choose one overall winning prompt. Choose one prompt for every level.

For each model, freeze:

```text
L0_minimal
L1_structured
L2_linguistic
L3_one_shot
L4_three_shot_optional
```

Use this selection order:

1. Remove variants with unacceptable failure, refusal, or malformed-output rates.
2. Compare macro-average WER across clean, noisy, and code-switched conditions.
3. Compare overall WER.
4. Use CER as the next criterion.
5. Prefer lower hallucination and translation rates.
6. Prefer more consistent performance across conditions.
7. Prefer the shorter prompt when performance is effectively tied.

Macro-averaging the three speech conditions prevents a larger condition from dominating prompt selection.

When two variants differ by less than approximately one absolute WER point and show no clear behavioral difference, treat them as tied and retain the simpler prompt.

Few-shot selection must additionally consider:

* Demonstration copying
* Context-window failures
* Increased latency
* GPU memory use
* Whether gains occur across multiple conditions rather than one example type

---

## 9. Final Comparison and Freezing Notebook

The fifth notebook should combine the four model-level summaries.

It should produce:

* Selected prompt for every level and model
* Exact prompt text
* Exact demonstration IDs and order
* Model-specific chat formatting
* Generation parameters
* Audio preprocessing settings
* Output-extraction rules
* Normalization policy
* Development WER and CER
* Failure and hallucination summaries
* Few-shot feasibility decision

Export one frozen configuration file:

```text
configs/frozen_prompt_registry.json
```

Once frozen, prompts must not be rewritten after viewing results from the final 500 utterances.

---

## 10. Benchmark Readiness Checklist

Move to the final benchmark only when:

* All four models load reliably.
* Single and batch inference work.
* Interrupted runs can resume correctly.
* Duplicate generations are prevented.
* Raw outputs are preserved.
* WER and CER calculations have been tested.
* Failure statuses are consistently assigned.
* Hallucination and translation flags are reviewable.
* One prompt has been selected for every retained level.
* Few-shot demonstrations are fixed.
* Model revisions and quantization are frozen.
* Generation parameters are frozen.
* Output-cleaning and normalization rules are frozen.
* Each notebook can switch from the development manifest to the final 500-utterance manifest without changing inference logic.

During benchmarking, only the manifest and output directory should change. The model-loading, prompting, inference, processing, and saving pipeline should remain unchanged.
