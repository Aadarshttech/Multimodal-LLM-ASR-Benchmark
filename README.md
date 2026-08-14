# Code-Switched ASR Prompt Development Benchmark

This repository contains the full methodology, scripts, and results for a systematic evaluation of multimodal Large Language Models (LLMs) on verbatim **code-switched (Nepali-English) Automatic Speech Recognition (ASR)**.

The primary objective of this project was to discover an optimal prompt configuration that forces LLMs to transcribe audio verbatim, while enforcing strict orthographic rules (Devanagari for Nepali, Latin for English), without allowing the model to grammatically correct or summarize the speech.

## 📊 The Challenge
Standard ASR struggles with code-switching. LLMs possess the linguistic reasoning to handle it, but they are inherently biased toward summarization and grammatical correction. This project evaluates if models can suppress these biases and act purely as transcription engines under strict formatting constraints.

## 🔬 Methodology

To prevent prompt overfitting, the benchmark was conducted in two discrete stages:
- **Stage 1 (Pilot Test):** 30 audio files used to quickly rapidly prototype and weed out underperforming models.
- **Stage 2 (Validation):** 60 unseen audio files used to validate the winning models and ensure generalization.

### Models Evaluated
1. **Voxtral-Mini-3B**
2. **Gemma4-12B**
3. **Qwen2.5-Omni-3B**
4. **Phi-4-Multimodal**

### Prompt Complexity Levels
We engineered 5 progressive prompt levels to test zero-shot vs few-shot capabilities:
- **L0:** Minimal zero-shot transcription.
- **L1:** Standard zero-shot transcription.
- **L2:** Zero-shot with explicit linguistic & formatting rules.
- **L3:** 1-Shot (Rules + 1 exemplar).
- **L4:** 3-Shot (Rules + 3 exemplars).

## 🏆 Key Findings & Final Verdict

After comprehensive two-stage testing, **Voxtral-Mini-3B** emerged as the undisputed champion. 

While rigid models like Gemma4-12B performed exceptionally well in zero-shot environments but degraded massively when shown examples (Exemplar Confusion), **Voxtral-Mini-3B** acted as a highly flexible learner. Utilizing the **L4 (3-Shot)** prompt, Voxtral achieved:
- A benchmark-leading **0.55 Word Error Rate (WER)**.
- **0.0% Hallucination Rate** (it completely refused to fabricate text).
- Exceptional scaling across both Stage 1 and Stage 2 unseen data.

### 🔥 Final Benchmark: Voxtral vs. Whisper
To establish a "real-world" baseline, the optimal Voxtral-Mini-3B model was benchmarked against OpenAI's state-of-the-art dedicated ASR model (**Whisper Large V3**) on a 50-utterance test set (`oslr54_50`). 

**The results were highly significant:**
- **Whisper Large V3:** 0.952 WER, 2.0% Hallucination Rate
- **Voxtral-Mini-3B:** 0.789 WER, 0.0% Hallucination Rate

Despite being a smaller, general-purpose LLM (3B parameters), Voxtral out-performed the dedicated Whisper ASR model on this code-switched task, cementing its position as the optimal pipeline.

**The final recommendation for production use on code-switched Nepali-English audio is the Voxtral-Mini-3B model.**

## 📁 Repository Structure
- `docs/`: Contains the final academic research report (`report.pdf`) and the compiled LaTeX source (`report.tex`).
- `scripts/`: Python automation scripts used to patch and build Jupyter notebooks for batch inference.
- `data/`: Raw manifests and validation sets used for Stage 1 and Stage 2 testing.
- `results/`: The raw outputs, generated transcriptions, and metric summaries (WER, CER, Hallucination rates) for all evaluated models.
- `notebooks/`: The Jupyter notebook pipelines used for evaluation.
- `src/`: Core logic and helper utilities.
- `configs/`: Environment variables and `requirements.txt`.
