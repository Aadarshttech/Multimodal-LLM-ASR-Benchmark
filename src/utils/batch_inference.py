"""
Batch inference pipeline for audio-LLM benchmarking.
Provides resumable batch processing with error handling and checkpointing.
"""

import json
import os
import jsonlines
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import traceback
from dataclasses import dataclass, asdict
from enum import Enum


class FailureStatus(str, Enum):
    SUCCESS = "success"
    EMPTY_OUTPUT = "empty_output"
    REFUSAL = "refusal"
    MALFORMED_OUTPUT = "malformed_output"
    INFERENCE_ERROR = "inference_error"
    OUT_OF_MEMORY = "out_of_memory"


@dataclass
class ProcessingResult:
    """Result structure for each utterance-prompt combination."""
    run_id: str
    model_id: str
    model_revision: str
    quantization: str
    utterance_id: str
    prompt_level: int
    prompt_variant: str
    prompt_id: str
    demonstration_ids: List[str]
    audio_path: str
    speech_condition: str
    reference_raw: str
    reference_normalized: str
    raw_output: str
    cleaned_prediction: str
    prediction_normalized: str
    wer: float
    cer: float
    word_insertions: int
    word_deletions: int
    word_substitutions: int
    status: FailureStatus
    hallucination_candidate: bool
    translation_detected: bool
    wrong_script: bool
    commentary_added: bool
    demonstration_copying: bool
    inference_seconds: float
    audio_duration_seconds: float
    generation_parameters: Dict[str, Any]
    timestamp: str


class BatchInferencePipeline:
    """Resumable batch inference pipeline for audio-LLM evaluation."""

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        quantization: str,
        batch_size: int = 8,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        device: str = "cuda",
        results_dir: str = "./results"
    ):
        self.model_id = model_id
        self.model_revision = model_revision
        self.quantization = quantization
        self.batch_size = batch_size
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Model and processor will be initialized in subclasses
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the model and processor. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_model()")

    def preprocess_audio(self, audio_path: str) -> Any:
        """Preprocess audio file. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement preprocess_audio()")

    def generate_transcription(self, processed_audio: Any, prompt: str) -> str:
        """Generate transcription from processed audio. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_transcription()")

    def process_batch(self, batch_items: List[Dict]) -> List[ProcessingResult]:
        """Process a batch of audio-prompt pairs."""
        results = []

        for item in batch_items:
            try:
                start_time = time.time()

                # Preprocess audio
                processed_audio = self.preprocess_audio(item['audio_path'])

                # Generate transcription
                raw_output = self.generate_transcription(processed_audio, item['prompt'])

                inference_time = time.time() - start_time

                # Process output (to be implemented by subclass or utility)
                cleaned_prediction = self._clean_output(raw_output)

                # Create result object (metrics will be filled later)
                result = ProcessingResult(
                    run_id=item.get('run_id', ''),
                    model_id=self.model_id,
                    model_revision=self.model_revision,
                    quantization=self.quantization,
                    utterance_id=item['utterance_id'],
                    prompt_level=item['prompt_level'],
                    prompt_variant=item['prompt_variant'],
                    prompt_id=item['prompt_id'],
                    demonstration_ids=item.get('demonstration_ids', []),
                    audio_path=item['audio_path'],
                    speech_condition=item.get('speech_condition', 'unknown'),
                    reference_raw=item.get('reference_raw', ''),
                    reference_normalized=item.get('reference_normalized', ''),
                    raw_output=raw_output,
                    cleaned_prediction=cleaned_prediction,
                    prediction_normalized=cleaned_prediction,  # Will be updated by normalization
                    wer=0.0,  # Placeholder
                    cer=0.0,  # Placeholder
                    word_insertions=0,
                    word_deletions=0,
                    word_substitutions=0,
                    status=FailureStatus.SUCCESS,
                    hallucination_candidate=False,
                    translation_detected=False,
                    wrong_script=False,
                    commentary_added=False,
                    demonstration_copying=False,
                    inference_seconds=inference_time,
                    audio_duration_seconds=item.get('audio_duration_seconds', 0.0),
                    generation_parameters={
                        'max_output_tokens': self.max_output_tokens,
                        'temperature': self.temperature
                    },
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )

                results.append(result)

            except torch.cuda.OutOfMemoryError as e:
                # Handle OOM error
                result = ProcessingResult(
                    run_id=item.get('run_id', ''),
                    model_id=self.model_id,
                    model_revision=self.model_revision,
                    quantization=self.quantization,
                    utterance_id=item['utterance_id'],
                    prompt_level=item['prompt_level'],
                    prompt_variant=item['prompt_variant'],
                    prompt_id=item['prompt_id'],
                    demonstration_ids=item.get('demonstration_ids', []),
                    audio_path=item['audio_path'],
                    speech_condition=item.get('speech_condition', 'unknown'),
                    reference_raw=item.get('reference_raw', ''),
                    reference_normalized=item.get('reference_normalized', ''),
                    raw_output="",
                    cleaned_prediction="",
                    prediction_normalized="",
                    wer=0.0,
                    cer=0.0,
                    word_insertions=0,
                    word_deletions=0,
                    word_substitutions=0,
                    status=FailureStatus.OUT_OF_MEMORY,
                    hallucination_candidate=False,
                    translation_detected=False,
                    wrong_script=False,
                    commentary_added=False,
                    demonstration_copying=False,
                    inference_seconds=0.0,
                    audio_duration_seconds=item.get('audio_duration_seconds', 0.0),
                    generation_parameters={
                        'max_output_tokens': self.max_output_tokens,
                        'temperature': self.temperature
                    },
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                results.append(result)

                # Clear GPU cache to recover
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                # Handle other inference errors
                result = ProcessingResult(
                    run_id=item.get('run_id', ''),
                    model_id=self.model_id,
                    model_revision=self.model_revision,
                    quantization=self.quantization,
                    utterance_id=item['utterance_id'],
                    prompt_level=item['prompt_level'],
                    prompt_variant=item['prompt_variant'],
                    prompt_id=item['prompt_id'],
                    demonstration_ids=item.get('demonstration_ids', []),
                    audio_path=item['audio_path'],
                    speech_condition=item.get('speech_condition', 'unknown'),
                    reference_raw=item.get('reference_raw', ''),
                    reference_normalized=item.get('reference_normalized', ''),
                    raw_output="",
                    cleaned_prediction="",
                    prediction_normalized="",
                    wer=0.0,
                    cer=0.0,
                    word_insertions=0,
                    word_deletions=0,
                    word_substitutions=0,
                    status=FailureStatus.INFERENCE_ERROR,
                    hallucination_candidate=False,
                    translation_detected=False,
                    wrong_script=False,
                    commentary_added=False,
                    demonstration_copying=False,
                    inference_seconds=0.0,
                    audio_duration_seconds=item.get('audio_duration_seconds', 0.0),
                    generation_parameters={
                        'max_output_tokens': self.max_output_tokens,
                        'temperature': self.temperature
                    },
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                results.append(result)

        return results

    def _clean_output(self, text: str) -> str:
        """Basic output cleaning - remove wrappers and extra whitespace."""
        if not text or not isinstance(text, str):
            return ""

        # Remove common markdown wrappers
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            # Extract content between triple backticks
            lines = text.split('\n')
            if len(lines) > 2:
                text = '\n'.join(lines[1:-1])
            else:
                text = text[3:-3]
        elif text.startswith("`") and text.endswith("`"):
            text = text[1:-1]

        # Remove common prefixes
        prefixes_to_remove = [
            "Transcription:", "transcription:",
            "Output:", "output:",
            "Here is the transcription:", "Here's the transcription:"
        ]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Remove trailing whitespace but keep internal spacing
        text = text.strip()

        return text

    def save_results(self, results: List[ProcessingResult], filename: str):
        """Save results to JSONL and CSV formats."""
        # Save as JSONL
        jsonl_path = self.results_dir / f"{filename}.jsonl"
        with jsonlines.open(jsonl_path, mode='w') as writer:
            for result in results:
                writer.write(asdict(result))

        # Save as CSV (would require pandas, simplified version here)
        # For now, just save JSONL which contains all needed information
        print(f"Saved {len(results)} results to {jsonl_path}")

    def load_completed_ids(self, filename: str) -> set:
        """Load set of already completed (utterance_id, prompt_id) pairs."""
        completed = set()
        jsonl_path = self.results_dir / f"{filename}.jsonl"

        if jsonl_path.exists():
            try:
                with jsonlines.open(jsonl_path, mode='r') as reader:
                    for obj in reader:
                        key = (obj['utterance_id'], obj['prompt_id'])
                        completed.add(key)
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")

        return completed

    def run_inference(
        self,
        manifest: List[Dict],
        prompt_configs: List[Dict],
        output_filename: str
    ) -> List[ProcessingResult]:
        """
        Run inference on manifest with given prompt configurations.

        Args:
            manifest: List of utterance dictionaries with audio_path, utterance_id, etc.
            prompt_configs: List of prompt configuration dictionaries
            output_filename: Base name for output files

        Returns:
            List of ProcessingResult objects
        """
        # Load already completed items to enable resumption
        completed_keys = self.load_completed_ids(output_filename)

        # Prepare all items to process
        items_to_process = []
        for utterance in manifest:
            for prompt_config in prompt_configs:
                key = (utterance['utterance_id'], prompt_config['prompt_id'])
                if key not in completed_keys:
                    item = {
                        'utterance_id': utterance['utterance_id'],
                        'audio_path': utterance['audio_path'],
                        'reference_raw': utterance.get('reference_raw', ''),
                        'reference_normalized': utterance.get('reference_normalized', ''),
                        'speech_condition': utterance.get('speech_condition', 'unknown'),
                        'audio_duration_seconds': utterance.get('audio_duration_seconds', 0.0),
                        'prompt_level': prompt_config['level'],
                        'prompt_variant': prompt_config['variant'],
                        'prompt_id': prompt_config['prompt_id'],
                        'prompt': prompt_config['text'],
                        'demonstration_ids': prompt_config.get('demonstration_ids', []),
                        'run_id': utterance.get('run_id', '')
                    }
                    items_to_process.append(item)

        print(f"Total items to process: {len(items_to_process)}")
        print(f"Already completed: {len(completed_keys)}")

        # Process in batches
        all_results = []

        for i in range(0, len(items_to_process), self.batch_size):
            batch = items_to_process[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(items_to_process) + self.batch_size - 1) // self.batch_size

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")

            try:
                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)

                # Save intermediate results after each batch
                if batch_results:
                    self.save_results(batch_results, f"{output_filename}_batch_{batch_num}")

            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                traceback.print_exc()
                # Continue with next batch

        # Save all results combined
        if all_results:
            self.save_results(all_results, output_filename)

        return all_results