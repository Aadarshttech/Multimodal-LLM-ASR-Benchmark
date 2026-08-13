"""
Output processing utilities for audio-LLM benchmarking.
Handles cleaning and normalization of model outputs.
"""

import re
from typing import Optional
from .metrics import normalize_text, detect_hallucination_indicators


def clean_model_output(raw_output: str) -> str:
    """
    Clean model output according to benchmarking guidelines.
    Removes only:
    - Markdown wrappers
    - Labels such as "Transcription:"
    - Leading or trailing whitespace
    - Repeated formatting tokens
    Does NOT:
    - Correct spelling
    - Translate words
    - Remove English tokens
    - Replace model's chosen words
    - Delete genuine repetitions

    Args:
        raw_output: Raw output from the model

    Returns:
        Cleaned prediction
    """
    if not isinstance(raw_output, str):
        return ""

    return normalize_text(raw_output)


def process_model_output(
    raw_output: str,
    reference_raw: str = None,
    return_flags: bool = False
) -> dict:
    """
    Process model output comprehensively.
    Cleans output, calculates metrics if reference provided,
    and detects potential issues.

    Args:
        raw_output: Raw output from the model
        reference_raw: Reference transcription for metric calculation (optional)
        return_flags: Whether to return hallucination/behavioral flags

    Returns:
        Dictionary containing cleaned_prediction and optionally
        metrics and flags
    """
    # Clean the output
    cleaned_prediction = clean_model_output(raw_output)

    result = {
        'raw_output': raw_output,
        'cleaned_prediction': cleaned_prediction
    }

    # Calculate metrics if reference is provided
    if reference_raw is not None:
        from .metrics import calculate_metrics, normalize_text

        # Normalize reference for fair comparison
        normalized_reference = normalize_text(reference_raw)
        normalized_prediction = normalize_text(cleaned_prediction)

        metrics = calculate_metrics(normalized_reference, normalized_prediction)
        result.update(metrics)

        # Add normalized fields
        result['reference_normalized'] = normalized_reference
        result['prediction_normalized'] = normalized_prediction

    # Add behavioral flags if requested
    if return_flags and reference_raw is not None:
        flags = detect_hallucination_indicators(reference_raw, cleaned_prediction)
        result.update(flags)

    return result


def save_prediction_pair(
    raw_output: str,
    reference_raw: str,
    output_dir: str,
    utterance_id: str,
    prompt_id: str
) -> dict:
    """
    Save both raw and cleaned predictions to files.

    Args:
        raw_output: Raw model output
        reference_raw: Reference transcription
        output_dir: Directory to save files
        utterance_id: ID of the utterance
        prompt_id: ID of the prompt used

    Returns:
        Dictionary with file paths and processed results
    """
    import os
    import json
    from pathlib import Path

    # Process the output
    result = process_model_output(raw_output, reference_raw, return_flags=True)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw output
    raw_file = output_path / f"{utterance_id}_{prompt_id}_raw.txt"
    with open(raw_file, 'w', encoding='utf-8') as f:
        f.write(raw_output)

    # Save cleaned prediction
    clean_file = output_path / f"{utterance_id}_{prompt_id}_cleaned.txt"
    with open(clean_file, 'w', encoding='utf-8') as f:
        f.write(result['cleaned_prediction'])

    # Save reference for reference
    ref_file = output_path / f"{utterance_id}_reference.txt"
    with open(ref_file, 'w', encoding='utf-8') as f:
        f.write(reference_raw)

    # Save metadata as JSON
    meta_file = output_path / f"{utterance_id}_{prompt_id}_metadata.json"
    metadata = {
        'utterance_id': utterance_id,
        'prompt_id': prompt_id,
        'raw_output_path': str(raw_file),
        'cleaned_prediction_path': str(clean_file),
        'reference_path': str(ref_file),
        'wer': result.get('wer', 0.0),
        'cer': result.get('cer', 0.0),
        'status': 'success' if raw_output.strip() else 'empty_output'
    }

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return {
        'raw_output_path': str(raw_file),
        'cleaned_prediction_path': str(clean_file),
        'reference_path': str(ref_file),
        'metadata_path': str(meta_file),
        **result
    }


def batch_save_predictions(
    results: list,
    base_output_dir: str
) -> list:
    """
    Save multiple prediction results in batch.

    Args:
        results: List of result dictionaries from process_model_output
        base_output_dir: Base directory for saving all results

    Returns:
        List of file path dictionaries
    """
    saved_files = []

    for result in results:
        # Extract required fields
        utterance_id = result.get('utterance_id', 'unknown')
        prompt_id = result.get('prompt_id', 'unknown')
        raw_output = result.get('raw_output', '')
        reference_raw = result.get('reference_raw', '')

        # Create utterance-specific directory
        utterance_dir = f"{base_output_dir}/{utterance_id}"

        # Save the pair
        file_paths = save_prediction_pair(
            raw_output=raw_output,
            reference_raw=reference_raw,
            output_dir=utterance_dir,
            utterance_id=utterance_id,
            prompt_id=prompt_id
        )

        # Add file paths to result
        result_with_files = result.copy()
        result_with_files.update(file_paths)
        saved_files.append(result_with_files)

    return saved_files