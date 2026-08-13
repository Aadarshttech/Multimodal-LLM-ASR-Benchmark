"""
Metrics calculation for audio-LLM benchmarking.
Implements WER, CER, and behavioral metric calculations.
"""

import jiwer
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ErrorCounts:
    """Container for error counts."""
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0
    hits: int = 0


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).

    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription

    Returns:
        WER as a float
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return float('inf') if hypothesis else 0.0

    # Use jiwer for standardized WER calculation
    try:
        wer = jiwer.wer(reference, hypothesis)
        return wer
    except Exception:
        # Fallback to basic implementation if jiwer fails
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # Handle empty cases
        if not ref_words:
            return 0.0 if not hyp_words else float('inf')

        # Calculate edit distance
        distances = _levenshtein_distance(ref_words, hyp_words)
        return distances / len(ref_words)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).

    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription

    Returns:
        CER as a float
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return float('inf') if hypothesis else 0.0

    # Use jiwer for standardized CER calculation
    try:
        cer = jiwer.cer(reference, hypothesis)
        return cer
    except Exception:
        # Fallback to basic implementation
        ref_chars = list(reference.replace(" ", ""))  # Remove spaces for character-level
        hyp_chars = list(hypothesis.replace(" ", ""))

        # Handle empty cases
        if not ref_chars:
            return 0.0 if not hyp_chars else float('inf')

        # Calculate edit distance at character level
        distances = _levenshtein_distance(ref_chars, hyp_chars)
        return distances / len(ref_chars)


def _levenshtein_distance(a: List[str], b: List[str]) -> int:
    """
    Calculate Levenshtein distance between two sequences.

    Args:
        a: First sequence
        b: Second sequence

    Returns:
        Levenshtein distance
    """
    if len(a) < len(b):
        return _levenshtein_distance(b, a)

    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """
    Get detailed error counts (insertions, deletions, substitutions).

    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription

    Returns:
        ErrorCounts object with detailed breakdown
    """
    # Use jiwer to get detailed alignment
    try:
        measures = jiwer.compute_measures(reference, hypothesis)
        return ErrorCounts(
            insertions=measures['insertions'],
            deletions=measures['deletions'],
            substitutions=measures['substitutions'],
            hits=measures['hits']
        )
    except Exception:
        # Fallback implementation
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # Initialize matrix
        rows, cols = len(ref_words) + 1, len(hyp_words) + 1
        dp = [[0] * cols for _ in range(rows)]

        # Initialize first row and column
        for i in range(rows):
            dp[i][0] = i
        for j in range(cols):
            dp[0][j] = j

        # Fill the matrix
        for i in range(1, rows):
            for j in range(1, cols):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )

        # Backtrack to get operations
        i, j = len(ref_words), len(hyp_words)
        insertions = deletions = substitutions = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                deletions += 1
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                insertions += 1
                j -= 1
            elif i > 0:
                deletions += 1
                i -= 1
            elif j > 0:
                insertions += 1
                j -= 1

        hits = len(ref_words) - (deletions + substitutions)

        return ErrorCounts(
            insertions=insertions,
            deletions=deletions,
            substitutions=substitutions,
            hits=max(0, hits)
        )


def calculate_metrics(reference: str, hypothesis: str) -> dict:
    """
    Calculate all metrics for a reference-hypothesis pair.

    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription

    Returns:
        Dictionary containing WER, CER, and error counts
    """
    wer = calculate_wer(reference, hypothesis)
    cer = calculate_cer(reference, hypothesis)
    error_counts = get_error_counts(reference, hypothesis)

    return {
        'wer': wer,
        'cer': cer,
        'word_insertions': error_counts.insertions,
        'word_deletions': error_counts.deletions,
        'word_substitutions': error_counts.substitutions,
        'word_hits': error_counts.hits
    }


def normalize_text(text: str) -> str:
    """
    Apply normalization to text for comparison.
    Based on the guideline: only remove markup, labels, whitespace, repeated formatting tokens.
    Does NOT correct spelling, translate, remove English tokens, or modify genuine content.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text or not isinstance(text, str):
        return ""

    # Start with stripped text
    normalized = text.strip()

    # Remove markdown code block wrappers
    if normalized.startswith("```") and normalized.endswith("```"):
        # Extract content between triple backticks
        lines = normalized.split('\n')
        if len(lines) > 2:
            normalized = '\n'.join(lines[1:-1])
        else:
            normalized = normalized[3:-3]
    elif normalized.startswith("`") and normalized.endswith("`"):
        normalized = normalized[1:-1]

    # Remove common transcription labels (case insensitive)
    labels_to_remove = [
        r'^Transcription:\s*',
        r'^transcription:\s*',
        r'^Output:\s*',
        r'^output:\s*',
        r'^Here is the transcription:\s*',
        r'^Here\'s the transcription:\s*',
        r'^Transcript:\s*',
        r'^transcript:\s*'
    ]

    for pattern in labels_to_remove:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)

    # Remove leading/trailing whitespace again
    normalized = normalized.strip()

    # Collapse multiple whitespace but preserve single spaces
    normalized = re.sub(r'\s+', ' ', normalized)

    # Remove repeated formatting tokens (like ====== or ****)
    # But be careful not to remove genuine repeated words in speech
    # Only remove obvious formatting artifacts
    normalized = re.sub(r'^[=\-_*#]{3,}[=\-_*#]*$', '', normalized, flags=re.MULTILINE)
    normalized = re.sub(r'^[=\-_*#]{3,}[=\-_*#]*$', '', normalized, flags=re.MULTILINE)

    # Final trim
    normalized = normalized.strip()

    return normalized


def detect_hallucination_indicators(reference: str, hypothesis: str) -> dict:
    """
    Detect potential hallucination indicators in hypothesis.

    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription

    Returns:
        Dictionary of boolean flags for hallucination indicators
    """
    if not hypothesis:
        return {
            'hallucination_candidate': False,
            'translation_detected': False,
            'wrong_script': False,
            'commentary_added': False,
            'demonstration_copying': False,
            'repetition_loop': False
        }

    # Initialize flags
    flags = {
        'hallucination_candidate': False,
        'translation_detected': False,
        'wrong_script': False,
        'commentary_added': False,
        'demonstration_copying': False,
        'repetition_loop': False
    }

    # Check for excessive length (potential hallucination)
    if reference and len(hypothesis) > len(reference) * 3:
        flags['hallucination_candidate'] = True

    # Check for very high insertion rate (possible hallucination)
    if reference:
        error_counts = get_error_counts(reference, hypothesis)
        insertion_rate = error_counts.insertions / max(len(ref.split()), 1)
        if insertion_rate > 2.0:  # More than 2 insertions per reference word on average
            flags['hallucination_candidate'] = True

    # Check for translation (simple heuristic - look for English words in Nepali context)
    # This is a placeholder - real translation detection would need language ID
    nepali_chars = set('अआइईउउऐऐओऔअअकखगघङचछजझञटठडढणतथधधनपफबभमयररलळवशषसह ळ क्षज्ञ')
    devanagari_chars = set('अआइईउऊऋएऐओऔअआइऊऋएऐओअअकखगघङचछजझ missão')

    # Check if hypothesis contains mostly Latin characters when reference is Devanagari
    if reference:
        ref_devanagari_ratio = sum(1 for c in reference if c in devanagari_chars) / max(len(reference), 1)
        hyp_latin_ratio = sum(1 for c in hypothesis if c.isalpha() and c.encode('ascii', 'ignore')) / max(len(hypothesis), 1)

        if ref_devanagari_ratio > 0.5 and hyp_latin_ratio > 0.7:
            flags['translation_detected'] = True

    # Check for commentary addition (look for explanatory phrases)
    commentary_indicators = [
        'this means', 'in other words', 'the speaker said',
        ' यह meaning ', 'का मतलब है', 'अर्थात्'
    ]
    hypothesis_lower = hypothesis.lower()
    for indicator in commentary_indicators:
        if indicator in hypothesis_lower:
            flags['commentary_added'] = True
            break

    # Check for repetitive patterns (potential looping)
    if len(hypothesis) > 20:
        words = hypothesis.split()
        if len(words) > 5:
            # Check for immediate repetition of phrases
            for i in range(len(words) - 3):
                phrase = ' '.join(words[i:i+3])
                if hypothesis.count(phrase) > 2:
                    flags['repetition_loop'] = True
                    break

    return flags