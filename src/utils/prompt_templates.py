"""
Prompt templates for audio-LLM benchmarking.
Defines prompt levels 0-4 with variants as specified in the guideline.
"""

from typing import List, Dict
from enum import Enum
from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    """Represents a prompt template."""
    level: int
    variant: str  # 'a', 'b', etc. for variants within a level
    text: str
    description: str
    demonstration_ids: List[str] = field(default_factory=list)


class PromptTemplates:
    """Collection of prompt templates for Nepali ASR task."""

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize all prompt templates according to guidelines."""

        # Level 0: Minimal zero-shot
        self.templates["L0_a"] = PromptTemplate(
            level=0,
            variant="a",
            text="Transcribe the audio.",
            description="Minimal zero-shot prompt - basic transcription instruction"
        )
        self.templates["L0_b"] = PromptTemplate(
            level=0,
            variant="b",
            text="Transcribe the audio in Nepali.",
            description="Minimal zero-shot with language specification"
        )

        # Level 1: Structured zero-shot
        self.templates["L1_a"] = PromptTemplate(
            level=1,
            variant="a",
            text="""Transcribe the audio.
The task is speech transcription.
The primary language is Nepali.
Nepali should be written in Devanagari script.
The transcription should be verbatim.
Do not translate.
Only the transcription should be returned.""",
            description="Structured zero-shot with clear transcription instructions"
        )
        self.templates["L1_b"] = PromptTemplate(
            level=1,
            variant="b",
            text="""Convert this speech to text.
You are transcribing Nepali speech.
Write what you hear in Devanagari script.
Provide a verbatim transcription without translation.
Output only the transcription text.""",
            description="Alternative structuring of zero-shot instructions"
        )

        # Level 2: Linguistically aware zero-shot
        # Based on measurable Nepali ASR behaviors from dataset
        self.templates["L2_a"] = PromptTemplate(
            level=2,
            variant="a",
            text="""Transcribe the audio following these rules:
1. Write spoken Nepali in Devanagari script
2. Preserve spoken English words in Latin script
3. Maintain Nepali-English code-switching in spoken order
4. Do not translate between languages
5. Keep audible fillers, repetitions, corrections, incomplete words
6. Do not correct speaker's grammar
7. Do not infer inaudible words
8. Follow dataset's number and punctuation conventions
9. Do not add timestamps, speaker labels, explanations
10. Return only the transcription""",
            description="Concise linguistically-aware prompt based on measurable ASR behaviors"
        )
        self.templates["L2_b"] = PromptTemplate(
            level=2,
            variant="b",
            text="""You are transcribing Nepali speech audio. Follow these specific guidelines:
LANGUAGE: Write Nepali in Devanagari, keep English words in Latin script
CODE-SWITCHING: Preserve Nepali-English mixing exactly as spoken
FIDELITY: Provide verbatim transcription including fillers, repetitions, self-corrections, and incomplete words
GRAMMAR: Do not correct grammatical errors or ungrammatical structures
INFERENCE: Only transcribe clearly audible content; do not guess missing words
NUMBERS/PUNCTUATION: Follow the dataset's specific conventions for numbers and punctuation
OUTPUT: Exclude timestamps, speaker identifiers, explanations, or confidence scores - output transcription only""",
            description="Explicit rule-based prompt for linguistic awareness in Nepali ASR"
        )

        # Level 3: One-shot (template - demonstrations will be added)
        self.templates["L3_template"] = PromptTemplate(
            level=3,
            variant="template",
            text="",  # Will be filled with demonstration + test instruction
            description="One-shot template - demonstration to be added"
        )

        # Level 4: Three-shot (template - demonstrations will be added)
        self.templates["L4_template"] = PromptTemplate(
            level=4,
            variant="template",
            text="",  # Will be filled with three demonstrations + test instruction
            description="Three-shot template - three demonstrations to be added"
        )

    def get_templates_by_level(self, level: int) -> List[PromptTemplate]:
        """Get all templates for a given level."""
        return [t for t in self.templates.values() if t.level == level]

    def get_template(self, template_id: str) -> PromptTemplate:
        """Get a specific template by ID."""
        return self.templates[template_id]

    def create_one_shot_prompt(self, demonstration_text: str, test_instruction: str = None) -> str:
        """
        Create a one-shot prompt by combining demonstration with test instruction.

        Args:
            demonstration_text: The demonstration audio-transcript pair description
            test_instruction: Instruction for the test audio (defaults to L2 principles)

        Returns:
            Complete one-shot prompt string
        """
        if test_instruction is None:
            test_instruction = self.templates["L2_a"].text  # Use L2 principles as default

        return f"""{demonstration_text}

Now transcribe this audio following the same principles:
{test_instruction}"""

    def create_three_shot_prompt(self, demonstrations: List[str], test_instruction: str = None) -> str:
        """
        Create a three-shot prompt by combining three demonstrations with test instruction.

        Args:
            demonstrations: List of three demonstration descriptions
            test_instruction: Instruction for the test audio (defaults to L2 principles)

        Returns:
            Complete three-shot prompt string
        """
        if test_instruction is None:
            test_instruction = self.templates["L2_a"].text  # Use L2 principles as default

        demo_section = "\n\n".join([f"Example {i+1}:\n{demo}" for i, demo in enumerate(demonstrations)])

        return f"""{demo_section}

Now transcribe this audio following the same principles:
{test_instruction}"""


# Predefined demonstration templates for one-shot and three-shot
DEMONSTRATION_TEMPLATES = {
    "standard_nepali": """Example 1 (Standard Nepali):
Audio: [Nepali speech: "मेरो नाम राम हो"]
Transcription: मेरो नाम राम हो""",

    "code_switching": """Example 2 (Code-switching):
Audio: [Nepali-English speech: "यो laptop रामको हो"]
Transcription: यो laptop RAMRO ho""",

    "numbers_fillers": """Example 3 (Numbers and fillers):
Audio: [Nepali with numbers and fillers: "एउटा, अहिले fuels भरيرا गएको छु, uh..."]
Transcription: एउटा, अहिले petrol भरिरहेको छु, उछ""",

    "repetitions": """Example 4 (Repetitions and corrections):
Audio: [Nepali with repetition: "म मato घर गए最後に bathing गर्ँ"]
Transcription: म maato घर गएस bath गर्ँ"""
}