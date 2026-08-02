from dataclasses import dataclass


@dataclass
class Segment:
    start_time: float       # seconds — set by Transcriber
    end_time: float         # seconds — set by Transcriber
    text: str               # source-language transcript
    language: str           # ISO code detected by Whisper (file-level)
    confidence: float       # faster-whisper avg_logprob for this segment
    translated_text: str = ""   # populated by Translator stage
    gender: str = ""            # populated by AudioDubber for Piper voice selection
