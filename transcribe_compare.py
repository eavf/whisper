"""
Transcribes an audio file using Whisper and compares the result
with a reference text from a .docx file.
"""

import re
import argparse
import whisper
import docx
from jiwer import wer, cer
from pathlib import Path


def read_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def clean_reference(text: str) -> str:
    """Remove timestamps [0:00], speaker labels (ALL CAPS lines), and extra whitespace."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\[\d+:\d+(?::\d+)?\]", "", line)  # remove [0:00] or [0:00:00]
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[A-Z0-9\s\-\.]+$", line) and len(line) < 60:
            continue  # skip all-caps speaker/title lines
        lines.append(line)
    return " ".join(lines)


def transcribe(audio_path: str, model_name: str = "large", language: str = None) -> str:
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"Transcribing '{audio_path}'...")
    result = model.transcribe(
        audio_path,
        language=language,
        condition_on_previous_text=False,  # prevents hallucination loops
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        verbose=True,
    )
    print(f"\nDetected language: {result['language']}")
    return result["text"].strip()


def save_transcription(hypothesis: str, reference_path: str):
    output_path = Path(reference_path).parent / (Path(reference_path).stem + "_transcription.txt")
    output_path.write_text(hypothesis, encoding="utf-8")
    print(f"Transcription saved to: {output_path}")
    return output_path


def compare(reference: str, hypothesis: str):
    word_error_rate = wer(reference, hypothesis)
    char_error_rate = cer(reference, hypothesis)

    print("\n" + "=" * 60)
    print("REFERENCIA (vyčistená):")
    print("-" * 60)
    print(reference[:1000] + ("..." if len(reference) > 1000 else ""))
    print("\nTRANSKRIPCIA (Whisper):")
    print("-" * 60)
    print(hypothesis[:1000] + ("..." if len(hypothesis) > 1000 else ""))
    print("\nVÝSLEDKY:")
    print("-" * 60)
    print(f"  WER (Word Error Rate):  {word_error_rate:.2%}")
    print(f"  CER (Char Error Rate):  {char_error_rate:.2%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio and compare with reference docx.")
    parser.add_argument("--audio",     default="data/cons.m4a",  help="Path to audio file")
    parser.add_argument("--reference", default="data/cons.docx", help="Path to reference .docx")
    parser.add_argument("--model",     default="large",          help="Whisper model size")
    parser.add_argument("--language",  default=None,             help="Language of the audio (default: auto-detect)")
    args = parser.parse_args()

    reference_raw = read_docx(args.reference)
    reference_clean = clean_reference(reference_raw)
    hypothesis = transcribe(args.audio, model_name=args.model, language=args.language)
    save_transcription(hypothesis, args.reference)
    compare(reference_clean, hypothesis)


if __name__ == "__main__":
    main()