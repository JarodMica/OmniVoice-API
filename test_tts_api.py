from __future__ import annotations

from pathlib import Path

from omnivoice.tts_api import OmniVoiceTTSEngine

MODULE_PATH = "omnivoice.tts_api"
CLASS_NAME = "OmniVoiceTTSEngine"
LOAD_METHOD = "tts_load"
INFER_METHOD = "tts_inference"
MODEL_PATH = Path("local_models/OmniVoice").resolve()
REFERENCE_AUDIO_PATH = None
REFERENCE_TEXT = None
TEST_TEXT = "This is a validation run from test_tts_api.py."


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {MODEL_PATH}. "
            "Run `uv run omnivoice-download-models --output-dir local_models/OmniVoice` first."
        )

    engine = OmniVoiceTTSEngine()
    output_dir = Path("tts_test_outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "integration_test.wav"

    engine.tts_load(
        model_path=MODEL_PATH,
        reference_audio_path=REFERENCE_AUDIO_PATH,
        reference_text=REFERENCE_TEXT,
        device="cuda",
    )

    result = engine.tts_inference(
        text=TEST_TEXT,
        output_path=output_path,
        model_path=MODEL_PATH,
        reference_audio_path=REFERENCE_AUDIO_PATH,
        reference_text=REFERENCE_TEXT,
        language="English",
        speed=1.05,
        num_step=16,
    )

    returned_path = result if isinstance(result, Path) else Path(result)
    if not returned_path.is_absolute():
        returned_path = (Path.cwd() / returned_path).resolve()

    if not returned_path.exists():
        raise FileNotFoundError(f"Expected output file does not exist: {returned_path}")
    if returned_path.stat().st_size <= 0:
        raise ValueError(f"Output file is empty: {returned_path}")

    print(f"SUCCESS: generated audio at {returned_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
