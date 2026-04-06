#!/usr/bin/env python3
"""Download the OmniVoice inference assets into a local model directory.

This script mirrors the repo's inference expectations:

- the main model is loaded from a local directory or Hugging Face repo id
- the audio tokenizer is expected at ``<model_dir>/audio_tokenizer``
- optional ASR assets can be downloaded separately for offline auto-transcription
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_REPO = "k2-fsa/OmniVoice"
DEFAULT_AUDIO_TOKENIZER_REPO = "eustlb/higgs-audio-v2-tokenizer"
DEFAULT_ASR_REPO = "openai/whisper-large-v3-turbo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download OmniVoice inference models into a local directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local OmniVoice model directory to create or update.",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face repo id for the OmniVoice model.",
    )
    parser.add_argument(
        "--audio-tokenizer-repo",
        default=DEFAULT_AUDIO_TOKENIZER_REPO,
        help="Fallback audio tokenizer repo if the model snapshot lacks audio_tokenizer/.",
    )
    parser.add_argument(
        "--include-asr",
        action="store_true",
        help="Also download Whisper ASR for offline auto-transcription.",
    )
    parser.add_argument(
        "--asr-repo",
        default=DEFAULT_ASR_REPO,
        help="Hugging Face repo id for the optional ASR model.",
    )
    parser.add_argument(
        "--asr-output-dir",
        type=Path,
        default=None,
        help="Where to place the optional ASR snapshot. Defaults under output-dir/asr/.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token passed to snapshot_download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without downloading anything.",
    )
    return parser


def repo_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def has_snapshot_contents(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def download_snapshot(
    repo_id: str,
    local_dir: Path,
    hf_token: str | None,
    dry_run: bool,
) -> Path:
    resolved_dir = local_dir.resolve()
    logging.info("Repo: %s", repo_id)
    logging.info("URL: %s", repo_url(repo_id))
    logging.info("Target: %s", resolved_dir)

    if dry_run:
        return resolved_dir

    resolved_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(resolved_dir),
        token=hf_token,
    )
    return resolved_dir


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    output_dir = args.output_dir.resolve()
    audio_tokenizer_dir = output_dir / "audio_tokenizer"

    logging.info("OmniVoice inference model repo: %s", args.model_repo)
    logging.info("OmniVoice model URL: %s", repo_url(args.model_repo))
    logging.info("Audio tokenizer fallback repo: %s", args.audio_tokenizer_repo)
    logging.info("Audio tokenizer fallback URL: %s", repo_url(args.audio_tokenizer_repo))

    download_snapshot(
        repo_id=args.model_repo,
        local_dir=output_dir,
        hf_token=args.hf_token,
        dry_run=args.dry_run,
    )

    if has_snapshot_contents(audio_tokenizer_dir):
        logging.info("Model snapshot already contains %s", audio_tokenizer_dir)
    elif args.dry_run:
        logging.info(
            "Dry run: will ensure %s exists, using %s as fallback if the model "
            "snapshot does not provide audio_tokenizer/.",
            audio_tokenizer_dir,
            args.audio_tokenizer_repo,
        )
    else:
        logging.info(
            "Model snapshot does not contain audio_tokenizer/. Downloading fallback."
        )
        download_snapshot(
            repo_id=args.audio_tokenizer_repo,
            local_dir=audio_tokenizer_dir,
            hf_token=args.hf_token,
            dry_run=args.dry_run,
        )

    asr_output_dir = None
    if args.include_asr:
        asr_output_dir = (
            args.asr_output_dir.resolve()
            if args.asr_output_dir is not None
            else output_dir / "asr" / sanitize_repo_id(args.asr_repo)
        )
        logging.info("Optional ASR repo: %s", args.asr_repo)
        logging.info("Optional ASR URL: %s", repo_url(args.asr_repo))
        download_snapshot(
            repo_id=args.asr_repo,
            local_dir=asr_output_dir,
            hf_token=args.hf_token,
            dry_run=args.dry_run,
        )

    print()
    print(f"Local OmniVoice model path: {output_dir}")
    print(f"Pass this to inference: --model {output_dir}")
    print(f"Expected local audio tokenizer path: {audio_tokenizer_dir}")
    if args.include_asr:
        print(f"Optional local ASR path: {asr_output_dir}")
        print(
            "Use the ASR path with OmniVoice.from_pretrained(..., "
            f"load_asr=True, asr_model_name=r'{asr_output_dir}')"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
