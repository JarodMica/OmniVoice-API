from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio

from omnivoice.models.omnivoice import OmniVoice

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class OmniVoiceTTSEngine:
    """Thin class-based adapter for OmniVoice inference."""

    def __init__(
        self,
        *,
        default_device: Optional[str] = None,
        default_output_ext: str = ".wav",
    ) -> None:
        self.default_device = default_device
        self.default_output_ext = (
            default_output_ext
            if default_output_ext.startswith(".")
            else f".{default_output_ext}"
        )
        self._model: Optional[OmniVoice] = None
        self._loaded_model_source: Optional[Union[str, Path]] = None
        self._loaded_device: Optional[str] = None
        self._loaded_dtype: Optional[torch.dtype] = None
        self._load_asr: bool = False
        self._asr_model_name: Optional[str] = None
        self._reference_audio_path: Optional[Path] = None
        self._reference_text: Optional[str] = None
        self._preprocess_prompt: bool = True
        self._voice_clone_prompt = None

    def tts_load(
        self,
        *,
        model_path: PathLike,
        reference_audio_path: Optional[PathLike] = None,
        reference_text: Optional[str] = None,
        device: Optional[str] = None,
        load_asr: bool = False,
        asr_model_name: Optional[str] = None,
        preprocess_prompt: bool = True,
    ) -> None:
        model_source = self._normalize_model_source(model_path)
        resolved_device = self._resolve_device(device)
        resolved_dtype = self._default_dtype_for_device(resolved_device)

        needs_reload = (
            self._model is None
            or self._loaded_model_source != model_source
            or self._loaded_device != resolved_device
            or self._loaded_dtype != resolved_dtype
            or self._load_asr != load_asr
            or self._asr_model_name != asr_model_name
        )
        if needs_reload:
            self._load_model(
                model_source=model_source,
                device=resolved_device,
                dtype=resolved_dtype,
                load_asr=load_asr,
                asr_model_name=asr_model_name,
            )

        self._reference_audio_path = self._validate_reference_audio_path(
            reference_audio_path
        )
        self._reference_text = self._normalize_reference_text(reference_text)
        self._preprocess_prompt = preprocess_prompt
        self._voice_clone_prompt = None
        if self._reference_audio_path is not None:
            self._voice_clone_prompt = self._build_voice_clone_prompt(
                reference_audio_path=self._reference_audio_path,
                reference_text=self._reference_text,
                preprocess_prompt=preprocess_prompt,
            )

    def tts_inference(
        self,
        *,
        text: str,
        output_path: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        model_path: Optional[PathLike] = None,
        reference_audio_path: Optional[PathLike] = None,
        reference_text: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        seed: Optional[int] = None,
        preprocess_prompt: Optional[bool] = None,
        **generate_kwargs,
    ) -> Path:
        normalized_text = self._validate_text(text)
        resolved_output_path = self._resolve_output_path(
            output_path=output_path,
            output_dir=output_dir,
        )

        if model_path is not None or self._model is None:
            self.tts_load(
                model_path=model_path or self._loaded_model_source,
                reference_audio_path=reference_audio_path
                if reference_audio_path is not None
                else self._reference_audio_path,
                reference_text=reference_text
                if reference_text is not None
                else self._reference_text,
                device=self._loaded_device or self.default_device,
                load_asr=self._load_asr,
                asr_model_name=self._asr_model_name,
                preprocess_prompt=(
                    self._preprocess_prompt
                    if preprocess_prompt is None
                    else preprocess_prompt
                ),
            )
        else:
            self._maybe_refresh_reference_prompt(
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
                preprocess_prompt=preprocess_prompt,
            )

        if self._model is None:
            raise RuntimeError("Model is not loaded. Call tts_load(...) first.")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        generation_args = dict(generate_kwargs)
        if language is not None:
            generation_args["language"] = language
        if instruct is not None:
            generation_args["instruct"] = instruct

        if self._voice_clone_prompt is not None:
            generation_args["voice_clone_prompt"] = self._voice_clone_prompt

        try:
            generated_audio = self._model.generate(
                text=normalized_text,
                **generation_args,
            )
        except Exception as exc:
            raise RuntimeError(f"OmniVoice synthesis failed: {exc}") from exc

        if not generated_audio:
            raise RuntimeError("OmniVoice synthesis returned no audio.")

        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torchaudio.save(
                str(resolved_output_path),
                generated_audio[0],
                self._model.sampling_rate,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write synthesized audio to {resolved_output_path}: {exc}"
            ) from exc

        if not resolved_output_path.exists():
            raise FileNotFoundError(
                f"Expected synthesized audio file does not exist: {resolved_output_path}"
            )
        if resolved_output_path.stat().st_size <= 0:
            raise ValueError(
                f"Synthesized audio file is empty: {resolved_output_path}"
            )

        return resolved_output_path

    def _load_model(
        self,
        *,
        model_source: Union[str, Path],
        device: str,
        dtype: torch.dtype,
        load_asr: bool,
        asr_model_name: Optional[str],
    ) -> None:
        load_kwargs = {
            "device_map": device,
            "dtype": dtype,
            "load_asr": load_asr,
        }
        if asr_model_name:
            load_kwargs["asr_model_name"] = asr_model_name

        try:
            self._model = OmniVoice.from_pretrained(model_source, **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load OmniVoice model from {model_source}: {exc}"
            ) from exc

        self._loaded_model_source = model_source
        self._loaded_device = device
        self._loaded_dtype = dtype
        self._load_asr = load_asr
        self._asr_model_name = asr_model_name

    def _build_voice_clone_prompt(
        self,
        *,
        reference_audio_path: Path,
        reference_text: Optional[str],
        preprocess_prompt: bool,
    ):
        if self._model is None:
            raise RuntimeError("Model must be loaded before building a voice prompt.")
        try:
            return self._model.create_voice_clone_prompt(
                ref_audio=str(reference_audio_path),
                ref_text=reference_text,
                preprocess_prompt=preprocess_prompt,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create voice clone prompt from {reference_audio_path}: {exc}"
            ) from exc

    def _maybe_refresh_reference_prompt(
        self,
        *,
        reference_audio_path: Optional[PathLike],
        reference_text: Optional[str],
        preprocess_prompt: Optional[bool],
    ) -> None:
        if reference_audio_path is None and reference_text is None and preprocess_prompt is None:
            return

        resolved_audio_path = (
            self._validate_reference_audio_path(reference_audio_path)
            if reference_audio_path is not None
            else self._reference_audio_path
        )
        normalized_reference_text = (
            self._normalize_reference_text(reference_text)
            if reference_text is not None
            else self._reference_text
        )
        resolved_preprocess_prompt = (
            self._preprocess_prompt
            if preprocess_prompt is None
            else preprocess_prompt
        )

        if normalized_reference_text and resolved_audio_path is None:
            raise ValueError(
                "reference_text was provided without reference_audio_path. "
                "Provide both for voice cloning, or omit both for auto voice/design."
            )

        self._reference_audio_path = resolved_audio_path
        self._reference_text = normalized_reference_text
        self._preprocess_prompt = resolved_preprocess_prompt
        self._voice_clone_prompt = None

        if resolved_audio_path is not None:
            self._voice_clone_prompt = self._build_voice_clone_prompt(
                reference_audio_path=resolved_audio_path,
                reference_text=normalized_reference_text,
                preprocess_prompt=resolved_preprocess_prompt,
            )

    def _normalize_model_source(self, model_path: PathLike) -> Union[str, Path]:
        if model_path is None:
            raise ValueError("model_path is required.")
        raw_value = str(model_path).strip()
        if not raw_value:
            raise ValueError("model_path must not be empty.")

        candidate = Path(raw_value)
        if candidate.exists():
            return candidate.resolve()

        if self._looks_like_local_path(raw_value):
            raise FileNotFoundError(
                f"Model path does not exist: {candidate.resolve()}. "
                "If you intended to use a Hugging Face repo id, pass it in the "
                "form 'owner/name'."
            )

        return raw_value

    def _validate_reference_audio_path(
        self, reference_audio_path: Optional[PathLike]
    ) -> Optional[Path]:
        if reference_audio_path is None:
            return None

        raw_value = str(reference_audio_path).strip()
        if not raw_value:
            raise ValueError("reference_audio_path must not be empty when provided.")

        candidate = Path(raw_value).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(
                f"Reference audio file does not exist: {candidate.resolve()}"
            )
        if not candidate.is_file():
            raise ValueError(f"reference_audio_path is not a file: {candidate.resolve()}")

        return candidate.resolve()

    def _normalize_reference_text(self, reference_text: Optional[str]) -> Optional[str]:
        if reference_text is None:
            return None
        normalized = reference_text.strip()
        return normalized or None

    def _resolve_output_path(
        self,
        *,
        output_path: Optional[PathLike],
        output_dir: Optional[PathLike],
    ) -> Path:
        if output_path is not None and output_dir is not None:
            raise ValueError("Provide only one of output_path or output_dir.")

        if output_path is not None:
            candidate = Path(output_path).expanduser()
            if candidate.exists() and candidate.is_dir():
                raise ValueError(
                    f"output_path points to a directory, not a file: {candidate.resolve()}"
                )
            if candidate.suffix == "":
                candidate = candidate.with_suffix(self.default_output_ext)
            return candidate.resolve()

        base_dir = (
            Path(output_dir).expanduser()
            if output_dir is not None
            else Path.cwd() / "tts_outputs"
        )
        filename = f"omnivoice_tts_{int(time.time() * 1000)}{self.default_output_ext}"
        return (base_dir / filename).resolve()

    def _validate_text(self, text: str) -> str:
        if text is None:
            raise ValueError("text is required.")
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty or whitespace only.")
        return normalized

    def _resolve_device(self, preferred_device: Optional[str]) -> str:
        requested = (preferred_device or self.default_device or "").strip().lower()
        if not requested:
            return self._best_available_device()

        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return preferred_device or self.default_device or "cuda"
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"

        if requested == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but unavailable; falling back to CPU.")
            return "cpu"

        if requested == "cpu":
            return "cpu"

        return preferred_device or self.default_device or requested

    def _default_dtype_for_device(self, device: str) -> torch.dtype:
        return torch.float16 if str(device).startswith("cuda") else torch.float32

    def _best_available_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _looks_like_local_path(self, value: str) -> bool:
        candidate = Path(value)
        return (
            candidate.is_absolute()
            or "\\" in value
            or "/" in value and value.count("/") != 1
            or value.startswith(".")
            or value.startswith("~")
            or (len(value) >= 2 and value[1] == ":")
        )
