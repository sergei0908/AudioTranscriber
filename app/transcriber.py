import os
import sys
from functools import lru_cache
from pathlib import Path

MODEL_SIZE = "small"


def _add_cuda_dlls() -> None:
    """Windows: добавляет DLL из pip-пакетов nvidia-* в путь поиска библиотек."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for name in ("cublas", "cudnn"):
            bin_dir = Path(base) / name / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


@lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel

    _add_cuda_dlls()
    return WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

def transcribe(file_path: str, language: str | None = None) -> str:
    model = _get_model()
    segments, _info = model.transcribe(file_path, language=language, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()