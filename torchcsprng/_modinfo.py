from pathlib import Path
from typing import List, Final

PARENT_DIR: Final[Path] = Path(__file__).parent
SRC_DIR: Final[Path] = PARENT_DIR / "csrc"
BUILD_DIR: Final[Path] = PARENT_DIR / "_build"

SOURCES: List[Path] = []
SOURCES += list(SRC_DIR.glob("*.cpp"))
SOURCES += list(SRC_DIR.glob("cpu/*.cpp"))

CUDA_SOURCES: List[Path] = []
CUDA_SOURCES += list(SRC_DIR.glob("*.cu"))
CUDA_SOURCES += list(SRC_DIR.glob("cuda/*.cu"))

__all__: Final[List[str]] = [
    "PARENT_DIR",
    "SRC_DIR",
    "BUILD_DIR",
]
