"""Check whether an existing server environment can run the VT-Bench adapter.

This script is intentionally read-only: it does not install or download
anything. Run it before using the server environment for MMPFN experiments.
"""

from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "PIL",
    "sklearn",
    "torch",
    "torchvision",
    "optuna",
    "transformers",
    "schedulefree",
    "fvcore",
    "iopath",
    "omegaconf",
)


def main() -> int:
    missing = []
    print(f"Python: {sys.version.split()[0]}")
    for module in REQUIRED_MODULES:
        try:
            imported = importlib.import_module(module)
            print(f"[OK]      {module:<14} {getattr(imported, '__version__', '')}")
        except Exception as error:  # Report all missing dependencies at once.
            missing.append(module)
            print(f"[MISSING] {module:<14} {error}")

    try:
        import torch

        print(f"Torch CUDA build: {torch.version.cuda}")
        print(f"CUDA available:   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU:              {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    if missing:
        print("\nMissing modules: " + ", ".join(missing))
        return 1
    if not __import__("torch").cuda.is_available():
        print("\nEnvironment imports succeeded, but a CUDA GPU is required for practical experiments.")
        return 2
    print("\nEnvironment is ready for the MMPFN VT-Bench adapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
