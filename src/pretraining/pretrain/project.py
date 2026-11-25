from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


def load_project(demo: bool = False, project_root: str | Path | None = None) -> SimpleNamespace:
    demo = bool(demo)
    src_root = (Path(project_root).resolve() if project_root is not None
                else Path(__file__).resolve().parent.parent.parent)

    if demo:
        module_root = Path(__file__).resolve().parent / "demo_project"
    else:
        module_root = src_root

    module_root_text = str(module_root)
    sys.path[:] = [entry for entry in sys.path if entry != module_root_text]
    sys.path.insert(0, module_root_text)

    # Encoding and model have to match in the same process.  Do not let modules
    # imported previously (or from demo_project) leak into a normal run.
    sys.modules.pop("encoding", None)
    sys.modules.pop("model", None)
    encoding = importlib.import_module("encoding")
    model = importlib.import_module("model")

    return SimpleNamespace(encode=encoding.encode, legal_moves_map=encoding.legal_moves_map,
                           ACTION_SIZE=encoding.ACTION_SIZE, INPUT_CHANNELS=encoding.INPUT_CHANNELS,
                           PolicyValueNet=model.PolicyValueNet, demo=demo)
