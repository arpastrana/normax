import importlib.util
from pathlib import Path


def load_tesseract_api(name):
    p = Path(__file__).parent.parent / "tesseracts" / name / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(f"ta_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
