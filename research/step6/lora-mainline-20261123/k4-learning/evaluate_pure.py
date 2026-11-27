"""Use the existing evaluation with model endpoints preserved at execution."""
import importlib.util
from pathlib import Path
import sys
from scripts.step6_arx import evaluate

path=Path(__file__).with_name('pure_scene.py')
spec=importlib.util.spec_from_file_location('scripts.step6_arx.scene',path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
evaluate.main()
