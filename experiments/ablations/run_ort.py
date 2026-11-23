"""Run a repo script (test_submission.py, proxy_eval.py, ...) unchanged, but with onnxruntime forced to CPU when no
GPU is visible: onnxruntime-gpu lists CUDA as available on GPU-less nodes and then crashes creating the session.

    python experiments/ablations/run_ort.py test_submission.py model.onnx
"""

import runpy
import sys
from pathlib import Path

import onnxruntime as ort
import torch

if not torch.cuda.is_available():
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]

script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
