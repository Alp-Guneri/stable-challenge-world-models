"""The model format and the evaluator's model loading and inference code.

The official evaluation loads and runs every submission with exactly these functions, so a model that works with
them (see test_submission.py) runs the same way there. The contract (see CHALLENGE.md):

    inputs   frames   float32 [64, HIST, 3, 64, 64]   the last HIST observations, SKIP env steps apart, in [0, 1]
             actions  float32 [64, HORIZON, 2*SKIP]   the next HORIZON*SKIP env actions, grouped SKIP per step
    output   z        float32 [64, HORIZON + 1, D]    z[:, 0] embeds the current observation (it must not depend
                                                      on `actions`); z[:, k] is the predicted embedding k steps
                                                      (k*SKIP env steps) later, using only actions[:, :k]
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort

BATCH, HIST, SKIP, HORIZON, SIZE, ACTION_DIM = 64, 3, 5, 5, 64, 2
MAX_DIM = 1024
FRAMES_SHAPE = (BATCH, HIST, 3, SIZE, SIZE)
ACTIONS_SHAPE = (BATCH, HORIZON, ACTION_DIM * SKIP)
CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"


class SubmissionError(Exception):
    """A problem with the submitted model, shown to the participant."""


def load_model(path):
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = [CUDA, CPU] if CUDA in ort.get_available_providers() else [CPU]
    try:
        sess = ort.InferenceSession(path, opts, providers=providers)
    except Exception as e:
        raise SubmissionError(f"Could not load the ONNX model: {e}") from None
    if sess.get_providers()[0] != CUDA:
        # one single-threaded batch per core in parallel (see run_batches)
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(path, opts, providers=[CPU])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 2 or len(outs) != 1:
        raise SubmissionError("ONNX model must have exactly two inputs (frames, actions) and one output (z)")
    names = sorted(i.name for i in ins)
    if names != ["actions", "frames"]:
        raise SubmissionError(f"ONNX inputs must be named 'frames' and 'actions'; got {names}")
    for i in ins:
        want = FRAMES_SHAPE if i.name == "frames" else ACTIONS_SHAPE
        if i.type != "tensor(float)":
            raise SubmissionError(f"Input '{i.name}' must be float32, not {i.type}")
        if len(i.shape) != len(want) or any(isinstance(g, int) and g != w for g, w in zip(i.shape, want)):
            raise SubmissionError(f"Input '{i.name}' must accept shape {want}; it declares {i.shape}")
    return sess


def to_input(frames):
    """uint8 [n, HIST, 64, 64, 3] -> float32 [n, HIST, 3, 64, 64] in [0, 1]."""
    return np.ascontiguousarray(np.asarray(frames, np.float32).transpose(0, 1, 4, 2, 3) / 255.0)


def run_batches(sess, frames, actions):
    """frames uint8 [n, HIST, 64, 64, 3], actions float32 [n, HORIZON, 2*SKIP] -> z float32 [n, HORIZON+1, D].

    Always runs full batches of BATCH; the last one is zero-padded and trimmed. On CPU, one batch per core.
    """
    n = len(frames)

    def run(i):
        f = np.zeros(FRAMES_SHAPE, np.float32)
        a = np.zeros(ACTIONS_SHAPE, np.float32)
        m = min(BATCH, n - i)
        f[:m] = to_input(frames[i : i + m])
        a[:m] = actions[i : i + m]
        try:
            z = sess.run(None, {"frames": f, "actions": a})[0]
        except Exception as e:
            raise SubmissionError(f"Model failed on frames {FRAMES_SHAPE} and actions {ACTIONS_SHAPE}: {e}") from None
        if not isinstance(z, np.ndarray) or z.ndim != 3 or z.shape[:2] != (BATCH, HORIZON + 1):
            raise SubmissionError(f"Output shape {getattr(z, 'shape', None)}, expected ({BATCH}, {HORIZON + 1}, D)")
        return z[:m]

    starts = range(0, n, BATCH)
    if sess.get_providers()[0] == CPU:
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        with ThreadPoolExecutor(cores) as pool:
            out = list(pool.map(run, starts))
    else:
        out = [run(i) for i in starts]
    dims = {z.shape[2] for z in out}
    if len(dims) != 1:
        raise SubmissionError("Model returned inconsistent embedding widths")
    dim = dims.pop()
    if not 1 <= dim <= MAX_DIM:
        raise SubmissionError(f"Embedding width {dim}, expected between 1 and {MAX_DIM}")
    z = np.concatenate(out).astype(np.float32)
    if not np.isfinite(z).all():
        raise SubmissionError("Embeddings contain NaN or infinite values")
    return z

