"""The official evaluator of the world-model challenge.

    python evaluate.py model.onnx path/to/eval_data [--out results.json] [--verbose]

``eval_data`` is the unzipped private evaluation data. It holds the hidden tasks (``tasks.py``) and their data, so
this file only loads and runs the model; what the tasks compute stays private. test_submission.py uses the same
loading and inference code (load_model, run_batches), so a model that passes there runs the same way here.

Always prints exactly one JSON object on stdout, and nothing else:
    {"score": <float>}     the evaluation succeeded
    {"error": "<message>"} the model breaks the format (see CHALLENGE.md) or the evaluation failed; exit code 1
Progress and, on an internal error, the traceback go to stderr. --out and --verbose give the per-task results
(organizers only; the task names are hidden from the leaderboard).

The model contract (see CHALLENGE.md):
    inputs   frames   float32 [64, HIST, 3, 64, 64]   the last HIST observations, SKIP env steps apart, in [0, 1]
             actions  float32 [64, HORIZON, 2*SKIP]   the next HORIZON*SKIP env actions, grouped SKIP per step
    output   z        float32 [64, HORIZON + 1, D]    z[:, 0] embeds the current observation (it must not depend
                                                      on `actions`); z[:, k] is the predicted embedding k steps
                                                      (k*SKIP env steps) later, using only actions[:, :k]
"""

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort

BATCH, HIST, SKIP, HORIZON, SIZE, ACTION_DIM = 64, 3, 5, 5, 64, 2
MAX_DIM = 1024
SUITE_SAMPLES = 33_792  # model inputs processed by the full evaluation
MAX_INFERENCE_SECONDS = 240  # a model whose inference alone would take longer is rejected before the evaluation
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
        try:
            sess = ort.InferenceSession(path, opts, providers=[CPU])
        except Exception as e:
            raise SubmissionError(f"Could not load the ONNX model: {e}") from None
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


def check_speed(sess):
    """Times the model on random inputs and rejects it if the evaluation's inference would take too long."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    n = BATCH * (1 if sess.get_providers()[0] != CPU else max(1, min(cores, 8)))
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (n, HIST, SIZE, SIZE, 3), dtype=np.uint8)
    actions = rng.standard_normal((n,) + ACTIONS_SHAPE[1:]).astype(np.float32)
    run_batches(sess, frames[:BATCH], actions[:BATCH])  # warm-up
    t0 = time.time()
    run_batches(sess, frames, actions)
    estimate = SUITE_SAMPLES * (time.time() - t0) / n
    print(f"inference estimate {estimate:.0f}s", file=sys.stderr, flush=True)
    if estimate > MAX_INFERENCE_SECONDS:
        raise SubmissionError(f"The model is too slow: the evaluation's {SUITE_SAMPLES:,} samples would take "
                              f"~{estimate / 60:.1f} min on the evaluation machine (CPU), over the "
                              f"{MAX_INFERENCE_SECONDS / 60:.0f} min limit. Use a smaller or faster model.")


def load_tasks(data):
    path = os.path.join(data, "tasks.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No tasks.py in the evaluation data at {data!r}")
    spec = importlib.util.spec_from_file_location("challenge_tasks", path)
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    return tasks


def evaluate(model_path, data, verbose=False):
    """Returns {"score": float, "tasks": {...}, "seconds": float}. Raises SubmissionError for problems with the model."""
    t0 = time.time()
    tasks = load_tasks(data)
    sess = load_model(model_path)
    print(f"running on {sess.get_providers()[0]}", file=sys.stderr, flush=True)
    check_speed(sess)
    result = tasks.evaluate(sess, data, sys.modules[__name__], verbose)
    result["seconds"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to the ONNX model")
    ap.add_argument("data", help="directory of the unzipped evaluation data")
    ap.add_argument("--out", default=None, help="also write the per-task results JSON here (organizers only)")
    ap.add_argument("--verbose", action="store_true", help="print per-task results to stderr (organizers only)")
    a = ap.parse_args()
    # nothing but the final JSON may reach stdout: point file descriptor 1 (Python and native libraries) at stderr
    sys.stdout.flush()
    stdout = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            r = evaluate(a.model, os.path.abspath(a.data), a.verbose)
            if not np.isfinite(r["score"]):
                raise SubmissionError("The model's embeddings gave a non-finite score")
            print(f"score {r['score']:.4f}  time {r['seconds']}s", file=sys.stderr)
            if a.out:
                with open(a.out, "w") as f:
                    json.dump(r, f, indent=2)
        out, code = {"score": round(float(r["score"]), 6)}, 0
    except SubmissionError as e:
        out, code = {"error": str(e)}, 1
    except BaseException as e:  # never leave the leaderboard without valid JSON
        traceback.print_exc(file=sys.stderr)
        out, code = {"error": f"The evaluation failed with an internal error ({type(e).__name__}); "
                              "the organizers have the details."}, 1
    print(json.dumps(out), file=stdout, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
