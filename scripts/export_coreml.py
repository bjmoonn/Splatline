#!/usr/bin/env python3
"""Export Apple ML-SHARP model to CoreML (.mlpackage) format.

This script loads the SHARP Gaussian predictor, traces it with torch.jit.trace,
and converts it to a CoreML ML Program using coremltools.  The resulting model
runs on the Apple GPU via CoreML's Metal backend, delivering ~1.4-1.5x speedup
over PyTorch MPS on M4 Max (and likely more on M-series chips with higher
GPU/Neural Engine bandwidth).

Requirements:
    pip install coremltools torch sharp

Usage:
    python scripts/export_coreml.py [--output sharp.mlpackage] [--precision float16]

The exported model accepts:
    - images:    float16 tensor of shape (1, 3, 1536, 1536)
    - disparity: float16 tensor of shape (1,)

and returns five tensors corresponding to the Gaussians3D NamedTuple fields:
    - mean_vectors:    (1, 1179648, 3)
    - singular_values: (1, 1179648, 3)
    - quaternions:     (1, 1179648, 4)
    - colors:          (1, 1179648, 3)
    - opacities:       (1, 1179648)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_OUTPUT = "sharp.mlpackage"
IMAGE_SIZE = 1536


class _PredictorTupleWrapper(torch.nn.Module):
    """Wrap RGBGaussianPredictor to return a plain tuple instead of NamedTuple.

    torch.jit.trace handles NamedTuple returns, but coremltools produces
    auto-numbered output names (var_NNNN).  By returning a plain tuple with
    a known order we can reliably map CoreML outputs back to Gaussians3D
    fields.

    Output order: mean_vectors, singular_values, quaternions, colors, opacities
    """

    def __init__(self, predictor: torch.nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(
        self, image: torch.Tensor, disparity_factor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.predictor(image, disparity_factor)
        return (
            g.mean_vectors,
            g.singular_values,
            g.quaternions,
            g.colors,
            g.opacities,
        )


def _load_predictor(device: str = "cpu") -> torch.nn.Module:
    """Load SHARP predictor with pretrained weights."""
    from sharp.models import PredictorParams, create_predictor

    print("Creating SHARP predictor...")
    predictor = create_predictor(PredictorParams())

    print("Loading pretrained weights...")
    state_dict = torch.hub.load_state_dict_from_url(
        DEFAULT_MODEL_URL, progress=True, map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)

    param_count = sum(p.numel() for p in predictor.parameters())
    print(f"Model loaded: {param_count / 1e6:.1f}M parameters")
    return predictor


def _trace_model(
    predictor: torch.nn.Module,
) -> torch.jit.ScriptModule:
    """Trace the predictor with dummy inputs."""
    wrapper = _PredictorTupleWrapper(predictor)
    wrapper.eval()

    dummy_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dummy_disp = torch.tensor([1.0])

    print(f"Tracing model with input shape (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE})...")
    t0 = time.perf_counter()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_img, dummy_disp))
    dt = time.perf_counter() - t0
    print(f"Tracing completed in {dt:.1f}s")
    return traced


def _convert_to_coreml(
    traced: torch.jit.ScriptModule,
    precision: str = "float16",
) -> "coremltools.models.MLModel":
    """Convert traced model to CoreML ML Program."""
    import coremltools as ct

    compute_precision = (
        ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    )

    print(f"Converting to CoreML (precision={precision})...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="images", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE)),
            ct.TensorType(name="disparity", shape=(1,)),
        ],
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.macOS15,
    )
    dt = time.perf_counter() - t0
    print(f"CoreML conversion completed in {dt:.1f}s")
    return mlmodel


def _get_dir_size(path: str) -> int:
    """Get total size of a directory tree in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Export SHARP model to CoreML .mlpackage"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for .mlpackage (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Compute precision (default: float16, ~1.3 GB; float32 ~2.6 GB)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a quick benchmark comparing PyTorch MPS vs CoreML",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # Step 1: Load model
    predictor = _load_predictor(device="cpu")

    # Step 2: Trace
    traced = _trace_model(predictor)

    # Step 3: Convert to CoreML
    mlmodel = _convert_to_coreml(traced, precision=args.precision)

    # Step 4: Save
    print(f"Saving to {output_path}...")
    mlmodel.save(str(output_path))
    size_gb = _get_dir_size(str(output_path)) / 1e9
    print(f"Saved: {output_path} ({size_gb:.2f} GB)")

    # Step 5: Optional benchmark
    if args.benchmark:
        _run_benchmark(predictor, str(output_path))

    print("\nDone.")


def _run_benchmark(predictor: torch.nn.Module, coreml_path: str):
    """Compare PyTorch MPS vs CoreML inference times."""
    import coremltools as ct
    import numpy as np

    print("\n=== Benchmark: PyTorch MPS vs CoreML ===\n")

    n_warmup = 3
    n_runs = 5

    # --- PyTorch MPS ---
    if torch.mps.is_available():
        predictor_mps = predictor.to("mps")
        img_mps = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="mps")
        disp_mps = torch.tensor([1.0], device="mps")

        print(f"PyTorch MPS: warming up ({n_warmup} runs)...")
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = predictor_mps(img_mps, disp_mps)
                torch.mps.synchronize()

        print(f"PyTorch MPS: benchmarking ({n_runs} runs)...")
        mps_times = []
        with torch.no_grad():
            for i in range(n_runs):
                torch.mps.synchronize()
                t0 = time.perf_counter()
                _ = predictor_mps(img_mps, disp_mps)
                torch.mps.synchronize()
                t1 = time.perf_counter()
                mps_times.append(t1 - t0)
                print(f"  Run {i + 1}: {mps_times[-1]:.3f}s")

        avg_mps = sum(mps_times) / len(mps_times)
        print(f"  Average: {avg_mps:.3f}s\n")

        del predictor_mps, img_mps, disp_mps
        torch.mps.empty_cache()
    else:
        avg_mps = None
        print("MPS not available, skipping PyTorch MPS benchmark.\n")

    # --- CoreML (CPU_AND_GPU, best config) ---
    print("CoreML (CPU_AND_GPU): loading model...")
    mlmodel = ct.models.MLModel(coreml_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    img_np = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float16)
    disp_np = np.array([1.0], dtype=np.float16)

    print(f"CoreML: warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        _ = mlmodel.predict({"images": img_np, "disparity": disp_np})

    print(f"CoreML: benchmarking ({n_runs} runs)...")
    coreml_times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        _ = mlmodel.predict({"images": img_np, "disparity": disp_np})
        t1 = time.perf_counter()
        coreml_times.append(t1 - t0)
        print(f"  Run {i + 1}: {coreml_times[-1]:.3f}s")

    avg_coreml = sum(coreml_times) / len(coreml_times)
    print(f"  Average: {avg_coreml:.3f}s\n")

    # --- Summary ---
    print("=== Summary ===")
    if avg_mps is not None:
        print(f"PyTorch MPS: {avg_mps:.3f}s per frame")
    print(f"CoreML:      {avg_coreml:.3f}s per frame")
    if avg_mps is not None:
        print(f"Speedup:     {avg_mps / avg_coreml:.2f}x")


if __name__ == "__main__":
    main()
