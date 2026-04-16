"""CoreML-based SHARP predictor -- drop-in replacement for PyTorch predictor.

This module loads an exported CoreML .mlpackage (produced by
``scripts/export_coreml.py``) and wraps it so callers can use the exact same
interface as the original ``RGBGaussianPredictor``:

    predictor = CoreMLPredictor("sharp.mlpackage")
    gaussians = predictor(image_tensor, disparity_factor_tensor)
    # gaussians is a Gaussians3D NamedTuple, same as the PyTorch version.

On M4 Max, CoreML (CPU_AND_GPU) runs ~1.5x faster than PyTorch MPS for the
full SHARP model at 1536x1536 resolution (~2.0s vs ~3.0s per frame).

Requirements:
    pip install coremltools torch

Notes:
    - The CoreML model uses FLOAT16 precision by default.  Numerical
      differences vs FP32 PyTorch are small (mean abs diff < 0.001 for
      all output fields except quaternions which has ~0.001).
    - Input tensors can be on any device; they will be moved to CPU/numpy
      for CoreML inference.  Output tensors are returned on CPU.
    - The model is fixed to 1536x1536 input resolution.  If your images
      are a different size, resize them before calling the predictor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sharp.utils.gaussians import Gaussians3D

logger = logging.getLogger(__name__)

# CoreML output names produced by the export script.  These are auto-generated
# by coremltools from the traced tuple output order:
#   (mean_vectors, singular_values, quaternions, colors, opacities)
#
# The exact names depend on the graph and may change if the model architecture
# changes.  We discover them at load time by matching output shapes.
_EXPECTED_SHAPES = {
    "mean_vectors": (1, 1179648, 3),     # 768*768*2 = 1179648 gaussians, xyz
    "singular_values": (1, 1179648, 3),   # 3 scale components
    "quaternions": (1, 1179648, 4),       # wxyz quaternion
    "colors": (1, 1179648, 3),            # RGB
    "opacities": (1, 1179648),            # scalar opacity per gaussian
}


def _discover_output_mapping(
    mlmodel: "coremltools.models.MLModel",
) -> dict[str, str]:
    """Map Gaussians3D field names to CoreML output names by shape + test run.

    The CoreML model has auto-numbered output names (var_NNNN).  We figure out
    which is which by running a dummy prediction and matching output shapes.
    For the three (1, N, 3) outputs (mean_vectors, singular_values, colors) we
    disambiguate by checking value ranges on a random input.
    """
    import coremltools as ct

    spec = mlmodel.get_spec()
    outputs = spec.description.output

    # Group CoreML output names by shape
    shape_to_names: dict[tuple, list[str]] = {}
    for out in outputs:
        arr_type = out.type.multiArrayType
        shape = tuple(arr_type.shape)
        shape_to_names.setdefault(shape, []).append(out.name)

    mapping: dict[str, str] = {}

    # Unique shapes map directly
    # quaternions: (1, 1179648, 4) -- unique
    quat_shape = _EXPECTED_SHAPES["quaternions"]
    if quat_shape in shape_to_names and len(shape_to_names[quat_shape]) == 1:
        mapping["quaternions"] = shape_to_names[quat_shape][0]

    # opacities: (1, 1179648) -- unique
    opac_shape = _EXPECTED_SHAPES["opacities"]
    if opac_shape in shape_to_names and len(shape_to_names[opac_shape]) == 1:
        mapping["opacities"] = shape_to_names[opac_shape][0]

    # For the three (1, N, 3) outputs, run a dummy prediction and
    # disambiguate by value range:
    #   - singular_values: all positive, small (typically < 0.02)
    #   - colors: all in [0, 1]
    #   - mean_vectors: can be negative, range roughly [-1.5, 1.5]
    trio_shape = _EXPECTED_SHAPES["mean_vectors"]
    trio_names = shape_to_names.get(trio_shape, [])

    if len(trio_names) == 3:
        img_np = np.random.randn(1, 3, 1536, 1536).astype(np.float16)
        disp_np = np.array([1.0], dtype=np.float16)
        result = mlmodel.predict({"images": img_np, "disparity": disp_np})

        # Score each candidate
        candidates = {}
        for name in trio_names:
            arr = result[name]
            candidates[name] = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean_abs": float(np.abs(arr).mean()),
            }

        # singular_values: smallest mean_abs (values are tiny ~0.001)
        sv_name = min(trio_names, key=lambda n: candidates[n]["mean_abs"])
        mapping["singular_values"] = sv_name
        remaining = [n for n in trio_names if n != sv_name]

        # colors: values in [0, 1], min >= 0
        # mean_vectors: can go negative
        for name in remaining:
            if candidates[name]["min"] >= -0.01:
                mapping["colors"] = name
            else:
                mapping["mean_vectors"] = name

        # Safety: if both remaining have negatives, use larger mean_abs for mean_vectors
        if "colors" not in mapping or "mean_vectors" not in mapping:
            remaining_sorted = sorted(
                remaining, key=lambda n: candidates[n]["mean_abs"], reverse=True
            )
            mapping.setdefault("mean_vectors", remaining_sorted[0])
            mapping.setdefault(
                "colors",
                remaining_sorted[1] if len(remaining_sorted) > 1 else remaining_sorted[0],
            )

    logger.info("CoreML output mapping: %s", mapping)
    return mapping


class CoreMLPredictor:
    """Drop-in CoreML replacement for ``sharp.models.predictor.RGBGaussianPredictor``.

    Loads a .mlpackage exported by ``scripts/export_coreml.py`` and provides
    the same ``__call__`` interface returning ``Gaussians3D``.

    Args:
        model_path: Path to the .mlpackage directory.
        compute_units: Which Apple compute units to use.  "cpu_and_gpu" is
            fastest on M4 Max.  Options: "all", "cpu_and_gpu", "cpu_and_ne",
            "cpu_only".
    """

    # Map from our friendly names to coremltools constants
    _COMPUTE_UNIT_MAP = {
        "all": "ALL",
        "cpu_and_gpu": "CPU_AND_GPU",
        "cpu_and_ne": "CPU_AND_NE",
        "cpu_only": "CPU_ONLY",
    }

    def __init__(
        self,
        model_path: str | Path,
        compute_units: str = "cpu_and_gpu",
    ):
        import coremltools as ct

        model_path = str(model_path)
        cu_name = self._COMPUTE_UNIT_MAP.get(
            compute_units.lower(), compute_units.upper()
        )
        cu = getattr(ct.ComputeUnit, cu_name)

        logger.info("Loading CoreML model from %s (compute_units=%s)", model_path, cu_name)
        self._model = ct.models.MLModel(model_path, compute_units=cu)

        # Discover which auto-generated output name maps to which field
        self._output_map = _discover_output_mapping(self._model)

    def __call__(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> Gaussians3D:
        """Run CoreML inference and return Gaussians3D.

        Args:
            image: (B, 3, H, W) image tensor.  B must be 1 for CoreML.
            disparity_factor: (B,) disparity factor tensor.
            depth: Ignored (CoreML model was traced without ground-truth depth).

        Returns:
            Gaussians3D NamedTuple with tensors on CPU.
        """
        if image.shape[0] != 1:
            raise ValueError(
                f"CoreML predictor only supports batch_size=1, got {image.shape[0]}. "
                "Loop over frames individually."
            )

        # Convert to numpy (float16 to match CoreML model precision)
        img_np = image.detach().cpu().numpy().astype(np.float16)
        disp_np = disparity_factor.detach().cpu().numpy().astype(np.float16)

        # Run CoreML inference
        result = self._model.predict({"images": img_np, "disparity": disp_np})

        # Reconstruct Gaussians3D from CoreML outputs
        return Gaussians3D(
            mean_vectors=torch.from_numpy(result[self._output_map["mean_vectors"]]),
            singular_values=torch.from_numpy(result[self._output_map["singular_values"]]),
            quaternions=torch.from_numpy(result[self._output_map["quaternions"]]),
            colors=torch.from_numpy(result[self._output_map["colors"]]),
            opacities=torch.from_numpy(result[self._output_map["opacities"]]),
        )

    def internal_resolution(self) -> int:
        """Internal resolution (matches PyTorch model)."""
        return 1536

    @property
    def output_resolution(self) -> int:
        """Output resolution of Gaussians."""
        return self.internal_resolution() // 2

    def to(self, device: torch.device | str) -> "CoreMLPredictor":
        """No-op: CoreML manages its own device placement.

        Provided for API compatibility with PyTorch predictor usage patterns
        like ``predictor.to(device)``.
        """
        return self

    def eval(self) -> "CoreMLPredictor":
        """No-op: CoreML model is always in eval mode."""
        return self


def create_coreml_predictor(
    model_path: str | Path = "sharp.mlpackage",
    compute_units: str = "cpu_and_gpu",
) -> CoreMLPredictor:
    """Factory function matching the pattern of ``sharp.models.create_predictor``.

    Args:
        model_path: Path to the exported .mlpackage.
        compute_units: Apple compute units to use.

    Returns:
        A CoreMLPredictor instance.
    """
    return CoreMLPredictor(model_path=model_path, compute_units=compute_units)
