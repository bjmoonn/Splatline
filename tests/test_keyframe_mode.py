"""Tests for --keyframe-interval symlink logic."""

import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so the converter module can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.converters.video_to_3d_high_quality import create_keyframe_symlinks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_fake_frames(frames_dir: Path, count: int) -> None:
    """Create *count* fake PNG files named frame_000000.png ... frame_NNNNNN.png."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (frames_dir / f"frame_{i:06d}.png").write_bytes(b"\x89PNG fake")


def _create_keyframe_plys(gaussians_dir: Path, total: int, interval: int) -> None:
    """Create small dummy PLY files for keyframe indices only."""
    gaussians_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, total, interval):
        ply = gaussians_dir / f"frame_{k:06d}.ply"
        ply.write_text(f"ply-keyframe-{k}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateKeyframeSymlinks:
    """Tests for ``create_keyframe_symlinks``."""

    def test_basic_symlink_creation(self, tmp_path: Path):
        """10 frames, interval=5 -> keyframes 0 and 5, 8 symlinks."""
        total = 10
        interval = 5
        gaussians_dir = tmp_path / "gaussians"

        _create_keyframe_plys(gaussians_dir, total, interval)

        symlinks, copies = create_keyframe_symlinks(gaussians_dir, total, interval)

        # All entries exist
        entries = sorted(gaussians_dir.glob("*.ply"))
        assert len(entries) == total, (
            f"Expected {total} PLY entries, got {len(entries)}"
        )

        # Exactly 2 real files (keyframes 0 and 5)
        real_files = [p for p in entries if not p.is_symlink()]
        assert len(real_files) == 2, (
            f"Expected 2 real keyframe files, got {len(real_files)}: "
            f"{[p.name for p in real_files]}"
        )

        # 8 symlinks
        sym_files = [p for p in entries if p.is_symlink()]
        assert len(sym_files) == 8, (
            f"Expected 8 symlinks, got {len(sym_files)}"
        )

        # Counts returned correctly
        assert symlinks + copies == 8

    def test_symlinks_resolve_to_correct_keyframe(self, tmp_path: Path):
        """Each symlink should point to the nearest (preceding) keyframe."""
        total = 10
        interval = 5
        gaussians_dir = tmp_path / "gaussians"

        _create_keyframe_plys(gaussians_dir, total, interval)
        create_keyframe_symlinks(gaussians_dir, total, interval)

        # Frames 1-4 should link to frame_000000.ply
        for i in range(1, 5):
            link = gaussians_dir / f"frame_{i:06d}.ply"
            assert link.is_symlink()
            target = os.readlink(str(link))
            assert target == "frame_000000.ply", (
                f"frame_{i:06d}.ply -> {target}, expected frame_000000.ply"
            )

        # Frames 6-9 should link to frame_000005.ply
        for i in range(6, 10):
            link = gaussians_dir / f"frame_{i:06d}.ply"
            assert link.is_symlink()
            target = os.readlink(str(link))
            assert target == "frame_000005.ply", (
                f"frame_{i:06d}.ply -> {target}, expected frame_000005.ply"
            )

    def test_symlink_content_matches_keyframe(self, tmp_path: Path):
        """Reading through a symlink should return the keyframe's content."""
        total = 10
        interval = 5
        gaussians_dir = tmp_path / "gaussians"

        _create_keyframe_plys(gaussians_dir, total, interval)
        create_keyframe_symlinks(gaussians_dir, total, interval)

        for i in range(1, 5):
            content = (gaussians_dir / f"frame_{i:06d}.ply").read_text()
            assert content == "ply-keyframe-0"

        for i in range(6, 10):
            content = (gaussians_dir / f"frame_{i:06d}.ply").read_text()
            assert content == "ply-keyframe-5"

    def test_interval_1_no_symlinks(self, tmp_path: Path):
        """Interval=1 means every frame is a keyframe; no symlinks created."""
        total = 5
        interval = 1
        gaussians_dir = tmp_path / "gaussians"

        _create_keyframe_plys(gaussians_dir, total, interval)
        symlinks, copies = create_keyframe_symlinks(gaussians_dir, total, interval)

        assert symlinks == 0
        assert copies == 0
        assert len(list(gaussians_dir.glob("*.ply"))) == total

    def test_interval_larger_than_total(self, tmp_path: Path):
        """If interval > total, only frame 0 is a keyframe."""
        total = 4
        interval = 10
        gaussians_dir = tmp_path / "gaussians"

        _create_keyframe_plys(gaussians_dir, total, interval)
        symlinks, copies = create_keyframe_symlinks(gaussians_dir, total, interval)

        entries = sorted(gaussians_dir.glob("*.ply"))
        assert len(entries) == total

        real = [p for p in entries if not p.is_symlink()]
        assert len(real) == 1
        assert real[0].name == "frame_000000.ply"

        assert symlinks + copies == 3

    def test_does_not_overwrite_existing(self, tmp_path: Path):
        """Pre-existing files in gaussians_dir are not overwritten."""
        total = 4
        interval = 2
        gaussians_dir = tmp_path / "gaussians"

        _create_keyframe_plys(gaussians_dir, total, interval)

        # Manually create frame_000001.ply as a regular file
        manual = gaussians_dir / "frame_000001.ply"
        manual.write_text("manually-created")

        create_keyframe_symlinks(gaussians_dir, total, interval)

        # The manually created file should be untouched
        assert not manual.is_symlink()
        assert manual.read_text() == "manually-created"

    def test_relative_symlink_paths(self, tmp_path: Path):
        """Symlinks must use relative (filename-only) targets, not absolute."""
        total = 6
        interval = 3
        gaussians_dir = tmp_path / "gaussians"

        _create_keyframe_plys(gaussians_dir, total, interval)
        create_keyframe_symlinks(gaussians_dir, total, interval)

        for i in range(total):
            p = gaussians_dir / f"frame_{i:06d}.ply"
            if p.is_symlink():
                target = os.readlink(str(p))
                assert "/" not in target, (
                    f"Symlink target should be relative (filename only), "
                    f"got: {target}"
                )
