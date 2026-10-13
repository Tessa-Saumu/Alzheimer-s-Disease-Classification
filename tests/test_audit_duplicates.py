"""Check exact contamination, perceptual candidates, and invalid manifests."""
import shutil

import pytest
from PIL import Image

from audit_duplicates import audit


def test_exact_and_perceptual_collisions(tmp_path):
    """Generated images establish mechanics without medical data or provenance claims."""
    Image.new("L", (16, 16), 32).save(tmp_path / "train.png")
    shutil.copyfile(tmp_path / "train.png", tmp_path / "test.png")
    Image.new("L", (16, 16), 64).save(tmp_path / "different.png")
    manifest = tmp_path / "split.csv"
    manifest.write_text("path,split\ntrain.png,train\ntest.png,test\ndifferent.png,validation\n")
    result = audit(manifest)
    assert len(result["cross_split_groups"]["sha256"]) == 1
    assert len(result["cross_split_groups"]["sha256"][0]["images"]) == 2
    # Flat but byte-distinct images deliberately collide perceptually.
    assert len(result["cross_split_groups"]["average_hash_8x8"][0]["images"]) == 3
    assert len(result["manifest_sha256"]) == 64


def test_invalid_split_and_repeated_file(tmp_path):
    """Reject invalid split labels and repeated paths rather than fabricating evidence."""
    Image.new("L", (8, 8)).save(tmp_path / "image.png")
    manifest = tmp_path / "split.csv"
    manifest.write_text("path,split\nimage.png,holdout\n")
    with pytest.raises(ValueError, match="Unknown split"):
        audit(manifest)
    manifest.write_text("path,split\nimage.png,train\nimage.png,test\n")
    with pytest.raises(ValueError, match="Repeated image"):
        audit(manifest)
