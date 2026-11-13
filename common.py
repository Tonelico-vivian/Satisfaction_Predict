"""Shared utilities for the single family-regularized stack."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require_compute():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run preparation, training and inference through sbatch run.sh.")


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def competition_data(directory):
    import numpy as np
    import pandas as pd

    directory = Path(directory)
    expected = read_json(ROOT / "metadata/data_audit.json")["data_sha256"]
    for name, digest in expected.items():
        assert fingerprint(directory / name) == digest, name
    train, test, sample = [pd.read_csv(directory / name) for name in
                           ("train.csv", "test.csv", "sample_submission.csv")]
    assert set(train.satisfaction.unique()) == {False, True}
    train["satisfaction"] = train.satisfaction.astype(np.int8)
    assert train.id.is_unique and test.id.is_unique
    assert not train.id.isin(test.id).any()
    assert sample.id.equals(test.id)
    assert list(sample.columns) == ["id", "satisfaction"]
    return train, test, sample
