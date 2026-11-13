"""Assemble the 245 pinned members directly, without earlier experiment code."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from common import ROOT, competition_data, fingerprint, read_json, require_compute, save_json


def csv_column(path, ids, column):
    frame = pd.read_csv(path)
    assert frame.id.is_unique and len(frame) == len(ids), path
    assert set(frame.id) == set(ids), path
    return frame.set_index("id").loc[ids, column].to_numpy()


def read_pair(member, public, train, test):
    paths = [public / file for file in member["files"]]
    tag = member["source_tag"]
    if paths[0].suffix == ".npy":
        return [np.load(path, allow_pickle=False).ravel() for path in paths]
    if paths[0].suffix == ".npz":
        with np.load(paths[0], allow_pickle=False) as data:
            if tag == "amanatar":
                assert np.array_equal(data["y"], train.satisfaction)
                stem = member["name"].removeprefix("av5_")
                return data["oof_" + stem].ravel(), data["test_" + stem].ravel()
            return data["oof"].ravel(), data["test"].ravel()
    column = {"abdullah": "pred", "yekenot": "satisfaction"}.get(tag)
    if tag == "kagankoral":
        column = member["name"].removeprefix("kg_")
    assert column is not None, member["name"]
    return csv_column(paths[0], train.id, column), csv_column(paths[1], test.id, column)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--public-dir", type=Path, default=ROOT / "data/public")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "inputs")
    args = parser.parse_args()
    require_compute()
    train, test, _ = competition_data(args.data_dir)
    members = read_json(ROOT / "metadata/members.json")
    assert len(members) == 245
    assert [m["index"] for m in members] == list(range(245))
    assert np.array_equal(pd.read_csv(args.public_dir / "koumei/train_ids_folds.csv").id, train.id)
    assert np.array_equal(pd.read_csv(args.public_dir / "koumei/test_ids.csv").id, test.id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = [
        np.lib.format.open_memmap(args.output_dir / name, mode="w+", dtype="float64", shape=(n, 245))
        for name, n in (("oof_logits.npy", len(train)), ("test_logits.npy", len(test)))
    ]
    for j, member in enumerate(members):
        pair = read_pair(member, args.public_dir, train, test)
        for matrix, values in zip(arrays, pair):
            assert len(values) == len(matrix) and np.isfinite(values).all(), member["name"]
            values = values.astype(np.float64)
            if member["input_representation"] == "probability":
                assert ((values >= 0) & (values <= 1)).all(), member["name"]
                values = np.clip(logit(np.clip(values, 1e-15, 1 - 1e-15)), -30, 30)
            matrix[:, j] = values
        auc = roc_auc_score(train.satisfaction, pair[0])
        assert abs(auc - member["oof_auc"]) < 2e-6, member["name"]
        print(f"{j + 1}/245 {member['name']}", flush=True)
    for matrix in arrays:
        matrix.flush()
    expected = read_json(ROOT / "metadata/reproduction.json")["matrix_sha256"]
    actual = {name: fingerprint(args.output_dir / name) for name in expected}
    assert actual == expected, "Prepared matrices differ from the original method-3 inputs."
    save_json(args.output_dir / "prepared.json", {"matrix_sha256": actual, "members": 245})


if __name__ == "__main__":
    main()
