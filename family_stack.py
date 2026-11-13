"""Method 3 only: family-scaled L2 logistic regression, CV and inference."""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from common import ROOT, competition_data, fingerprint, read_json, require_compute, save_json


def family(name):
    if "tabpfn" in name or name.startswith("pfn_"):
        return "tabpfn"
    if name.startswith("tfm_") or "tabicl" in name or "tabm" in name:
        return "tfm"
    if "realmlp" in name or "rmlp" in name or "mlp" in name:
        return "mlp"
    return "gbdt"


def fit_lr(x, y, tol=1e-4):
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        return LogisticRegression(C=.003, solver="lbfgs", tol=tol, max_iter=5000).fit(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cv", "train", "predict"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs")
    parser.add_argument("--model", type=Path, default=ROOT / "model/model_family.joblib")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 101])
    parser.add_argument("--tol", type=float, default=1e-4, help="CV tolerance only; final training remains 1e-4.")
    args = parser.parse_args()
    require_compute()
    train, _, sample = competition_data(args.data_dir)
    members = read_json(ROOT / "metadata/members.json")
    names = [m["name"] for m in members]
    scale = np.array([{"tabpfn": 3., "tfm": 1.5}.get(family(n), 1.) for n in names])
    assert np.array_equal(scale, [m["scale"] for m in members])
    expected = read_json(ROOT / "metadata/reproduction.json")
    required = ["test_logits.npy"] if args.command == "predict" else ["oof_logits.npy", "test_logits.npy"]
    for name in required:
        assert fingerprint(args.input_dir / name) == expected["matrix_sha256"][name], name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command != "predict":
        x = np.load(args.input_dir / "oof_logits.npy", mmap_mode="r") * scale
        y = train.satisfaction.to_numpy()
        if args.command == "cv":
            results = []
            for seed in args.seeds:
                oof = np.full(len(y), np.nan)
                folds = []
                splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
                for k, (a, b) in enumerate(splitter.split(np.zeros(len(y)), y)):
                    model = fit_lr(x[a], y[a], args.tol)
                    with threadpool_limits(limits=1):
                        oof[b] = model.decision_function(x[b])
                    folds.append(float(roc_auc_score(y[b], oof[b])))
                    print(f"seed={seed} fold={k} AUC={folds[-1]:.9f}", flush=True)
                results.append({"seed": seed, "tol": args.tol, "auc": float(roc_auc_score(y, oof)),
                                "fold_auc": folds, "scope": "meta CV of fixed public OOF"})
                save_json(args.output_dir / "cv.json", results)
            return
        model = fit_lr(x, y)
        del x
        saved = {"model": model, "scale": scale, "names": names,
                 "protocol_sha256": fingerprint(ROOT / "metadata/protocol.json")}
        model_path = args.output_dir / "model_family.joblib"
        joblib.dump(saved, model_path)
        reloaded = joblib.load(model_path)
        assert np.array_equal(reloaded["model"].coef_, model.coef_)
    else:
        if args.model.resolve() == (ROOT / "model/model_family.joblib").resolve():
            assert fingerprint(args.model) == expected["model_sha256"]
        saved = joblib.load(args.model)
    assert saved["names"] == names and np.array_equal(saved["scale"], scale)
    test = np.load(args.input_dir / "test_logits.npy", mmap_mode="r")
    assert test.shape == (len(sample), 245)
    with threadpool_limits(limits=1):
        prediction = expit(saved["model"].decision_function(test * saved["scale"]))
    assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    output = args.output_dir / "submission_family.csv"
    pd.DataFrame({"id": sample.id, "satisfaction": prediction}).to_csv(output, index=False)
    reread = pd.read_csv(output)
    assert reread.id.equals(sample.id)
    error = float(np.max(np.abs(reread.satisfaction - prediction)))
    assert error < 1e-12
    digest = fingerprint(output)
    if args.command == "predict" and args.model.resolve() == (ROOT / "model/model_family.joblib").resolve():
        assert digest == expected["submission_sha256"], "Original model must reproduce the registered CSV."
    save_json(args.output_dir / "verification.json",
              {"rows": len(prediction), "csv_reload_max_error": error, "submission_sha256": digest})
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
