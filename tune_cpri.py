"""
tune_cpri.py — hyperparameter search on CPRI's own dataset, CPRI's protocol.

    python tune_cpri.py

Every model, CPRI's own kNN included, is tuned by 5-fold cross-validation on the
Train split ONLY (1400 rows). The Validation + Hidden Test rows (600) are never
seen during tuning; they are scored once, after refitting on all of Train.
Writes regression_cpri_tuning.csv (chosen settings) and
regression_cpri_tuned.csv (default vs tuned scores, paired against CPRI's kNN).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import regression as R

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
APPDIR = os.path.join(HERE, "app")
CV = KFold(5, shuffle=True, random_state=0)
TARGETS = ("front", "tail", "crest")


def _grids():
    return {
        "Ridge": (make_pipeline(StandardScaler(), Ridge()),
                  {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}),
        "Ridge + degree-2 terms": (
            make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler(),
                          Ridge()),
            {"ridge__alpha": [0.1, 1, 10, 100, 1000, 10000]}),
        "kNN, standardised": (
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            {"kneighborsregressor__n_neighbors": [3, 5, 7, 11, 15, 25, 40],
             "kneighborsregressor__weights": ["uniform", "distance"]}),
        "Random forest": (
            RandomForestRegressor(300, random_state=0, n_jobs=-1),
            {"min_samples_leaf": [1, 3, 8, 20], "max_features": [0.5, 1.0]}),
        "Gradient boosting": (
            GradientBoostingRegressor(random_state=0),
            {"n_estimators": [100, 300], "max_depth": [2, 3, 4],
             "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.7, 1.0]}),
        "Neural network (MLP)": (
            R.make_mlp(),
            {"regressor__mlpregressor__hidden_layer_sizes": [(32,), (64,), (64, 64)],
             "regressor__mlpregressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1]}),
    }


def tune_cpri_knn_k(train):
    """CPRI's own kNN with the same distance and weights, k searched too, so the
    baseline gets the same tuning effort as everything it is compared with."""
    best = {}
    folds = list(CV.split(train))
    for k_try in (3, 5, 7, 11, 15, 25):
        err = np.zeros(3)
        for tr, va in folds:
            a, b = train.iloc[tr], train.iloc[va]
            pred = _knn_k(a, b, k_try)
            for j in range(3):
                y = b[R.CPRI_RS[j]].to_numpy() / b[R.CPRI_PH[j]].to_numpy()
                err[j] += np.abs(pred[:, j] / b[R.CPRI_PH[j]].to_numpy() - y).mean()
        best[k_try] = err / len(folds)
    return {j: min(best, key=lambda k: best[k][j]) for j in range(3)}, best


def _knn_k(train, query, k):
    A = train[R.CPRI_X].to_numpy(float)
    Res = train[R.CPRI_RS].to_numpy(float)
    base = np.array([1500, 1300, 800, 350, 30, 1.0])
    out = []
    for q in query[R.CPRI_X].to_numpy(float):
        rs = 100.0 if q[0] == 0 else 30000.0
        dist = (A[:, 0] != q[0]) + (((A[:, 1:] - q[1:]) / np.r_[base, rs, rs]) ** 2).sum(1)
        idx = np.argsort(dist, kind="stable")[:k]
        w = 1.0 / (dist[idx] + 1e-6)
        out.append((Res[idx] * w[:, None]).sum(0) / w.sum())
    return np.array(out)


def main():
    train, test = R.load_cpri()
    ph = test[R.CPRI_PH].to_numpy()
    act = test[R.CPRI_OB].to_numpy()
    Xtr, Xte = train[R.CPRI_X].to_numpy(float), test[R.CPRI_X].to_numpy(float)

    default_tbl, _, _, _ = R.cpri_benchmark()
    preds = {"CPRI weighted kNN (their method, k=7)":
             ph + R.cpri_knn(train, test)}

    tuning_rows = []
    ks, grid = tune_cpri_knn_k(train)
    preds["CPRI weighted kNN (k tuned by CV)"] = ph + np.column_stack(
        [_knn_k(train, test, ks[j])[:, j] for j in range(3)])
    tuning_rows.append({"model": "CPRI weighted kNN (k tuned by CV)",
                        **{f"{TARGETS[j]}_best": f"k={ks[j]}" for j in range(3)},
                        "cv_mae_front": round(grid[ks[0]][0], 5)})
    print(f"CPRI kNN k by CV: {ks}", flush=True)

    for name, (est, params) in _grids().items():
        cols, picks = [], []
        for j in range(3):
            y = train[R.CPRI_RS[j]].to_numpy() / train[R.CPRI_PH[j]].to_numpy()
            gs = GridSearchCV(est, params, cv=CV, n_jobs=-1, refit=True,
                              scoring="neg_mean_absolute_error").fit(Xtr, y)
            cols.append(gs.predict(Xte) * ph[:, j])
            picks.append(gs.best_params_)
            print(f"  {name:26s} {TARGETS[j]:5s} cv-MAE {-gs.best_score_:.5f} "
                  f"{gs.best_params_}", flush=True)
        preds[f"{name} (tuned)"] = ph + np.column_stack(cols)
        tuning_rows.append({"model": f"{name} (tuned)",
                            **{f"{TARGETS[j]}_best": str(picks[j]) for j in range(3)}})

    rows, paired = [], []
    bases = {"CPRI kNN k=7 (as shipped)": "CPRI weighted kNN (their method, k=7)",
             "CPRI kNN k tuned by CV": "CPRI weighted kNN (k tuned by CV)"}
    for name, p in preds.items():
        rows.append(R._cpri_score(test, p, name))
        err = np.abs(p - act)
        for bname, bkey in bases.items():
            if name == bkey:
                continue
            base_err = np.abs(preds[bkey] - act)
            for j, t in enumerate(TARGETS):
                pv = float(wilcoxon(err[:, j], base_err[:, j]).pvalue)
                paired.append({"model": name, "against": bname, "target": t,
                               "better": int((err[:, j] < base_err[:, j]).sum()),
                               "worse": int((err[:, j] > base_err[:, j]).sum()),
                               "wilcoxon_p": round(pv, 4)})
    out = pd.DataFrame(rows)
    out = pd.concat([default_tbl.assign(model=default_tbl.model + " [default]"), out])
    out.round(4).to_csv(os.path.join(RESULTS, "regression_cpri_tuned.csv"), index=False)
    pd.DataFrame(tuning_rows).to_csv(os.path.join(RESULTS, "regression_cpri_tuning.csv"),
                                     index=False)
    pd.DataFrame(paired).to_csv(os.path.join(RESULTS, "regression_cpri_tuned_paired.csv"),
                                index=False)
    pd.set_option("display.width", 220)
    print("\nMAPE % — default (hand-set) vs tuned, scored on Validation + Hidden Test")
    print(out[["model", "front_mape_%", "tail_mape_%", "crest_mape_%"]]
          .round(3).to_string(index=False))
    print("\nPaired vs CPRI's kNN (k=7)")
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
