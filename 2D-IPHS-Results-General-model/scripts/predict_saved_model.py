from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from src.data import MOLECULAR_COLS, PRESSURE_COL, TARGET_COL, TEXTURAL_COLS
from src.results_plots import save_scatter_true_vs_pred_with_ci


def _histogram_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feature_")]


def _build_feature_columns(
    df: pd.DataFrame,
    include_textural: bool,
    include_ads: bool,
    include_histogram: bool,
) -> list[str]:
    cols = (_histogram_columns(df) if include_histogram else []) + MOLECULAR_COLS + [PRESSURE_COL]
    if include_textural:
        cols.extend(TEXTURAL_COLS)
    if include_ads and "ads" in df.columns:
        cols.append("ads")
    return cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a saved model + preprocessor.")
    parser.add_argument("--input-csv", required=True, help="CSV to predict.")
    parser.add_argument("--model-path", required=True, help="Path to *.keras model.")
    parser.add_argument("--preprocessor-path", required=True, help="Path to preprocessor.joblib.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Write predictions CSV to this path (use this or --output-dir, not both).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help=(
            "Directory for predictions.csv plus evaluation plots when the "
            f"input has a '{TARGET_COL}' column (parity scatter and residual histogram)."
        ),
    )
    parser.add_argument("--include-textural", action="store_true", help="Include textural columns.")
    parser.add_argument("--include-ads", action="store_true", help="Include ads categorical column.")
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Exclude feature_* MOF histogram columns (must match training).",
    )
    return parser.parse_args()


def _save_eval_plots(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mae = float(np.mean(np.abs(yt - yp)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    residuals = yt - yp

    save_scatter_true_vs_pred_with_ci(
        y_true,
        y_pred,
        output_dir / "parity_true_vs_pred.png",
        title="Prediction vs ground truth",
        xlabel="Ground truth loading",
        ylabel="Predicted loading",
        test_mae=mae,
        test_r2=r2,
    )

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_hist.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    if bool(args.output_csv) == bool(args.output_dir):
        raise SystemExit("Specify exactly one of --output-csv or --output-dir.")

    in_path = Path(args.input_csv)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.csv"
    else:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    feature_cols = _build_feature_columns(
        df,
        include_textural=bool(args.include_textural),
        include_ads=bool(args.include_ads),
        include_histogram=not bool(args.no_histogram),
    )

    preprocessor = joblib.load(args.preprocessor_path)
    model = keras.models.load_model(args.model_path)

    x = preprocessor.transform(df[feature_cols])
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    pred = model.predict(x, verbose=0).reshape(-1)

    output = df.copy()
    output["prediction_loading"] = pred
    output.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path.resolve()}")

    if args.output_dir and TARGET_COL in df.columns:
        y_true = df[TARGET_COL].to_numpy(dtype=np.float64)
        _save_eval_plots(y_true, pred, Path(args.output_dir))
        od = Path(args.output_dir).resolve()
        print(f"Saved parity plot to: {od / 'parity_true_vs_pred.png'}")
        print(f"Saved residual histogram to: {od / 'residual_hist.png'}")
    elif args.output_dir:
        print(f"Column '{TARGET_COL}' not in input; skipped parity and residual plots.")


if __name__ == "__main__":
    main()

