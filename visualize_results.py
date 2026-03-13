import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training results from results/<model>/<dataset>/results.json")
    parser.add_argument("--results_root", default="results", help="Root directory containing model/dataset results folders")
    parser.add_argument("--output_dir", default="results_visualizations", help="Directory where plots and summaries are saved")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    return parser.parse_args()


def find_result_files(results_root: str) -> List[str]:
    result_files = []
    for root, _, files in os.walk(results_root):
        for filename in files:
            if filename == "results.json":
                result_files.append(os.path.join(root, filename))
    return sorted(result_files)


def infer_model_dataset(results_root: str, file_path: str) -> Dict[str, str]:
    rel = os.path.relpath(file_path, results_root)
    parts = rel.split(os.sep)
    # Expected layout:
    # results/<model_path...>/<dataset>/results.json
    # where <model_path...> can contain one or more nested folders.
    if len(parts) >= 3:
        dataset = parts[-2]
        model = "/".join(parts[:-2])
    else:
        model = parts[0] if len(parts) > 0 else "unknown_model"
        dataset = parts[1] if len(parts) > 1 else "unknown_dataset"
    return {"model": model, "dataset": dataset}


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_array_accuracy(y_true: List, y_pred: List) -> Optional[float]:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return None
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return float((y_true_arr == y_pred_arr).mean())


def load_runs_df(results_root: str) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], Dict[str, List]]]:
    runs = []
    run_details = {}
    for path in find_result_files(results_root):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = infer_model_dataset(results_root, path)
        y_true = data.get("True Labels", [])
        y_pred = data.get("Predicted Labels", [])
        run_key = (meta["model"], meta["dataset"], path)
        runs.append(
            {
                "file_path": path,
                "model": meta["model"],
                "dataset": meta["dataset"],
                "validation_accuracy": safe_float(data.get("Validation Accuracy")),
                "test_accuracy": safe_float(data.get("Test Accuracy")),
                "array_accuracy": compute_array_accuracy(y_true, y_pred),
                "num_examples": len(y_true) if isinstance(y_true, list) else 0,
                "run_key": run_key,
            }
        )
        run_details[run_key] = {
            "true_labels": y_true if isinstance(y_true, list) else [],
            "predicted_labels": y_pred if isinstance(y_pred, list) else [],
        }

    df = pd.DataFrame(runs)
    if not df.empty:
        df = df.sort_values(by=["dataset", "model"]).reset_index(drop=True)
    return df, run_details


def save_summary(df: pd.DataFrame, output_dir: str):
    csv_path = os.path.join(output_dir, "summary.csv")
    json_path = os.path.join(output_dir, "summary.json")
    summary_cols = [
        "model",
        "dataset",
        "validation_accuracy",
        "test_accuracy",
        "array_accuracy",
        "num_examples",
        "file_path",
    ]
    export_df = df[summary_cols].copy()
    export_df.to_csv(csv_path, index=False)
    export_df.to_json(json_path, orient="records", indent=2)


def plot_accuracy_bars(df: pd.DataFrame, output_dir: str, show: bool):
    labels = [f"{row.model} | {row.dataset}" for row in df.itertuples()]
    val_scores = df["validation_accuracy"].astype(float).to_numpy()
    test_scores = df["test_accuracy"].astype(float).to_numpy()

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(max(10, len(labels) * 1.2), 6))
    plt.bar(x - width / 2, val_scores, width=width, label="Validation Accuracy")
    plt.bar(x + width / 2, test_scores, width=width, label="Test Accuracy")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Validation/Test Accuracy by Model and Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_overview.png"), dpi=180)
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrices(
    df: pd.DataFrame,
    run_details: Dict[Tuple[str, str, str], Dict[str, List]],
    output_dir: str,
    show: bool,
):
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    for row in df.itertuples():
        details = run_details.get(row.run_key, {})
        y_true = details.get("true_labels", [])
        y_pred = details.get("predicted_labels", [])
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            continue

        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix: {row.model} | {row.dataset}")
        plt.tight_layout()
        filename = f"{row.model}__{row.dataset}.png".replace("/", "__")
        plt.savefig(os.path.join(cm_dir, filename), dpi=180)
        if show:
            plt.show()
        plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    runs_df, run_details = load_runs_df(args.results_root)
    if runs_df.empty:
        print(f"No results.json files found under: {args.results_root}")
        return

    save_summary(runs_df, args.output_dir)
    plot_accuracy_bars(runs_df, args.output_dir, args.show)
    plot_confusion_matrices(runs_df, run_details, args.output_dir, args.show)

    print(f"Processed {len(runs_df)} result files.")
    print(f"Saved outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
