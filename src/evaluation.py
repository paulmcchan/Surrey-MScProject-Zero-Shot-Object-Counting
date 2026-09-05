# -*- coding: utf-8 -*-
"""
Shared counting evaluation utilities for the
MSc Zero-Shot Object Counting project.
"""

import numpy as np
import pandas as pd


def evaluate_single_label_counts(results):
    """
    Evaluate single-label counting predictions.

    Expected result structure:
        result["image_id"]
        result["categories"]
        result["pred_count"]

    Assumes one target category per image, as in FSC-147.

    Returns
    -------
    evaluation_df : pandas.DataFrame
        Per-image counting results.

    metrics : dict
        Aggregate counting metrics and error-direction statistics.
    """

    evaluation_rows = []

    for r in results:

        # Single target category per image
        category_name = next(iter(r["categories"]))

        gt_count = r["categories"][category_name]["count"]
        pred_count = r["pred_count"]

        error = pred_count - gt_count
        abs_error = abs(error)
        squared_error = error ** 2

        evaluation_rows.append({
            "image_id": r["image_id"],
            "category": category_name,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "error": error,
            "abs_error": abs_error,
            "squared_error": squared_error,
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    mae = evaluation_df["abs_error"].mean()
    rmse = np.sqrt(
        evaluation_df["squared_error"].mean()
    )

    num_undercount = (
        evaluation_df["error"] < 0
    ).sum()

    num_exact = (
        evaluation_df["error"] == 0
    ).sum()

    num_overcount = (
        evaluation_df["error"] > 0
    ).sum()

    mean_signed_error = (
        evaluation_df["error"].mean()
    )

    metrics = {
        "num_samples": len(evaluation_df),
        "mae": float(mae),
        "rmse": float(rmse),
        "mean_signed_error": float(mean_signed_error),
        "num_undercount": int(num_undercount),
        "num_exact": int(num_exact),
        "num_overcount": int(num_overcount),
    }

    return evaluation_df, metrics