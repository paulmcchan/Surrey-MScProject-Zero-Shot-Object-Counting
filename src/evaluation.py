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

    # NAE over images with gt > 0 (all FSC-147 images)
    nz = evaluation_df["gt_count"] > 0
    nae = (
        evaluation_df.loc[nz, "abs_error"]
        / evaluation_df.loc[nz, "gt_count"]
    ).mean()

    metrics = {
        "num_samples": len(evaluation_df),
        "mae": float(mae),
        "rmse": float(rmse),
        "mean_signed_error": float(mean_signed_error),
        "nae": float(nae),
        "num_undercount": int(num_undercount),
        "num_exact": int(num_exact),
        "num_overcount": int(num_overcount),
    }

    return evaluation_df, metrics


# ============================================================
# Multi-label evaluation (OmniCount-191)
# ============================================================
#
# Two protocols are provided and should be reported together:
#
# 1. STANDARD (Chattopadhyay et al., 2017) — evaluate_multi_label_counts
#    Every image is queried for every class of its domain vocabulary
#    (absent classes have GT 0, so false positives are penalised).
#    Per (domain, class):
#        RMSE_c    = sqrt(mean over the domain's images of err^2)
#        RMSE-nz_c = same, restricted to images with GT_c > 0
#    mRMSE    = mean of RMSE_c over classes
#    mRMSE-nz = mean of RMSE-nz_c over classes with >= 1 GT > 0
#
# 2. OMNICOUNT REFERENCE CODE — omnicount_reference_metrics
#    An exact port of OmniCount's released metrics/metric_multi.py.
#    It differs from the standard definition in two ways:
#      (a) only classes PRESENT in the GT of an image are evaluated
#          (predictions for absent classes are ignored);
#      (b) the sum of per-class RMSEs is divided by the number of
#          IMAGES, not the number of classes.
#    Needed only to compare against numbers reported in the
#    OmniCount paper; it is not a standard metric.
# ============================================================

def build_multi_label_pairs(results):
    """
    results: iterable of dicts with
        image_id, domain,
        categories     : GT dict {class: {"count": int, ...}}
        pred_counts    : {class: predicted count}
        query_classes  : classes to evaluate for this image
                         (normally the domain vocabulary)

    Returns one row per (image, queried class).
    Missing predictions count as 0; absent GT classes count as 0.
    """
    rows = []
    for r in results:
        gt = {c: v["count"] for c, v in r["categories"].items()}
        query = list(r["query_classes"])
        missing = set(gt) - set(query)
        if missing:
            raise ValueError(
                f"{r['image_id']}: GT classes not in query set: {missing}"
            )
        for c in query:
            g = float(gt.get(c, 0))
            p = float(r["pred_counts"].get(c, 0))
            rows.append({
                "image_id": r["image_id"],
                "domain": r["domain"],
                "class": c,
                "gt_count": g,
                "pred_count": p,
                "error": p - g,
            })
    return pd.DataFrame(rows)


def evaluate_multi_label_counts(pairs_df):
    """
    STANDARD multi-label metrics from build_multi_label_pairs output.

    Classes are keyed by (domain, class): the same name in two
    domains (e.g. 'apples') is evaluated separately.

    Returns
    -------
    per_class_df : per (domain, class) RMSE, RMSE-nz, counts
    per_domain_df: per-domain mRMSE / mRMSE-nz
    metrics      : overall dict
    """
    df = pairs_df.copy()
    df["sq"] = df["error"] ** 2
    df["abs"] = df["error"].abs()
    df["nz"] = df["gt_count"] > 0

    rows = []
    for (domain, cls), g in df.groupby(["domain", "class"], sort=True):
        nz = g[g["nz"]]
        rows.append({
            "domain": domain,
            "class": cls,
            "n_images": len(g),
            "n_nz": len(nz),
            "rmse": float(np.sqrt(g["sq"].mean())),
            "rmse_nz": float(np.sqrt(nz["sq"].mean())) if len(nz) else np.nan,
            "mae_nz": float(nz["abs"].mean()) if len(nz) else np.nan,
            "false_positive_units": float(g.loc[~g["nz"], "pred_count"].sum()),
        })
    per_class_df = pd.DataFrame(rows)

    per_domain_df = (
        per_class_df.groupby("domain")
        .agg(
            n_classes=("class", "size"),
            n_classes_nz=("n_nz", lambda s: int((s > 0).sum())),
            mRMSE=("rmse", "mean"),
            mRMSE_nz=("rmse_nz", "mean"),   # NaN classes skipped
        )
        .reset_index()
    )

    nz_pairs = df[df["nz"]]
    metrics = {
        "num_images": int(df["image_id"].nunique()),
        "num_pairs": int(len(df)),
        "num_nz_pairs": int(len(nz_pairs)),
        "num_classes": int(len(per_class_df)),
        "mRMSE": float(per_class_df["rmse"].mean()),
        "mRMSE_nz": float(per_class_df["rmse_nz"].mean()),
        # pooled, over present (image, class) pairs
        "MAE_nz": float(nz_pairs["abs"].mean()) if len(nz_pairs) else np.nan,
        "RMSE_nz": float(np.sqrt(nz_pairs["sq"].mean())) if len(nz_pairs) else np.nan,
        "NAE_nz": float((nz_pairs["abs"] / nz_pairs["gt_count"]).mean())
                  if len(nz_pairs) else np.nan,
        "false_positive_units": float(df.loc[~df["nz"], "pred_count"].sum()),
    }
    return per_class_df, per_domain_df, metrics


def omnicount_reference_metrics(ground_truth, predictions):
    """
    Exact port of OmniCount's metrics/metric_multi.py::calculate_metrics.

    ground_truth : {image_id: {class: gt_count}}   (classes present in GT)
    predictions  : {image_id: {class: pred_count}}

    Returns dict with m_rmse, m_rmse_nz, m_rel_rmse, m_rel_rmse_nz and
    the per-class dicts. Reproduces OmniCount's normalisation
    (sum over classes divided by number of images).
    """
    sum_sq_error, sum_rel_sq_error, non_zero_count = {}, {}, {}
    total_images = len(ground_truth)

    for counts in ground_truth.values():
        for class_name in counts:
            sum_sq_error[class_name] = 0
            sum_rel_sq_error[class_name] = 0
            non_zero_count[class_name] = 0

    for image_id, counts in ground_truth.items():
        pred_counts = predictions.get(image_id, {})
        for class_name, gt_count in counts.items():
            pred_count = pred_counts.get(class_name, 0)
            diff = abs(pred_count - gt_count)
            sum_sq_error[class_name] += diff ** 2
            sum_rel_sq_error[class_name] += (diff ** 2) / (gt_count + 1)
            if gt_count > 0:
                non_zero_count[class_name] += 1

    rmse = {c: np.sqrt(sum_sq_error[c] / total_images) for c in sum_sq_error}
    rmse_nz = {c: np.sqrt(sum_sq_error[c] / non_zero_count[c])
               for c in sum_sq_error if non_zero_count[c] > 0}
    rel_rmse = {c: np.sqrt(sum_rel_sq_error[c] / total_images) for c in sum_rel_sq_error}
    rel_rmse_nz = {c: np.sqrt(sum_rel_sq_error[c] / non_zero_count[c])
                   for c in sum_rel_sq_error if non_zero_count[c] > 0}

    return {
        "total_images": total_images,
        "m_rmse": float(sum(rmse.values()) / total_images),
        "m_rmse_nz": float(sum(rmse_nz.values()) / total_images),
        "m_rel_rmse": float(sum(rel_rmse.values()) / total_images),
        "m_rel_rmse_nz": float(sum(rel_rmse_nz.values()) / total_images),
        "rmse": rmse,
        "rmse_nz": rmse_nz,
    }


def omnicount_reference_by_domain(results):
    """
    Apply omnicount_reference_metrics per domain (OmniCount evaluates
    one GT/prediction file per domain) and over all images pooled.

    results: same structure as build_multi_label_pairs input.
    """
    by_domain = {}
    for r in results:
        d = by_domain.setdefault(r["domain"], ({}, {}))
        d[0][r["image_id"]] = {c: v["count"] for c, v in r["categories"].items()}
        d[1][r["image_id"]] = dict(r["pred_counts"])

    rows = []
    all_gt, all_pred = {}, {}
    for domain, (gt, pred) in sorted(by_domain.items()):
        m = omnicount_reference_metrics(gt, pred)
        rows.append({"domain": domain, "n_images": m["total_images"],
                     "m_rmse": m["m_rmse"], "m_rmse_nz": m["m_rmse_nz"]})
        all_gt.update(gt)
        all_pred.update(pred)

    pooled = omnicount_reference_metrics(all_gt, all_pred)
    per_domain_df = pd.DataFrame(rows)
    return per_domain_df, {
        "pooled_m_rmse": pooled["m_rmse"],
        "pooled_m_rmse_nz": pooled["m_rmse_nz"],
        "mean_over_domains_m_rmse": float(per_domain_df["m_rmse"].mean()),
        "mean_over_domains_m_rmse_nz": float(per_domain_df["m_rmse_nz"].mean()),
    }
