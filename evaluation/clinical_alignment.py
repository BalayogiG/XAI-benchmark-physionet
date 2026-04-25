"""
evaluation/clinical_alignment.py
---------------------------------
Measure how well XAI attributions align with clinically annotated
ECG features: P-wave, QRS complex, ST segment, T-wave, RR intervals.

Clinical features are annotated as (start_sample, end_sample) tuples
by expert cardiologists. We compute:

  Intersection-over-Union (IoU) between the high-attribution temporal
  region (top 20% by |attribution|) and each annotated feature window.

Format of annotation CSV:
  record_id, feature_name, start_sample, end_sample
  A00001,    P_wave,       85,           115
  A00001,    QRS,          130,          160
  ...

Usage:
    scorer = ClinicalAlignmentScorer(annotation_csv)
    scores = scorer.score(record_id, attribution)
    # {'P_wave': 0.72, 'QRS': 0.81, 'ST': 0.31, 'T_wave': 0.27, 'RR': 0.65}
"""

import csv
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


CLINICAL_FEATURES = ["P_wave", "QRS", "ST_segment", "T_wave", "RR_interval"]


def _iou_1d(
    pred_mask: np.ndarray,
    gt_start:  int,
    gt_end:    int,
) -> float:
    """
    Intersection-over-Union for a binary 1D prediction mask and a
    ground-truth interval [gt_start, gt_end].

    Parameters
    ----------
    pred_mask : binary array (T,)
    gt_start  : inclusive start sample of GT annotation
    gt_end    : exclusive end sample of GT annotation

    Returns
    -------
    IoU in [0, 1]
    """
    T     = len(pred_mask)
    gt    = np.zeros(T, dtype=bool)
    gt[gt_start:gt_end] = True

    pred = pred_mask.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()

    return float(inter / union) if union > 0 else 0.0


class ClinicalAlignmentScorer:
    """
    Parameters
    ----------
    annotation_csv : path to CSV with columns:
                     record_id, feature_name, start_sample, end_sample
    top_percentile : fraction of timesteps selected as high-attribution
    """

    def __init__(
        self,
        annotation_csv: str,
        top_percentile: float = 0.20,
    ) -> None:
        self.top_p = top_percentile
        self._annotations: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
        self._load(annotation_csv)

    def _load(self, csv_path: str) -> None:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec  = row["record_id"].strip()
                feat = row["feature_name"].strip()
                s    = int(row["start_sample"])
                e    = int(row["end_sample"])
                self._annotations[rec].append((feat, s, e))

    def score(
        self,
        record_id:   str,
        attribution: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute per-feature IoU for a single record.

        Returns
        -------
        dict: feature_name → IoU score (mean over all annotated instances)
        """
        anns = self._annotations.get(record_id, [])
        if not anns:
            return {f: np.nan for f in CLINICAL_FEATURES}

        # Build high-attribution binary mask
        threshold     = np.percentile(np.abs(attribution), 100 * (1 - self.top_p))
        high_attr_mask = np.abs(attribution) >= threshold

        # Per-feature IoU
        feature_ious: Dict[str, List[float]] = defaultdict(list)
        for (feat, start, end) in anns:
            iou = _iou_1d(high_attr_mask, start, end)
            feature_ious[feat].append(iou)

        return {
            feat: float(np.mean(ious)) if ious else np.nan
            for feat, ious in feature_ious.items()
        }

    def score_dataset(
        self,
        records:      List[str],
        attributions: Dict[str, np.ndarray],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Aggregate IoU across multiple records.

        Parameters
        ----------
        records      : list of record IDs
        attributions : {record_id: attribution array}

        Returns
        -------
        {feature_name: (mean_IoU, std_IoU)}
        """
        all_scores: Dict[str, List[float]] = defaultdict(list)
        for rec in records:
            if rec not in attributions:
                continue
            per_feature = self.score(rec, attributions[rec])
            for feat, iou in per_feature.items():
                if not np.isnan(iou):
                    all_scores[feat].append(iou)

        return {
            feat: (float(np.mean(v)), float(np.std(v)))
            for feat, v in all_scores.items()
        }


# ── Synthetic annotation generator (for testing without real annotations) ─────

def generate_synthetic_annotations(
    output_csv:  str,
    record_ids:  List[str],
    fs:          int = 300,
    seed:        int = 42,
) -> None:
    """
    Write a synthetic annotation CSV with plausible ECG feature timings
    for unit testing and dry runs.

    Typical durations @ 300 Hz:
      P-wave: 40–100 ms  → 12–30 samples
      PR interval: 120–200 ms
      QRS: 80–120 ms     → 24–36 samples
      ST: 80–120 ms
      T-wave: 160–320 ms → 48–96 samples
    """
    rng = np.random.default_rng(seed)
    rows = []

    for rec in record_ids:
        # Simulate 8 beats per 10-second window (average HR 48 bpm — conservative)
        beat_starts = np.arange(0, 2900, 375)  # ~375 samples apart

        for bs in beat_starts:
            # P-wave
            p_start = int(bs)
            p_end   = p_start + int(rng.integers(12, 30))
            rows.append((rec, "P_wave", p_start, p_end))

            # QRS
            q_start = p_end + int(rng.integers(36, 60))
            q_end   = q_start + int(rng.integers(24, 36))
            rows.append((rec, "QRS", q_start, q_end))

            # ST segment
            st_start = q_end
            st_end   = st_start + int(rng.integers(24, 36))
            rows.append((rec, "ST_segment", st_start, st_end))

            # T-wave
            t_start  = st_end
            t_end    = min(t_start + int(rng.integers(48, 96)), 3000)
            rows.append((rec, "T_wave", t_start, t_end))

            # RR interval (distance to next beat)
            rr_start = q_start
            rr_end   = min(rr_start + 375, 3000)
            rows.append((rec, "RR_interval", rr_start, rr_end))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "feature_name", "start_sample", "end_sample"])
        writer.writerows(rows)

    print(f"[INFO] Synthetic annotations written to {output_csv} ({len(rows)} entries).")
