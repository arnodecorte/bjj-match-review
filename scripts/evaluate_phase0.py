"""
Evaluate Phase 0 classifier quality on a real match clip by comparing:
1) heuristic-only protocol
2) ML model + temporal smoothing engine

This script does not require frame-level ground truth. It reports proxy metrics
that matter for review quality: confidence, transition churn, and stability.

Usage:

    python scripts/evaluate_phase0.py \
        --video /path/to/match.mp4 \
        --model models/classifier.pt \
        --fps 2.0 \
        --output reports/phase0_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifier.inference import PositionInference


def _transition_count(labels: list[str]) -> int:
    if not labels:
        return 0
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def _avg_segment_len(labels: list[str]) -> float:
    if not labels:
        return 0.0
    segments = []
    current = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            current += 1
        else:
            segments.append(current)
            current = 1
    segments.append(current)
    return float(sum(segments) / len(segments))


def summarise(results: list[dict]) -> dict:
    labels = [r["position"] for r in results]
    confidences = [float(r.get("confidence", 0.0)) for r in results]
    transitions = _transition_count(labels)
    n = max(len(labels), 1)
    return {
        "frames": len(labels),
        "mean_confidence": round(sum(confidences) / n, 4),
        "high_confidence_rate": round(sum(c >= 0.6 for c in confidences) / n, 4),
        "transition_count": transitions,
        "transition_rate": round(transitions / n, 4),
        "avg_segment_length": round(_avg_segment_len(labels), 3),
    }


def compare(heuristic: list[dict], ml: list[dict]) -> dict:
    n = min(len(heuristic), len(ml))
    if n == 0:
        return {
            "frame_overlap": 0,
            "agreement_rate": 0.0,
        }

    agree = sum(
        1
        for h, m in zip(heuristic[:n], ml[:n])
        if h["position"] == m["position"]
    )
    return {
        "frame_overlap": n,
        "agreement_rate": round(agree / n, 4),
    }


def verdict(h: dict, m: dict) -> dict:
    checks = {
        "confidence_up": m["mean_confidence"] > h["mean_confidence"],
        "high_conf_rate_up": m["high_confidence_rate"] > h["high_confidence_rate"],
        "transition_rate_down": m["transition_rate"] < h["transition_rate"],
        "segment_length_up": m["avg_segment_length"] > h["avg_segment_length"],
    }
    passed = sum(bool(v) for v in checks.values())
    return {
        "checks": checks,
        "pass_count": passed,
        "improved": passed >= 3,
        "note": (
            "ML+engine improves stability/confidence proxy metrics."
            if passed >= 3
            else "No clear proxy-metric improvement; inspect qualitative timeline output."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 heuristic-vs-ML evaluator")
    parser.add_argument("--video", type=Path, required=True, help="Path to match video")
    parser.add_argument("--model", type=Path, default=Path("models/classifier.pt"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("reports/phase0_eval.json"))
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    heuristic = PositionInference(model_path=None)
    ml_engine = PositionInference(model_path=str(args.model))

    heuristic_results = heuristic.process_video(str(args.video), fps=args.fps)
    ml_results = ml_engine.process_video(str(args.video), fps=args.fps)

    h_summary = summarise(heuristic_results)
    m_summary = summarise(ml_results)

    report = {
        "video": str(args.video),
        "fps": args.fps,
        "heuristic": h_summary,
        "ml_engine": m_summary,
        "comparison": compare(heuristic_results, ml_results),
        "verdict": verdict(h_summary, m_summary),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to: {args.output}")


if __name__ == "__main__":
    main()
