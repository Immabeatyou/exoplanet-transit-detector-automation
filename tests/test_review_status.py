import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web.pipeline import classify_review_status


def test_review_now_uses_configured_threshold():
    status, _ = classify_review_status(
        final_ranking_score=80.0,
        period_stability_flag="stable",
        quality_flag="high_confidence",
        review_threshold=70.0,
        review_now_threshold=80.0,
    )

    assert status == "review_now"


def test_review_now_requires_supporting_signals():
    status, _ = classify_review_status(
        final_ranking_score=90.0,
        period_stability_flag="unstable",
        quality_flag="high_confidence",
        review_threshold=70.0,
        review_now_threshold=80.0,
    )

    assert status == "review_with_caution"


def test_score_below_review_threshold_is_low_priority():
    status, _ = classify_review_status(
        final_ranking_score=69.9,
        period_stability_flag="stable",
        quality_flag="high_confidence",
        review_threshold=70.0,
        review_now_threshold=80.0,
    )

    assert status == "low_priority"