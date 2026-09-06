"""测试 §6 下游任务增益统计: compute_gain_stats, per_haze_level_report, paired_t_test."""

from __future__ import annotations

import pytest
import numpy as np

from icewave.eval.downstream import (
    compute_gain_stats,
    per_haze_level_report,
    paired_t_test,
)


# ---------------------------------------------------------------------------
# compute_gain_stats
# ---------------------------------------------------------------------------
class TestComputeGainStats:
    def _make_result(self, hazy=0.412, dehazed=0.483, clear=0.602):
        return {
            "hazy": {"mAP": hazy, "per_image": [hazy - 0.01 * i for i in range(10)]},
            "dehazed": {"mAP": dehazed, "per_image": [dehazed - 0.01 * i for i in range(10)]},
            "clear": {"mAP": clear, "per_image": [clear - 0.01 * i for i in range(10)]},
            "delta_mAP_dehazed_minus_hazy": dehazed - hazy,
            "gap_clear_minus_dehazed": clear - dehazed,
        }

    def test_basic_stats(self):
        result = self._make_result()
        stats = compute_gain_stats(result)
        assert stats["delta_gain"] == pytest.approx(0.071, abs=1e-4)
        assert stats["delta_gap"] == pytest.approx(0.119, abs=1e-4)
        expected_R = 0.071 / (0.602 - 0.412)
        assert stats["R"] == pytest.approx(expected_R, abs=0.01)

    def test_no_clear_field(self):
        result = {
            "hazy": {"mAP": 0.4, "per_image": [0.4]},
            "dehazed": {"mAP": 0.5, "per_image": [0.5]},
        }
        stats = compute_gain_stats(result)
        assert stats["delta_gain"] == pytest.approx(0.1)
        assert stats["R"] is None
        assert stats["delta_gap"] is None

    def test_ci_values(self):
        result = self._make_result()
        stats = compute_gain_stats(result, n_bootstrap=100)
        assert stats["delta_gain_ci_low"] is not None
        assert stats["delta_gain_ci_high"] is not None
        assert stats["delta_gain_ci_low"] <= stats["delta_gain"]
        assert stats["delta_gain_ci_high"] >= stats["delta_gain"]

    def test_R_one_when_perfect_recovery(self):
        """完美恢复: dehazed == clear → R == 1"""
        result = self._make_result(hazy=0.4, dehazed=0.6, clear=0.6)
        stats = compute_gain_stats(result)
        assert stats["R"] == pytest.approx(1.0, abs=0.01)

    def test_R_negative_when_dehaze_hurts(self):
        """去雾更差: dehazed < hazy → R < 0"""
        result = self._make_result(hazy=0.4, dehazed=0.35, clear=0.6)
        stats = compute_gain_stats(result)
        assert stats["R"] < 0

    def test_paper_example(self):
        """论文表格样例数值验证."""
        # Cascade baseline
        result = self._make_result(hazy=0.412, dehazed=0.483, clear=0.602)
        stats = compute_gain_stats(result)
        assert stats["delta_gain"] == pytest.approx(0.071, abs=1e-4)
        R_cascade = 0.071 / (0.602 - 0.412)
        assert stats["R"] == pytest.approx(R_cascade, abs=0.01)
        # Joint (本文)
        result_j = self._make_result(hazy=0.412, dehazed=0.531, clear=0.602)
        stats_j = compute_gain_stats(result_j)
        assert stats_j["delta_gain"] == pytest.approx(0.119, abs=1e-4)
        R_joint = 0.119 / (0.602 - 0.412)
        assert stats_j["R"] == pytest.approx(R_joint, abs=0.01)


# ---------------------------------------------------------------------------
# per_haze_level_report
# ---------------------------------------------------------------------------
class TestPerHazeLevelReport:
    def test_grouping(self):
        per_img = [
            {"mAP_hazy": 0.3, "mAP_dehazed": 0.5, "mAP_clear": 0.6},
            {"mAP_hazy": 0.35, "mAP_dehazed": 0.55, "mAP_clear": 0.65},
            {"mAP_hazy": 0.2, "mAP_dehazed": 0.3, "mAP_clear": 0.5},
            {"mAP_hazy": 0.15, "mAP_dehazed": 0.2, "mAP_clear": 0.4},
        ]
        levels = ["thin", "thin", "medium", "dense"]
        report = per_haze_level_report(per_img, levels)
        assert "thin" in report
        assert "medium" in report
        assert "dense" in report
        assert report["thin"]["n_images"] == 2
        assert report["medium"]["n_images"] == 1
        assert report["dense"]["n_images"] == 1
        # thin gain > dense gain (期望)
        assert report["thin"]["delta_gain"] > report["dense"]["delta_gain"]

    def test_empty_input(self):
        report = per_haze_level_report([], [])
        assert report == {}


# ---------------------------------------------------------------------------
# paired_t_test
# ---------------------------------------------------------------------------
class TestPairedTTest:
    def test_significant_difference(self):
        """有显著差异 → p < 0.05."""
        pytest.importorskip("scipy")
        a = [0.5, 0.55, 0.6, 0.58, 0.52]
        b = [0.3, 0.35, 0.4, 0.38, 0.32]
        result = paired_t_test(a, b)
        assert result["n"] == 5
        assert result["mean_diff"] > 0
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_no_difference(self):
        """无显著差异 → p >= 0.05 (相同数据时 p=1.0)."""
        pytest.importorskip("scipy")
        a = [0.5, 0.55, 0.6, 0.52, 0.58]
        b = [0.5, 0.55, 0.6, 0.52, 0.58]
        result = paired_t_test(a, b)
        assert result["mean_diff"] == pytest.approx(0.0, abs=1e-8)
        assert result["p_value"] >= 0.05
        assert result["significant"] is False

    def test_too_few_samples(self):
        """样本太少 → 不显著."""
        a = [0.5]
        b = [0.3]
        result = paired_t_test(a, b)
        assert result["n"] == 1
        assert result["significant"] is False
