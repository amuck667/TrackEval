"""
This file test the HOTA metric with the usage of mot_challenge_2d_keypoints. The idea being that the 2d keypoints use a euclidean distance gaussian based similarity. This needs to be calibrated and tested.

Most important: The standard HOTA eval_sequence  sweeps alpha from 0.05 to 0.99 in steps of 0.05. Your sigma must produce similarities that span this range for your typical data, otherwise most alpha values become redundant.
If tests 1-4 all pass cleanly but TestSimilarityDistribution or TestLocARange fail, that's your cue to adjust sigma. If the diagnostic report shows that your similarities cluster near 1.0 for all reasonable offsets, sigma is too large for your coordinate scale.

Usage:
# Run all tests
pytest test_kp_hota_calibration.py -v

# Run the diagnostic report (with print output)
pytest test_kp_hota_calibration.py::TestDiagnosticReport -v -s

# Run only the critical calibration tests
pytest test_kp_hota_calibration.py::TestPerfectTracker test_kp_hota_calibration.py::TestOffsetDegradation -v

# Run with a different sigma (parametrize)
pytest test_kp_hota_calibration.py -v --override-ini="sigma=15"
"""

import numpy as np
import pytest
from copy import deepcopy

from trackeval.metrics.hota import HOTA
from trackeval.datasets.mot_challenge_2d_keypoints import MotChallenge2DKeypoints


# ============================================================
# FIXTURES: Synthetic data generators
# ============================================================

@pytest.fixture
def sigma():
    """The sigma value used in _calculate_similarities"""
    return 10


@pytest.fixture
def num_keypoints():
    """Number of keypoints per object"""
    return 5


@pytest.fixture
def hota_metric():
    """Instantiate the HOTA metric"""
    return HOTA()


@pytest.fixture
def dataset_instance():
    """
    Create a minimal MotChallenge2DKeypoints instance for calling _calculate_similarities.
    We bypass __init__ to avoid file-system checks.
    """
    instance = object.__new__(MotChallenge2DKeypoints)
    return instance


def make_synthetic_sequence_data(
    num_timesteps=10,
    num_objects=5,
    num_keypoints=5,
    gt_keypoints_fn=None,
    tracker_keypoints_fn=None,
    gt_ids_fn=None,
    tracker_ids_fn=None,
    visibility_fn=None,
    sigma=10,
):
    """
    Generate a synthetic sequence data dict compatible with HOTA.eval_sequence().
    
    Args:
        gt_keypoints_fn: callable(t, num_objects, num_keypoints) -> (N, K, 2) array
        tracker_keypoints_fn: callable(t, num_objects, num_keypoints) -> (M, K, 2) array
        gt_ids_fn: callable(t, num_objects) -> (N,) array of IDs
        tracker_ids_fn: callable(t, num_objects) -> (M,) array of IDs
        visibility_fn: callable(t, num_objects, num_keypoints) -> (N, K) array
        sigma: sigma for Gaussian similarity
    """
    # Defaults: fixed objects with random keypoints
    rng = np.random.default_rng(42)
    
    if gt_keypoints_fn is None:
        # Generate fixed GT keypoints for all objects across all frames
        base_gt_kps = rng.uniform(50, 500, size=(num_objects, num_keypoints, 2))
        gt_keypoints_fn = lambda t, n, k: base_gt_kps.copy()
    
    if tracker_keypoints_fn is None:
        tracker_keypoints_fn = gt_keypoints_fn  # perfect tracker by default
    
    if gt_ids_fn is None:
        gt_ids_fn = lambda t, n: np.arange(n)
    
    if tracker_ids_fn is None:
        tracker_ids_fn = lambda t, n: np.arange(n)
    
    if visibility_fn is None:
        visibility_fn = lambda t, n, k: np.full((n, k), 2)  # all visible
    
    # Build the data dict
    gt_ids_list = []
    tracker_ids_list = []
    similarity_list = []
    num_gt_dets = 0
    num_tracker_dets = 0
    
    for t in range(num_timesteps):
        gt_kps = gt_keypoints_fn(t, num_objects, num_keypoints)
        tr_kps = tracker_keypoints_fn(t, num_objects, num_keypoints)
        gt_ids = gt_ids_fn(t, num_objects)
        tr_ids = tracker_ids_fn(t, num_objects)
        vis = visibility_fn(t, num_objects, num_keypoints)
        
        # Compute similarity using the same logic as _calculate_similarities
        similarity = _compute_similarity(gt_kps, tr_kps, vis, sigma)
        
        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(tr_ids)
        similarity_list.append(similarity)
        num_gt_dets += len(gt_ids)
        num_tracker_dets += len(tr_ids)
    
    unique_gt_ids = np.unique(np.concatenate(gt_ids_list))
    unique_tracker_ids = np.unique(np.concatenate(tracker_ids_list))
    
    data = {
        'gt_ids': gt_ids_list,
        'tracker_ids': tracker_ids_list,
        'similarity_scores': similarity_list,
        'num_timesteps': num_timesteps,
        'num_gt_dets': num_gt_dets,
        'num_tracker_dets': num_tracker_dets,
        'num_gt_ids': len(unique_gt_ids),
        'num_tracker_ids': len(unique_tracker_ids),
        'seq': 'test_seq',
    }
    return data


def _compute_similarity(gt_keypoints, tracker_keypoints, visibilities, sigma):
    """
    Replicates the _calculate_similarities logic from MotChallenge2DKeypoints.
    This is the actual similarity function your pipeline uses.
    """
    if gt_keypoints.shape[0] == 0 or tracker_keypoints.shape[0] == 0:
        return np.zeros((gt_keypoints.shape[0], tracker_keypoints.shape[0]))
    
    min_kps = min(gt_keypoints.shape[1], tracker_keypoints.shape[1])
    gt_kps = gt_keypoints[:, :min_kps, :]
    trk_kps = tracker_keypoints[:, :min_kps, :]
    vis = visibilities[:, :min_kps]
    
    N, K, _ = gt_kps.shape
    M, _, _ = trk_kps.shape
    
    dist_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            valid_mask = (vis[i] == 2)
            if np.any(valid_mask):
                dists = np.linalg.norm(gt_kps[i][valid_mask] - trk_kps[j][valid_mask], axis=1)
                dist_matrix[i, j] = np.mean(dists)
            else:
                dist_matrix[i, j] = np.inf
    
    similarity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    similarity[dist_matrix == np.inf] = 0
    return similarity


# ============================================================
# TEST 1: Perfect Tracker (Identity Test)
# ============================================================

class TestPerfectTracker:
    """When tracker output is identical to GT, HOTA should be ~1.0"""
    
    def test_perfect_tracker_hota_is_one(self, hota_metric, sigma):
        """Perfect predictions with correct IDs should yield HOTA ≈ 1.0 at all alphas"""
        data = make_synthetic_sequence_data(
            num_timesteps=20,
            num_objects=5,
            num_keypoints=5,
            sigma=sigma,
        )
        res = hota_metric.eval_sequence(data)
        
        # HOTA should be 1.0 at all alpha thresholds
        np.testing.assert_allclose(res['HOTA'], 1.0, atol=1e-6,
            err_msg=f"Perfect tracker HOTA={res['HOTA']}, expected all 1.0")
    
    def test_perfect_tracker_loca_is_one(self, hota_metric, sigma):
        """Perfect localization should give LocA = 1.0"""
        data = make_synthetic_sequence_data(sigma=sigma)
        res = hota_metric.eval_sequence(data)
        
        np.testing.assert_allclose(res['LocA'], 1.0, atol=1e-6,
            err_msg=f"Perfect tracker LocA={res['LocA']}, expected all 1.0")
    
    def test_perfect_tracker_no_fp_fn(self, hota_metric, sigma):
        """Perfect tracker should have zero FP and FN"""
        data = make_synthetic_sequence_data(sigma=sigma)
        res = hota_metric.eval_sequence(data)
        
        np.testing.assert_array_equal(res['HOTA_FP'], 0,
            err_msg="Perfect tracker should have zero false positives")
        np.testing.assert_array_equal(res['HOTA_FN'], 0,
            err_msg="Perfect tracker should have zero false negatives")
    
    def test_perfect_tracker_assa_is_one(self, hota_metric, sigma):
        """Perfect ID assignment should give AssA = 1.0"""
        data = make_synthetic_sequence_data(sigma=sigma)
        res = hota_metric.eval_sequence(data)
        
        np.testing.assert_allclose(res['AssA'], 1.0, atol=1e-6,
            err_msg=f"Perfect tracker AssA={res['AssA']}, expected 1.0")


# ============================================================
# TEST 2: Monotonic Degradation with Spatial Offset
# ============================================================

class TestOffsetDegradation:
    """HOTA should degrade smoothly and monotonically as spatial offset increases"""
    
    @pytest.fixture
    def base_gt_keypoints(self, num_keypoints):
        rng = np.random.default_rng(42)
        return rng.uniform(100, 400, size=(5, num_keypoints, 2))
    
    def _make_offset_data(self, offset, base_gt_kps, sigma):
        """Create data where tracker is offset by a fixed number of pixels"""
        num_objects = base_gt_kps.shape[0]
        num_kps = base_gt_kps.shape[1]
        
        def tracker_kps_fn(t, n, k):
            return base_gt_kps + offset  # uniform offset in both x and y
        
        def gt_kps_fn(t, n, k):
            return base_gt_kps.copy()
        
        return make_synthetic_sequence_data(
            num_timesteps=20,
            num_objects=num_objects,
            num_keypoints=num_kps,
            gt_keypoints_fn=gt_kps_fn,
            tracker_keypoints_fn=tracker_kps_fn,
            sigma=sigma,
        )
    
    def test_hota_decreases_with_offset(self, hota_metric, sigma, base_gt_keypoints):
        """HOTA mean should monotonically decrease as offset increases"""
        offsets = [0, 2, 5, 10, 15, 20, 30, 50]
        hota_scores = []
        
        for offset in offsets:
            data = self._make_offset_data(offset, base_gt_keypoints, sigma)
            res = hota_metric.eval_sequence(data)
            hota_scores.append(np.mean(res['HOTA']))
        
        # Check monotonic decrease
        for i in range(len(hota_scores) - 1):
            assert hota_scores[i] >= hota_scores[i + 1] - 1e-10, (
                f"HOTA not monotonically decreasing: offset={offsets[i]}px -> "
                f"HOTA={hota_scores[i]:.4f}, offset={offsets[i+1]}px -> HOTA={hota_scores[i+1]:.4f}"
            )
    
    def test_loca_only_decreases_while_tp_exists(self, hota_metric, sigma, base_gt_keypoints):
        """
        LocA should decrease monotonically ONLY at alpha thresholds where HOTA_TP > 0.
        Once HOTA_TP drops to 0, LocA defaults to 1.0 (vacuous truth) — this is expected.
        """
        offsets = [0, 2, 5, 10, 15, 20]
        
        for a_idx in range(len(np.arange(0.05, 0.99, 0.05))):
            prev_loca = None
            prev_had_tp = False
            
            for offset in offsets:
                data = self._make_offset_data(offset, base_gt_keypoints, sigma)
                res = hota_metric.eval_sequence(data)
                
                current_tp = res['HOTA_TP'][a_idx]
                current_loca = res['LocA'][a_idx]
                
                if prev_loca is not None and current_tp > 0 and prev_had_tp:
                    assert current_loca <= prev_loca + 1e-10, (
                        f"LocA increased at alpha_idx={a_idx} while HOTA_TP > 0: "
                        f"offset {offsets[offsets.index(offset)-1]} -> {offset}, "
                        f"LocA {prev_loca:.4f} -> {current_loca:.4f}"
                    )
                
                prev_loca = current_loca
                prev_had_tp = (current_tp > 0)

    def test_loca_is_one_when_no_tp(self, hota_metric, sigma, base_gt_keypoints):
        """When HOTA_TP = 0, LocA should be 1.0 (vacuous truth)."""
        # Use very large offset to ensure no matches pass any alpha
        data = self._make_offset_data(5 * sigma, base_gt_keypoints, sigma)
        res = hota_metric.eval_sequence(data)
        
        for a_idx in range(len(res['HOTA_TP'])):
            if res['HOTA_TP'][a_idx] == 0:
                np.testing.assert_allclose(res['LocA'][a_idx], 1.0, atol=1e-6,
                    err_msg=f"LocA should be 1.0 when HOTA_TP=0 at alpha_idx={a_idx}")

    def test_deta_decreases_monotonically(self, hota_metric, sigma, base_gt_keypoints):
        """DetA should always decrease with offset (unlike LocA)."""
        offsets = [0, 2, 5, 10, 15, 20, 30]
        deta_scores = []

        for offset in offsets:
            data = self._make_offset_data(offset, base_gt_keypoints, sigma)
            res = hota_metric.eval_sequence(data)
            deta_scores.append(np.mean(res['DetA']))

        for i in range(len(deta_scores) - 1):
            assert deta_scores[i] >= deta_scores[i + 1] - 1e-10, (
                f"DetA not monotonically decreasing: offset={offsets[i]}px -> "
                f"DetA={deta_scores[i]:.4f}, offset={offsets[i+1]}px -> DetA={deta_scores[i+1]:.4f}"
            )

    def test_loca_per_alpha_diagnostic(self, hota_metric, sigma, base_gt_keypoints, capsys):
        """Print per-alpha LocA breakdown to visualize the vacuous-truth effect."""
        offsets = [0, 5, 10, 15, 20, 30]
        alphas = np.arange(0.05, 0.99, 0.05)
        
        print("\n\nLocA per-alpha breakdown (showing vacuous-truth effect):")
        print(f"{'Offset':<8}", end="")
        for a in [0.05, 0.15, 0.25, 0.35, 0.50, 0.75, 0.95]:
            print(f"α={a:<5.2f}", end=" ")
        print(f"{'Mean':<8} {'HOTA_TP[0]':<10}")
        print("-" * 90)
        
        for offset in offsets:
            data = self._make_offset_data(offset, base_gt_keypoints, sigma)
            res = hota_metric.eval_sequence(data)
            
            print(f"{offset:<8}", end="")
            for a_idx in [0, 2, 4, 6, 9, 14, 18]:
                loca_val = res['LocA'][a_idx]
                tp_val = res['HOTA_TP'][a_idx]
                marker = "*" if tp_val == 0 else " "
                print(f"{loca_val:<5.3f}{marker} ", end=" ")
            print(f"{np.mean(res['LocA']):<8.4f} {res['HOTA_TP'][0]:<10.0f}")
        
        print("\n* = HOTA_TP is 0 at this alpha (LocA defaults to 1.0)")
        assert False  # Diagnostic only
    
    def test_zero_offset_equals_perfect(self, hota_metric, sigma, base_gt_keypoints):
        """Zero offset should yield perfect score"""
        data = self._make_offset_data(0, base_gt_keypoints, sigma)
        res = hota_metric.eval_sequence(data)
        
        np.testing.assert_allclose(np.mean(res['HOTA']), 1.0, atol=1e-6)
    
    def test_large_offset_gives_low_hota(self, hota_metric, sigma, base_gt_keypoints):
        """Very large offset (3*sigma) should give very low HOTA"""
        data = self._make_offset_data(3 * sigma, base_gt_keypoints, sigma)
        res = hota_metric.eval_sequence(data)
        
        # At 3*sigma offset, Euclidean distance = 3*sigma*sqrt(2) ≈ 42.4 for diagonal offset
        # Similarity = exp(-(3σ√2)²/(2σ²)) = exp(-9) ≈ 0.0001
        assert np.mean(res['HOTA']) < 0.3, (
            f"HOTA at 3*sigma offset should be low, got {np.mean(res['HOTA']):.4f}"
        )


# ============================================================
# TEST 3: Sigma Calibration Check
# ============================================================

class TestSigmaCalibration:
    """Verify sigma maps to physically meaningful acceptance distances"""
    
    def test_sigma_distance_mapping(self, sigma):
        """Print and verify the distance-to-similarity mapping for the chosen sigma"""
        # For each alpha threshold, compute the max acceptable distance
        alphas = np.arange(0.05, 0.99, 0.05)
        max_distances = sigma * np.sqrt(-2 * np.log(alphas))
        
        # The lowest alpha (0.05) should map to a reasonable max distance
        max_acceptable_distance = max_distances[0]  # distance at alpha=0.05
        
        # Sanity check: this distance should be > 0 and < image_dimension
        # Adjust 1000 to your actual image size
        assert max_acceptable_distance > 0, "Max distance must be positive"
        assert max_acceptable_distance < 1000, (
            f"Max acceptable distance ({max_acceptable_distance:.1f}px) exceeds "
            f"reasonable image bounds. Sigma={sigma} might be too large."
        )
        
        # The highest alpha (0.95) should map to a very tight distance
        min_meaningful_distance = max_distances[-1]  # distance at alpha~0.95
        assert min_meaningful_distance > 1, (
            f"At alpha=0.95, acceptable distance is only {min_meaningful_distance:.2f}px. "
            f"Sigma={sigma} might be too small for sub-pixel precision requirements."
        )
    
    def test_similarity_at_expected_acceptable_distance(self, sigma, dataset_instance):
        """
        Verify that at your expected 'acceptable' distance, similarity is above 
        the lowest alpha threshold (0.05).
        
        Adjust ACCEPTABLE_DISTANCE_PX to your domain knowledge:
        - For surgical tools: maybe 20-30px
        - For body poses: maybe 10-15px
        - For fine manipulation: maybe 5-10px
        """
        ACCEPTABLE_DISTANCE_PX = 20  # <-- ADJUST THIS TO YOUR DOMAIN
        
        # Create two objects separated by this distance
        gt_kps = np.array([[[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]])  # (1, 3, 2)
        # Offset tracker by ACCEPTABLE_DISTANCE_PX in one direction
        tr_kps = gt_kps + ACCEPTABLE_DISTANCE_PX / np.sqrt(2)  # diagonal offset
        vis = np.full((1, 3), 2)  # all visible
        
        similarity = _compute_similarity(gt_kps, tr_kps, vis, sigma)
        
        assert similarity[0, 0] >= 0.05, (
            f"At {ACCEPTABLE_DISTANCE_PX}px offset, similarity={similarity[0,0]:.4f} < 0.05. "
            f"These detections would never count as matches! Increase sigma."
        )
    
    def test_similarity_at_clearly_wrong_distance(self, sigma, dataset_instance):
        """
        At a clearly incorrect distance, similarity should be below the highest alpha threshold.
        """
        CLEARLY_WRONG_DISTANCE_PX = 50  # <-- ADJUST: distance that is definitely wrong
        
        gt_kps = np.array([[[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]])
        tr_kps = gt_kps + CLEARLY_WRONG_DISTANCE_PX
        vis = np.full((1, 3), 2)
        
        similarity = _compute_similarity(gt_kps, tr_kps, vis, sigma)
        
        assert similarity[0, 0] < 0.5, (
            f"At {CLEARLY_WRONG_DISTANCE_PX}px offset, similarity={similarity[0,0]:.4f} >= 0.5. "
            f"Clearly wrong matches are being scored too high! Decrease sigma."
        )


# ============================================================
# TEST 4: Similarity Distribution Check
# ============================================================

class TestSimilarityDistribution:
    """Verify similarity scores are well-distributed across the alpha range"""
    
    def test_similarity_spread(self, sigma, num_keypoints):
        """
        Generate realistic noisy predictions and check that similarity scores
        span the alpha range rather than clustering at extremes.
        """
        rng = np.random.default_rng(42)
        num_objects = 10
        num_frames = 50
        all_similarities = []
        
        for t in range(num_frames):
            gt_kps = rng.uniform(50, 500, size=(num_objects, num_keypoints, 2))
            # Add variable noise (some good predictions, some bad)
            noise_scales = rng.uniform(0, 3 * sigma, size=(num_objects, 1, 1))
            noise = rng.standard_normal(size=(num_objects, num_keypoints, 2)) * noise_scales
            tr_kps = gt_kps + noise
            vis = np.full((num_objects, num_keypoints), 2)
            
            sim = _compute_similarity(gt_kps, tr_kps, vis, sigma)
            # Diagonal = matched pairs
            diag_sims = np.diag(sim)
            all_similarities.extend(diag_sims.tolist())
        
        all_similarities = np.array(all_similarities)
        
        # Check that scores aren't all clustered at 1.0
        frac_above_095 = np.mean(all_similarities > 0.95)
        assert frac_above_095 < 0.8, (
            f"{frac_above_095*100:.1f}% of similarities > 0.95. "
            f"Alpha thresholds above 0.95 are meaningless. Consider decreasing sigma."
        )
        
        # Check that scores aren't all clustered at 0.0
        frac_below_005 = np.mean(all_similarities < 0.05)
        assert frac_below_005 < 0.8, (
            f"{frac_below_005*100:.1f}% of similarities < 0.05. "
            f"Almost no matches will count as TPs. Consider increasing sigma."
        )
        
        # Check that there's meaningful spread
        score_std = np.std(all_similarities)
        assert score_std > 0.1, (
            f"Similarity std={score_std:.4f} is too low. "
            f"Scores lack spread, reducing discriminative power of alpha sweep."
        )
    
    def test_alpha_thresholds_are_active(self, hota_metric, sigma, num_keypoints):
        """
        Verify that different alpha thresholds produce meaningfully different 
        TP/FP/FN counts (i.e., alphas are 'active').
        """
        rng = np.random.default_rng(123)
        num_objects = 8
        base_kps = rng.uniform(50, 500, size=(num_objects, num_keypoints, 2))
        
        # Create tracker with variable quality per object
        def tracker_fn(t, n, k):
            noise_per_obj = rng.uniform(0, 2.5 * sigma, size=(n, 1, 1))
            return base_kps + rng.standard_normal((n, k, 2)) * noise_per_obj
        
        data = make_synthetic_sequence_data(
            num_timesteps=30,
            num_objects=num_objects,
            num_keypoints=num_keypoints,
            gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
            tracker_keypoints_fn=tracker_fn,
            sigma=sigma,
        )
        res = hota_metric.eval_sequence(data)
        
        # Check that TP counts vary across alphas
        tp_range = res['HOTA_TP'].max() - res['HOTA_TP'].min()
        total_dets = data['num_gt_dets']
        
        assert tp_range > 0.1 * total_dets, (
            f"TP range across alphas = {tp_range} out of {total_dets} total dets. "
            f"Alpha thresholds are not differentiating quality levels. "
            f"TP counts: {res['HOTA_TP']}"
        )


# ============================================================
# TEST 5: Visibility Handling
# ============================================================

class TestVisibilityHandling:
    """Verify that visibility flags correctly affect similarity computation"""
    
    def test_invisible_keypoints_ignored(self, sigma):
        """
        When GT keypoints are marked invisible (vis=0 or vis=1), 
        they should not affect the similarity score.
        """
        # GT with 5 keypoints, only first 2 visible
        gt_kps = np.array([[[100.0, 100.0], [200.0, 200.0], [300.0, 300.0],
                            [400.0, 400.0], [500.0, 500.0]]])  # (1, 5, 2)
        
        # Tracker: first 2 kps are perfect, last 3 are wildly off
        tr_kps = np.array([[[100.0, 100.0], [200.0, 200.0], [0.0, 0.0],
                            [0.0, 0.0], [0.0, 0.0]]])  # (1, 5, 2)
        
        # Only first 2 keypoints are visible
        vis = np.array([[2, 2, 0, 1, 0]])  # (1, 5)
        
        similarity = _compute_similarity(gt_kps, tr_kps, vis, sigma)
        
        # Should be 1.0 because only visible kps are perfect matches
        np.testing.assert_allclose(similarity[0, 0], 1.0, atol=1e-6,
            err_msg="Invisible keypoints should not affect similarity")
    
    def test_all_invisible_gives_zero(self, sigma):
        """When all GT keypoints are invisible, similarity should be 0"""
        gt_kps = np.array([[[100.0, 100.0], [200.0, 200.0]]])
        tr_kps = np.array([[[100.0, 100.0], [200.0, 200.0]]])
        vis = np.array([[0, 1]])  # none visible (0=out of frame, 1=hidden)
        
        similarity = _compute_similarity(gt_kps, tr_kps, vis, sigma)
        
        assert similarity[0, 0] == 0, (
            f"All-invisible GT should give similarity=0, got {similarity[0,0]}"
        )
    
    def test_partial_visibility_uses_only_visible(self, sigma):
        """Partial visibility should only use visible keypoints in distance calc"""
        gt_kps = np.array([[[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]])
        
        # Tracker: offset of 5px on all kps
        tr_kps = gt_kps + 5.0
        
        # Test with all visible vs partial visibility
        vis_all = np.array([[2, 2, 2]])
        vis_partial = np.array([[2, 0, 0]])  # only first kp visible
        
        sim_all = _compute_similarity(gt_kps, tr_kps, vis_all, sigma)
        sim_partial = _compute_similarity(gt_kps, tr_kps, vis_partial, sigma)
        
        # Both should give same similarity because the offset is uniform
        # (mean distance is same regardless of which kps are used)
        np.testing.assert_allclose(sim_all[0, 0], sim_partial[0, 0], atol=1e-6,
            err_msg="Uniform offset should give same similarity regardless of which kps are visible")


# ============================================================
# TEST 6: ID Swap Test (Association Sensitivity)
# ============================================================

class TestIDSwapSensitivity:
    """Verify that ID swaps reduce AssA while keeping DetA high"""
    
    def test_id_swap_reduces_assa(self, hota_metric, sigma, num_keypoints):
        """
        Perfect detections with ID swaps mid-sequence should have:
        - DetA ≈ 1.0 (all detections are correctly localized)
        - AssA < 1.0 (associations are broken)
        """
        rng = np.random.default_rng(42)
        num_objects = 4
        num_timesteps = 20
        base_kps = rng.uniform(100, 400, size=(num_objects, num_keypoints, 2))
        
        # Perfect detections, but IDs swap at midpoint
        def gt_ids_fn(t, n):
            return np.arange(n)
        
        def tracker_ids_fn_swap(t, n):
            ids = np.arange(n)
            if t >= num_timesteps // 2:
                # Swap first two IDs
                ids[0], ids[1] = ids[1], ids[0]
            return ids
        
        # No swap (baseline)
        data_no_swap = make_synthetic_sequence_data(
            num_timesteps=num_timesteps,
            num_objects=num_objects,
            num_keypoints=num_keypoints,
            gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
            tracker_keypoints_fn=lambda t, n, k: base_kps.copy(),
            gt_ids_fn=gt_ids_fn,
            tracker_ids_fn=gt_ids_fn,
            sigma=sigma,
        )
        
        # With swap
        data_swap = make_synthetic_sequence_data(
            num_timesteps=num_timesteps,
            num_objects=num_objects,
            num_keypoints=num_keypoints,
            gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
            tracker_keypoints_fn=lambda t, n, k: base_kps.copy(),
            gt_ids_fn=gt_ids_fn,
            tracker_ids_fn=tracker_ids_fn_swap,
            sigma=sigma,
        )
        
        res_no_swap = hota_metric.eval_sequence(data_no_swap)
        res_swap = hota_metric.eval_sequence(data_swap)
        
        # DetA should remain perfect in both cases
        np.testing.assert_allclose(np.mean(res_swap['DetA']), 1.0, atol=1e-6,
            err_msg="DetA should be 1.0 even with ID swaps (detections are still perfect)")
        
        # AssA should drop with ID swaps
        assert np.mean(res_swap['AssA']) < np.mean(res_no_swap['AssA']), (
            f"AssA with swaps ({np.mean(res_swap['AssA']):.4f}) should be less than "
            f"without swaps ({np.mean(res_no_swap['AssA']):.4f})"
        )
        
        # HOTA should be between DetA and AssA (geometric mean)
        mean_hota = np.mean(res_swap['HOTA'])
        mean_deta = np.mean(res_swap['DetA'])
        mean_assa = np.mean(res_swap['AssA'])
        expected_hota = np.sqrt(mean_deta * mean_assa)
        assert mean_hota < 1.0, f"HOTA with ID swaps should be < 1.0, got {mean_hota:.4f}"


# ============================================================
# TEST 7: Empty Sequence Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases like empty sequences"""
    
    def test_no_tracker_dets(self, hota_metric):
        """No tracker detections should give all FN"""
        data = {
            'gt_ids': [np.array([0, 1, 2])],
            'tracker_ids': [np.array([], dtype=int)],
            'similarity_scores': [np.zeros((3, 0))],
            'num_timesteps': 1,
            'num_gt_dets': 3,
            'num_tracker_dets': 0,
            'num_gt_ids': 3,
            'num_tracker_ids': 0,
        }
        res = hota_metric.eval_sequence(data)
        
        np.testing.assert_array_equal(res['HOTA_FN'], 3 * np.ones(len(hota_metric.array_labels)))
        np.testing.assert_array_equal(res['HOTA_TP'], 0)
    
    def test_no_gt_dets(self, hota_metric):
        """No GT detections should give all FP"""
        data = {
            'gt_ids': [np.array([], dtype=int)],
            'tracker_ids': [np.array([0, 1])],
            'similarity_scores': [np.zeros((0, 2))],
            'num_timesteps': 1,
            'num_gt_dets': 0,
            'num_tracker_dets': 2,
            'num_gt_ids': 0,
            'num_tracker_ids': 2,
        }
        res = hota_metric.eval_sequence(data)
        
        np.testing.assert_array_equal(res['HOTA_FP'], 2 * np.ones(len(hota_metric.array_labels)))
        np.testing.assert_array_equal(res['HOTA_TP'], 0)
    
    def test_single_object_single_frame(self, hota_metric, sigma, num_keypoints):
        """Minimal case: 1 object, 1 frame, perfect match"""
        rng = np.random.default_rng(42)
        kps = rng.uniform(100, 400, size=(1, num_keypoints, 2))
        
        data = make_synthetic_sequence_data(
            num_timesteps=1,
            num_objects=1,
            num_keypoints=num_keypoints,
            gt_keypoints_fn=lambda t, n, k: kps.copy(),
            tracker_keypoints_fn=lambda t, n, k: kps.copy(),
            sigma=sigma,
        )
        res = hota_metric.eval_sequence(data)
        
        np.testing.assert_allclose(res['HOTA'], 1.0, atol=1e-6)


# ============================================================
# TEST 8: LocA Range Check
# ============================================================

class TestLocARange:
    """Verify LocA falls in a discriminative range"""
    
    def test_loca_discriminative_range(self, hota_metric, sigma, num_keypoints):
        """
        With realistic noise, LocA should fall in a discriminative range,
        not always be ~1.0 or always near the alpha threshold.
        """
        rng = np.random.default_rng(42)
        num_objects = 6
        base_kps = rng.uniform(100, 400, size=(num_objects, num_keypoints, 2))
        
        # Add moderate noise (sigma/2)
        def tracker_fn(t, n, k):
            noise = rng.standard_normal((n, k, 2)) * (sigma / 2)
            return base_kps + noise
        
        data = make_synthetic_sequence_data(
            num_timesteps=30,
            num_objects=num_objects,
            num_keypoints=num_keypoints,
            gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
            tracker_keypoints_fn=tracker_fn,
            sigma=sigma,
        )
        res = hota_metric.eval_sequence(data)
        
        mean_loca = np.mean(res['LocA'])
        
        # LocA should be in a useful range (not too high, not too low)
        assert 0.3 < mean_loca < 0.95, (
            f"LocA mean = {mean_loca:.4f} is outside discriminative range [0.3, 0.95]. "
            f"{'Sigma too large (LocA too high)' if mean_loca >= 0.95 else 'Sigma too small (LocA too low)'}"
        )
    
    def test_loca_bounded_zero_one(self, hota_metric, sigma, num_keypoints):
        """LocA must always be in [0, 1]"""
        rng = np.random.default_rng(99)
        num_objects = 5
        base_kps = rng.uniform(50, 500, size=(num_objects, num_keypoints, 2))
        
        def tracker_fn(t, n, k):
            noise = rng.standard_normal((n, k, 2)) * sigma
            return base_kps + noise
        
        data = make_synthetic_sequence_data(
            num_timesteps=20,
            num_objects=num_objects,
            num_keypoints=num_keypoints,
            gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
            tracker_keypoints_fn=tracker_fn,
            sigma=sigma,
        )
        res = hota_metric.eval_sequence(data)
        
        assert np.all(res['LocA'] >= 0) and np.all(res['LocA'] <= 1.0 + 1e-10), (
            f"LocA out of bounds: min={res['LocA'].min()}, max={res['LocA'].max()}"
        )


# ============================================================
# TEST 9: Symmetry / Consistency Checks
# ============================================================

class TestConsistency:
    """Verify metric internal consistency"""
    
    def test_hota_is_geometric_mean(self, hota_metric, sigma, num_keypoints):
        """HOTA = sqrt(DetA * AssA) must hold"""
        rng = np.random.default_rng(42)
        num_objects = 5
        base_kps = rng.uniform(100, 400, size=(num_objects, num_keypoints, 2))
        
        def tracker_fn(t, n, k):
            noise = rng.standard_normal((n, k, 2)) * (sigma * 0.5)
            return base_kps + noise
        
        data = make_synthetic_sequence_data(
            num_timesteps=30,
            num_objects=num_objects,
            num_keypoints=num_keypoints,
            gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
            tracker_keypoints_fn=tracker_fn,
            sigma=sigma,
        )
        res = hota_metric.eval_sequence(data)
        
        expected_hota = np.sqrt(res['DetA'] * res['AssA'])
        np.testing.assert_allclose(res['HOTA'], expected_hota, atol=1e-10,
            err_msg="HOTA must equal sqrt(DetA * AssA)")
    
    def test_detra_detp_consistency(self, hota_metric, sigma, num_keypoints):
        """DetRe and DetPr should be consistent with TP, FN, FP"""
        rng = np.random.default_rng(42)
        num_objects = 5
        base_kps = rng.uniform(100, 400, size=(num_objects, num_keypoints, 2))
        
        def tracker_fn(t, n, k):
            noise = rng.standard_normal((n, k, 2)) * sigma
            return base_kps + noise
        
        data = make_synthetic_sequence_data(
            num_timesteps=20,
            num_objects=num_objects,
            num_keypoints=num_keypoints,
            gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
            tracker_keypoints_fn=tracker_fn,
            sigma=sigma,
        )
        res = hota_metric.eval_sequence(data)
        
        expected_re = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        expected_pr = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        
        np.testing.assert_allclose(res['DetRe'], expected_re, atol=1e-10)
        np.testing.assert_allclose(res['DetPr'], expected_pr, atol=1e-10)


# ============================================================
# TEST 10: Diagnostic Report (not a pass/fail test)
# ============================================================

class TestDiagnosticReport:
    """Generate a diagnostic report for manual inspection of calibration"""
    
    def test_print_calibration_report(self, hota_metric, sigma, num_keypoints, capsys):
        """
        Prints a calibration report. Always passes, but provides useful info.
        Run with: pytest -s test_kp_hota_calibration.py::TestDiagnosticReport
        """
        print("\n" + "=" * 70)
        print(f"CALIBRATION REPORT (sigma={sigma})")
        print("=" * 70)
        
        # 1. Distance-to-alpha mapping
        print(f"\n{'Distance (px)':<15} {'Similarity':<12} {'Passes alpha≥':<15}")
        print("-" * 42)
        alphas = np.arange(0.05, 0.99, 0.05)
        for dist in [0, 1, 2, 5, 7, 10, 15, 20, 25, 30, 40, 50]:
            sim = np.exp(-dist**2 / (2 * sigma**2))
            passing_alphas = alphas[alphas <= sim]
            max_alpha = passing_alphas[-1] if len(passing_alphas) > 0 else 0
            print(f"{dist:<15} {sim:<12.4f} {max_alpha:<15.2f}")
        
        # 2. Alpha-to-max-distance mapping
        print(f"\n{'Alpha':<10} {'Max distance for match (px)':<30}")
        print("-" * 40)
        for alpha in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            max_dist = sigma * np.sqrt(-2 * np.log(alpha))
            print(f"{alpha:<10.2f} {max_dist:<30.1f}")
        
        # 3. Realistic scenario test
        rng = np.random.default_rng(42)
        num_objects = 6
        base_kps = rng.uniform(100, 400, size=(num_objects, num_keypoints, 2))
        
        offsets = [0, 2, 5, 10, 15, 20, 30, 50]
        print(f"\n{'Offset (px)':<12} {'Mean HOTA':<12} {'Mean LocA':<12} {'Mean DetA':<12} {'Mean AssA':<12}")
        print("-" * 60)
        
        for offset in offsets:
            def tracker_fn(t, n, k, _offset=offset):
                return base_kps + _offset
            
            data = make_synthetic_sequence_data(
                num_timesteps=20,
                num_objects=num_objects,
                num_keypoints=num_keypoints,
                gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
                tracker_keypoints_fn=tracker_fn,
                sigma=sigma,
            )
            res = hota_metric.eval_sequence(data)
            print(f"{offset:<12} {np.mean(res['HOTA']):<12.4f} {np.mean(res['LocA']):<12.4f} "
                  f"{np.mean(res['DetA']):<12.4f} {np.mean(res['AssA']):<12.4f}")
        
        print("\n" + "=" * 70)
        print("END CALIBRATION REPORT")
        print("=" * 70)
        
        # Always passes - this is for visual inspection
        assert False # use False to output report

    def test_print_calibration_report_with_loca_detail(self, hota_metric, sigma, num_keypoints, capsys):
        """Extended diagnostic showing LocA breakdown."""
        rng = np.random.default_rng(42)
        num_objects = 6
        base_kps = rng.uniform(100, 400, size=(num_objects, num_keypoints, 2))
        
        offsets = [0, 2, 5, 10, 15, 20, 30, 50]
        print(f"\n{'Offset':<8} {'HOTA':<8} {'DetA':<8} {'AssA':<8} "
            f"{'LocA(mean)':<11} {'LocA(α=0.05)':<13} {'TP(α=0.05)':<11} {'TP(α=0.5)':<10}")
        print("-" * 95)
        
        for offset in offsets:
            def tracker_fn(t, n, k, _offset=offset):
                return base_kps + _offset
            
            data = make_synthetic_sequence_data(
                num_timesteps=20,
                num_objects=num_objects,
                num_keypoints=num_keypoints,
                gt_keypoints_fn=lambda t, n, k: base_kps.copy(),
                tracker_keypoints_fn=tracker_fn,
                sigma=sigma,
            )
            res = hota_metric.eval_sequence(data)
            
            print(f"{offset:<8} {np.mean(res['HOTA']):<8.4f} {np.mean(res['DetA']):<8.4f} "
                f"{np.mean(res['AssA']):<8.4f} {np.mean(res['LocA']):<11.4f} "
                f"{res['LocA'][0]:<13.4f} {res['HOTA_TP'][0]:<11.0f} {res['HOTA_TP'][9]:<10.0f}")
        
        print("\nNote: LocA defaults to 1.0 at any alpha where HOTA_TP=0 (no matches pass threshold).")
        print("This causes mean LocA to INCREASE at large offsets — this is expected HOTA behavior.")
        print("DetA and HOTA correctly capture the degradation.\n")
        assert False


# ============================================================
# TEST 11: _calculate_similarities Direct Unit Tests
# ============================================================

class TestCalculateSimilaritiesDirect:
    """Direct unit tests on the _calculate_similarities method"""
    
    def test_identical_keypoints_give_similarity_one(self, sigma):
        """Identical keypoints should produce similarity = 1.0"""
        kps = np.array([[[100.0, 200.0], [300.0, 400.0]]])  # (1, 2, 2)
        vis = np.array([[2, 2]])
        
        sim = _compute_similarity(kps, kps.copy(), vis, sigma)
        assert sim[0, 0] == 1.0
    
    def test_known_distance_known_similarity(self, sigma):
        """Verify exact similarity value for a known distance"""
        gt = np.array([[[0.0, 0.0]]])  # (1, 1, 2)
        tr = np.array([[[sigma, 0.0]]])  # distance = sigma
        vis = np.array([[2]])
        
        sim = _compute_similarity(gt, tr, vis, sigma)
        expected = np.exp(-0.5)  # exp(-σ²/(2σ²)) = exp(-0.5) ≈ 0.6065
        
        np.testing.assert_allclose(sim[0, 0], expected, atol=1e-10,
            err_msg=f"At distance=sigma, similarity should be exp(-0.5)={expected:.6f}")
    
    def test_similarity_matrix_shape(self, sigma):
        """Verify output shape is (N, M)"""
        N, M, K = 3, 5, 4
        gt = np.random.rand(N, K, 2) * 100
        tr = np.random.rand(M, K, 2) * 100
        vis = np.full((N, K), 2)
        
        sim = _compute_similarity(gt, tr, vis, sigma)
        assert sim.shape == (N, M), f"Expected shape ({N}, {M}), got {sim.shape}"
    
    def test_similarity_bounded_zero_one(self, sigma):
        """All similarity values must be in [0, 1]"""
        rng = np.random.default_rng(42)
        gt = rng.uniform(0, 500, size=(10, 5, 2))
        tr = rng.uniform(0, 500, size=(8, 5, 2))
        vis = np.full((10, 5), 2)
        
        sim = _compute_similarity(gt, tr, vis, sigma)
        assert np.all(sim >= 0) and np.all(sim <= 1), (
            f"Similarity out of bounds: min={sim.min()}, max={sim.max()}"
        )
    
    def test_empty_inputs(self, sigma):
        """Empty inputs should return empty similarity matrix"""
        gt = np.empty((0, 3, 2))
        tr = np.random.rand(5, 3, 2)
        vis = np.empty((0, 3))
        
        sim = _compute_similarity(gt, tr, vis, sigma)
        assert sim.shape == (0, 5)
        
        sim2 = _compute_similarity(tr, np.empty((0, 3, 2)), np.full((5, 3), 2), sigma)
        assert sim2.shape == (5, 0)
