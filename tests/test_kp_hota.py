import numpy as np
import pytest

from trackeval.metrics.hota import KP_HOTA


def test_eval_sequence():
    # Dummy test data
    '''
    # frame, id, x, y, w, h, x1, y1, v1, x2, y2, v2, x3, y3, v3
    1, 1, -1, -1, -1, -1, 150, 200, 1, 160, 210, 1, 170,220, 1
    1, 2, -1, -1, -1, -1, 300, 400, 1, 310, 410, 1, 320, 420, 1
    2, 1, -1, -1, -1, -1, 155, 205, 1, 165, 215, 1, 175, 225, 1
    2, 2, -1, -1, -1, -1, 305, 405, 1, 315, 415, 1, 325, 425, 1
    this is my pred data:
    # frame, id, x, y, w, h, confidence, x1, y1, v1, x2, y2, v2, x3, y3, v3
    1, 1, -1, -1, -1, -1, 0.9, 152, 202, 1, 162, 212, 1, 172, 222, 1
    1, 2, -1, -1, -1, -1, 0.9, 298, 398, 1, 308, 408, 1,318, 418, 1
    2, 1, -1, -1, -1, -1, 0.85, 157, 207, 1, 167, 217, 1, 177, 227, 1
    2, 2, -1, -1, -1, -1, 0.85, 307, 407, 1, 317, 417, 1, 327, 427, 1
    '''
    data = {
        'num_tracker_dets': 4,
        'num_gt_dets': 4,
        'num_gt_ids': 2,
        'num_tracker_ids': 2,
        'gt_ids': [np.array([0, 1]), np.array([0, 1])],
        'tracker_ids': [np.array([0, 1]), np.array([0, 1])],
        'gt_keypoints': [
            np.array([[[150, 200], [160, 210], [170, 220]],
                      [[300, 400], [310, 410], [320, 420]]]),
            np.array([[[155, 205], [165, 215], [175, 225]],
                      [[305, 405], [315, 415], [325, 425]]])
        ],
        'tracker_keypoints': [
            np.array([[[152, 202], [162, 212], [172, 222]],
                      [[298, 398], [308, 408], [318, 418]]]),
            np.array([[[157, 207], [167, 217], [177, 227]],
                      [[307, 407], [317, 417], [327, 427]]])
        ],
        'similarity_scores': [
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([[0.85, 0.15], [0.15, 0.85]])
        ]
    }

    class DummyEvaluator:
        def __init__(self):
            self.array_labels = [0.5]  # Threshold for matching
            self.float_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'LocA', 'AssA', 'AssRe', 'AssPr']
            self.integer_array_fields = []
            self.float_fields = []

        def _compute_final_fields(self, res):
            return res  # Dummy function to keep structure

    evaluator = DummyEvaluator()
    result = KP_HOTA.eval_sequence(evaluator, data)

    # Basic assertions
    assert 'HOTA_TP' in result
    assert 'HOTA_FN' in result
    assert 'HOTA_FP' in result
    assert 'LocA' in result
    assert 'AssA' in result
    assert 'AssRe' in result
    assert 'AssPr' in result

    assert result['HOTA_TP'][0] >= 0
    assert result['HOTA_FN'][0] >= 0
    assert result['HOTA_FP'][0] >= 0
    assert 0 <= result['LocA'][0] <= 1
    assert 0 <= result['AssA'][0] <= 1
    assert 0 <= result['AssRe'][0] <= 1
    assert 0 <= result['AssPr'][0] <= 1