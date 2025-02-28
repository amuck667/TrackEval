import numpy as np
import pytest

from trackeval.metrics.hota import KP_HOTA


def test_compute_keypoint_distances():
    evaluator = KP_HOTA()

    # Define ground truth keypoints and predicted keypoints
    gt_keypoints = np.array([
        [[150, 200], [160, 210], [170, 220]],
        [[300, 400], [310, 410], [320, 420]]
    ])
    pred_keypoints = np.array([
        [[152, 202], [162, 212], [172, 222]],
        [[298, 398], [308, 408], [318, 418]]
    ])

    # Expected distance matrix
    expected_dist_matrix = np.array([
        [2.82842712, 247.20032],
        [247.20032, 2.82842712]
    ])

    # Compute the distance matrix
    dist_matrix = evaluator.compute_keypoint_distances(gt_keypoints, pred_keypoints)

    # Assert that the computed distance matrix is close to the expected distance matrix
    np.testing.assert_allclose(dist_matrix, expected_dist_matrix, rtol=1e-5)


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
        'confidence_matrix': [
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([[0.85, 0.15], [0.15, 0.85]])
        ]
    }

    class DummyEvaluator(KP_HOTA):
        def __init__(self):
            super().__init__()
            self.array_labels = [0.5]  # Threshold for matching
            self.float_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'LocA', 'AssA', 'AssRe', 'AssPr']
            self.integer_array_fields = []
            self.float_fields = []

        def compute_keypoint_distances(self, gt_keypoints, pred_keypoints):
            expected_dist_matrix_1 = np.array([
                [2.82842712, 247.20032],
                [247.20032, 2.82842712]
            ])
            expected_dist_matrix_2 = np.array([
                [2.82842712, 252.8003164],
                [247.20032, 2.82842712]
            ])

            dist_matrix = super().compute_keypoint_distances(gt_keypoints, pred_keypoints)

            # Check if the computed distance matrix matches either of the expected matrices
            # Check if it matches either expected matrix
            matches_matrix_1 = np.allclose(dist_matrix, expected_dist_matrix_1, rtol=1e-5)
            matches_matrix_2 = np.allclose(dist_matrix, expected_dist_matrix_2, rtol=1e-5)

            assert matches_matrix_1 or matches_matrix_2, \
                f"dist_matrix did not match either expected matrix. Got: {dist_matrix}"

            return dist_matrix

        def _compute_final_fields(self, res):
            return res  # Dummy function to keep structure

    evaluator = DummyEvaluator()
    result = evaluator.eval_sequence(data)

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


@pytest.mark.parametrize(
    "data, expected_hota",
    [
        (
                # Keypoints Very Divergent
                {
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
                        np.array([[[130, 130], [230, 230], [330, 330]],
                                  [[430, 430], [530, 530], [630, 630]]])
                    ],
                    'confidence_matrix': [
                        np.array([[0.9, 0.1], [0.1, 0.9]]),
                        np.array([[0.85, 0.15], [0.15, 0.85]])
                    ]
                },
                0.1  # Expected HOTA value for divergent keypoints
        ),
        (
                # Keypoints Somewhat Divergent
                {
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
                    'confidence_matrix': [
                        np.array([[0.9, 0.1], [0.1, 0.9]]),
                        np.array([[0.85, 0.15], [0.15, 0.85]])
                    ]
                },
                0.6  # Expected HOTA value for somewhat divergent keypoints
        ),
        (
                # Missing Predictions for Frames
                {
                    'num_tracker_dets': 2,
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
                        np.array([[], [], []]),  # Missing predictions for frame 2
                    ],
                    'confidence_matrix': [
                        np.array([[0.9, 0.1], [0.1, 0.9]]),
                        np.array([[], []])  # Missing confidence for frame 2
                    ]
                },
                0.4  # Expected HOTA value for missing predictions for frames
        ),
        (
                # Missing Predictions for Objects
                {
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
                                  [[], [], []]], dtype=object)  # Missing predictions for object 2 in frame 2 todo
                    ],
                    'confidence_matrix': [
                        np.array([[0.9, 0.1], [0.1, 0.9]]),
                        np.array([[0.85, 0.15], [0, 0]])  # Missing confidence for object 2 in frame 2
                    ]
                },
                0.5  # Expected HOTA value for missing predictions for objects
        )
    ]
)
def test_kphota_scenarios(data, expected_hota):
    # Calculate the HOTA score using the 'calculate_hota' function
    evaluator = KP_HOTA()
    result_hota_dict = evaluator.eval_sequence(data)
    result_hota = sum(result_hota_dict['HOTA']) / len(result_hota_dict['HOTA'])

    # Check if the calculated HOTA is within an acceptable tolerance (e.g., ±0.05)
    assert abs(result_hota - expected_hota) < 0.05

def test_kphota_moreframes():
    # Dummy test data
    '''
    # frame, id, x, y, w, h, x1, y1, v1, x2, y2, v2, x3, y3, v3
    1, 1, -1, -1, -1, -1, 150, 200, 1, 160, 210, 1, 170,220, 1
    1, 2, -1, -1, -1, -1, 300, 400, 1, 310, 410, 1, 320, 420, 1
    2, 1, -1, -1, -1, -1, 155, 205, 1, 165, 215, 1, 175, 225, 1
    2, 2, -1, -1, -1, -1, 305, 405, 1, 315, 415, 1, 325, 425, 1
    3, 1, -1, -1, -1, -1, 160, 210, 1, 170, 220, 1, 180, 230, 1
    3, 2, -1, -1, -1, -1, 310, 410, 1, 320, 420, 1, 330, 430, 1
    4, 1, -1, -1, -1, -1, 165, 215, 1, 175, 225, 1, 185, 235, 1
    4, 2, -1, -1, -1, -1, 315, 415, 1, 325, 425, 1, 335, 435, 1
    5, 1, -1, -1, -1, -1, 170, 220, 1, 180, 230, 1, 190, 240, 1
    5, 2, -1, -1, -1, -1, 320, 420, 1, 330, 430, 1, 340, 440, 1
    this is my pred data:
    # frame, id, x, y, w, h, confidence, x1, y1, v1, x2, y2, v2, x3, y3, v3
    1, 1, -1, -1, -1, -1, 0.9, 152, 202, 1, 162, 212, 1, 172, 222, 1
    1, 2, -1, -1, -1, -1, 0.9, 298, 398, 1, 308, 408, 1,318, 418, 1
    2, 1, -1, -1, -1, -1, 0.85, 157, 207, 1, 167, 217, 1, 177, 227, 1
    2, 2, -1, -1, -1, -1, 0.85, 307, 407, 1, 317, 417, 1, 327, 427, 1
    3, 1, -1, -1, -1, -1, 0.9, 162, 212, 1, 172, 222, 1, 182, 232, 1
    3, 2, -1, -1, -1, -1, 0.9, 312, 412, 1, 322, 422, 1, 332, 432, 1
    4, 1, -1, -1, -1, -1, 0.85, 167, 217, 1, 177, 227, 1, 187, 237, 1
    4, 2, -1, -1, -1, -1, 0.85, 317, 417, 1, 327, 427, 1, 337, 437, 1
    5, 1, -1, -1, -1, -1, 0.9, 172, 222, 1, 182, 232, 1, 192, 242, 1
    5, 2, -1, -1, -1, -1, 0.9, 322, 422, 1, 332, 432, 1, 342, 442, 1
    '''

    data_more_frames = {
        'num_tracker_dets': 6,
        'num_gt_dets': 6,
        'num_gt_ids': 2,
        'num_tracker_ids': 2,
        'gt_ids': [np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1])],
        'tracker_ids': [np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1])],
        'gt_keypoints': [
            np.array([[[150, 200], [160, 210], [170, 220]],
                      [[300, 400], [310, 410], [320, 420]]]),
            np.array([[[155, 205], [165, 215], [175, 225]],
                      [[305, 405], [315, 415], [325, 425]]]),
            np.array([[[160, 210], [170, 220], [180, 230]],
                      [[310, 410], [320, 420], [330, 430]]]),
            np.array([[[165, 215], [175, 225], [185, 235]],
                      [[315, 415], [325, 425], [335, 435]]]),
            np.array([[[170, 220], [180, 230], [190, 240]],
                      [[320, 420], [330, 430], [340, 440]]])
        ],
        'tracker_keypoints': [
            np.array([[[152, 202], [162, 212], [172, 222]],
                      [[298, 398], [308, 408], [318, 418]]]),
            np.array([[[157, 207], [167, 217], [177, 227]],
                      [[303, 403], [313, 413], [323, 423]]]),
            np.array([[[162, 212], [172, 222], [182, 232]],
                      [[308, 408], [318, 418], [328, 428]]]),
            np.array([[[167, 217], [177, 227], [187, 237]],
                      [[313, 413], [323, 423], [333, 433]]]),
            np.array([[[172, 222], [182, 232], [192, 242]],
                      [[318, 418], [328, 428], [338, 438]]])
        ],
        'confidence_matrix': [
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([[0.85, 0.15], [0.15, 0.85]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([[0.85, 0.15], [0.15, 0.85]]),
            np.array([[0.9, 0.1], [0.1, 0.9]])
        ]
    }

    evaluator = KP_HOTA()
    result_hota_dict = evaluator.eval_sequence(data_more_frames)
    result_hota = sum(result_hota_dict['HOTA'])/len(result_hota_dict['HOTA'])
    expected_hota = 0.92
    assert abs(result_hota - expected_hota) < 0.05


def test_kphota_moreobjects():
    # Dummy test data
    '''
    # frame, id, x, y, w, h, x1, y1, v1, x2, y2, v2, x3, y3, v3
    1, 1, -1, -1, -1, -1, 150, 200, 1, 160, 210, 1, 170,220, 1
    1, 2, -1, -1, -1, -1, 300, 400, 1, 310, 410, 1, 320, 420, 1
    1, 3, -1, -1, -1, -1, 450, 500, 1, 460, 510, 1, 470, 520, 1
    2, 1, -1, -1, -1, -1, 155, 205, 1, 165, 215, 1, 175, 225, 1
    2, 2, -1, -1, -1, -1, 305, 405, 1, 315, 415, 1, 325, 425, 1
    2, 3, -1, -1, -1, -1, 455, 505, 1, 465, 515, 1, 475, 525, 1
    3, 1, -1, -1, -1, -1, 160, 210, 1, 170, 220, 1, 180, 230, 1
    3, 2, -1, -1, -1, -1, 310, 410, 1, 320, 420, 1, 330, 430, 1
    4, 1, -1, -1, -1, -1, 165, 215, 1, 175, 225, 1, 185, 235, 1
    4, 2, -1, -1, -1, -1, 315, 415, 1, 325, 425, 1, 335, 435, 1
    4, 3, -1, -1, -1, -1, 460, 510, 1, 470, 520, 1, 480, 530, 1
    5, 1, -1, -1, -1, -1, 170, 220, 1, 180, 230, 1, 190, 240, 1
    5, 2, -1, -1, -1, -1, 320, 420, 1, 330, 430, 1, 340, 440, 1
    5, 3, -1, -1, -1, -1, 465, 515, 1, 475, 525, 1, 485, 535, 1
    this is my pred data:
    # frame, id, x, y, w, h, confidence, x1, y1, v1, x2, y2, v2, x3, y3, v3
    1, 1, -1, -1, -1, -1, 0.9, 152, 202, 1, 162, 212, 1, 172, 222, 1
    1, 2, -1, -1, -1, -1, 0.9, 298, 398, 1, 308, 408, 1,318, 418, 1
    1, 3, -1, -1, -1, -1, 0.9, 448, 498, 1, 458, 508, 1, 468, 518, 1
    2, 1, -1, -1, -1, -1, 0.85, 157, 207, 1, 167, 217, 1, 177, 227, 1
    2, 2, -1, -1, -1, -1, 0.85, 307, 407, 1, 317, 417, 1, 327, 427, 1
    2, 3, -1, -1, -1, -1, 0.85, 457, 507, 1, 467, 517, 1, 477, 527, 1
    3, 1, -1, -1, -1, -1, 0.9, 162, 212, 1, 172, 222, 1, 182, 232, 1
    3, 2, -1, -1, -1, -1, 0.9, 312, 412, 1, 322, 422, 1, 332, 432, 1
    4, 1, -1, -1, -1, -1, 0.85, 167, 217, 1, 177, 227, 1, 187, 237, 1
    4, 2, -1, -1, -1, -1, 0.85, 317, 417, 1, 327, 427, 1, 337, 437, 1
    4, 3, -1, -1, -1, -1, 0.85, 463, 513, 1, 473, 523, 1, 483, 533, 1
    5, 1, -1, -1, -1, -1, 0.9, 172, 222, 1, 182, 232, 1, 192, 242, 1
    5, 2, -1, -1, -1, -1, 0.9, 322, 422, 1, 332, 432, 1, 342, 442, 1
    5, 3, -1, -1, -1, -1, 0.9, 468, 518, 1, 478, 528, 1, 488, 538, 1
    '''
    data_more_objects = {
        'num_tracker_dets': 7,
        'num_gt_dets': 7,
        'num_gt_ids': 3,
        'num_tracker_ids': 3,
        'gt_ids': [np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1]), np.array([0, 1, 2]),
                   np.array([0, 1, 2])],
        'tracker_ids': [np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1]), np.array([0, 1, 2]),
                        np.array([0, 1, 2])],
        'gt_keypoints': [
            np.array([[[150, 200], [160, 210], [170, 220]],
                      [[300, 400], [310, 410], [320, 420]],
                      [[450, 500], [460, 510], [470, 520]]]),
            np.array([[[155, 205], [165, 215], [175, 225]],
                      [[305, 405], [315, 415], [325, 425]],
                      [[455, 505], [465, 515], [475, 525]]]),
            np.array([[[160, 210], [170, 220], [180, 230]],
                      [[310, 410], [320, 420], [330, 430]]]),  # Only 2 objects in frame 3
            np.array([[[165, 215], [175, 225], [185, 235]],
                      [[315, 415], [325, 425], [335, 435]],
                      [[460, 510], [470, 520], [480, 530]]]),
            np.array([[[170, 220], [180, 230], [190, 240]],
                      [[320, 420], [330, 430], [340, 440]],
                      [[465, 515], [475, 525], [485, 535]]])
        ],
        'tracker_keypoints': [
            np.array([[[152, 202], [162, 212], [172, 222]],
                      [[298, 398], [308, 408], [318, 418]],
                      [[448, 498], [458, 508], [468, 518]]]),
            np.array([[[157, 207], [167, 217], [177, 227]],
                      [[303, 403], [313, 413], [323, 423]],
                      [[453, 503], [463, 513], [473, 523]]]),
            np.array([[[162, 212], [172, 222], [182, 232]],
                      [[308, 408], [318, 418], [328, 428]]]),  # Missing object 2 in frame 3
            np.array([[[167, 217], [177, 227], [187, 237]],
                      [[313, 413], [323, 423], [333, 433]],
                      [[463, 513], [473, 523], [483, 533]]]),
            np.array([[[172, 222], [182, 232], [192, 242]],
                      [[318, 418], [328, 428], [338, 438]],
                      [[468, 518], [478, 528], [488, 538]]])
        ],
        'confidence_matrix': [
            np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
            np.array([[0.85, 0.1, 0.05], [0.1, 0.85, 0.05], [0.05, 0.05, 0.9]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),  # Missing object 2 in frame 3
            np.array([[0.85, 0.05, 0.1], [0.05, 0.85, 0.1], [0.1, 0.1, 0.8]]),
            np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        ]
    }

    evaluator = KP_HOTA()
    result_hota_dict = evaluator.eval_sequence(data_more_objects)
    result_hota = sum(result_hota_dict['HOTA']) / len(result_hota_dict['HOTA'])
    expected_hota = 0.9
    assert abs(result_hota - expected_hota) < 0.05