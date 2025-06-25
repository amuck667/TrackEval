import os
import csv
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException

class MotChallenge2DKeypoints(_BaseDataset):
    """Dataset class for MOT Challenge 2D keypoint tracking (for KP_HOTA)"""

    @staticmethod
    def get_default_dataset_config():
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),
            'OUTPUT_FOLDER': None,
            'TRACKERS_TO_EVAL': None,
            'CLASSES_TO_EVAL': ['pedestrian'],
            'BENCHMARK': 'MOT17',
            'SPLIT_TO_EVAL': 'train',
            'INPUT_AS_ZIP': False,
            'PRINT_CONFIG': True,
            'DO_PREPROC': True,
            'TRACKER_SUB_FOLDER': 'data',
            'OUTPUT_SUB_FOLDER': '',
            'TRACKER_DISPLAY_NAMES': None,
            'SEQMAP_FOLDER': None,
            'SEQMAP_FILE': None,
            'SEQ_INFO': None,
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
            'SKIP_SPLIT_FOL': False,
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.benchmark = self.config['BENCHMARK']
        gt_set = self.config['BENCHMARK'] + '-' + self.config['SPLIT_TO_EVAL']
        self.gt_set = gt_set
        if not self.config['SKIP_SPLIT_FOL']:
            split_fol = gt_set
        else:
            split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], split_fol)
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], split_fol)
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']
        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']
        self.class_name_to_class_id = {
            'left hand': 0,
            'right hand': 1,
            'scissors': 2,
            'tweezers': 3,
            'needle holder': 4,
            'needle': 5
        }
        self.valid_classes = list(self.class_name_to_class_id.keys())

    def _load_raw_file(self, tracker, seq, is_gt):
        # File location
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, 'data.zip')
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
            file = seq + '.txt'
        else:
            zip_file = None
            if is_gt:
                file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
            else:
                file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
        read_data, ignore_data = self._load_simple_text_file(file, is_zipped=self.data_is_zipped, zip_file=zip_file)
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'keypoints', 'kp_confs']
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        current_time_keys = [str(t + 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            text = 'Ground-truth' if is_gt else 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))
        for t in range(num_timesteps):
            time_key = str(t + 1)
            if time_key in read_data.keys():
                time_data = np.asarray(read_data[time_key], dtype=np.float)
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                kp_cols = time_data[:, 6:]
                n_kps = kp_cols.shape[1] // 3
                keypoints = np.stack([kp_cols[:, i*3:i*3+2] for i in range(n_kps)], axis=1)  # (N, K, 2)
                kp_confs = np.stack([kp_cols[:, i*3+2] for i in range(n_kps)], axis=1)  # (N, K)
                raw_data['keypoints'][t] = keypoints
                raw_data['kp_confs'][t] = kp_confs
            else:
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['keypoints'][t] = np.empty((0, 0, 2))
                raw_data['kp_confs'][t] = np.empty((0, 0))
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data


    def get_processed_seq_data(self, raw_data, cls):
        """
        Preprocess data for a single sequence for a single class for keypoint-based MOT.
        - raw_data: dict from get_raw_seq_data()
        - cls: class to evaluate
        Returns a dict with keys:
            [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets]
            [gt_ids, tracker_ids, gt_keypoints, tracker_keypoints, tracker_confidences]
        """
        # Map class name to class id
        cls_id = self.class_name_to_class_id[cls]
        data_keys = [
            'gt_ids', 'tracker_ids',
            'gt_keypoints', 'tracker_keypoints',
            'tracker_confidences'
        ]
        num_timesteps = raw_data['num_timesteps']
        data = {k: [None] * num_timesteps for k in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(num_timesteps):
            # GT
            gt_ids = raw_data['gt_ids'][t]
            gt_classes = raw_data['gt_classes'][t]
            gt_keypoints = raw_data['gt_keypoints'][t]  # shape: (num_gt, num_kp, 2)
            # Only keep gt of correct class
            keep_gt = (gt_classes == cls_id)
            data['gt_ids'][t] = gt_ids[keep_gt]
            data['gt_keypoints'][t] = gt_keypoints[keep_gt]
            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            num_gt_dets += len(data['gt_ids'][t])

            # Tracker
            tracker_ids = raw_data['tracker_ids'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_keypoints = raw_data['tracker_keypoints'][t]  # shape: (num_tr, num_kp, 2)
            tracker_confidences = raw_data['tracker_confidences'][t]  # shape: (num_tr, num_kp)
            # Only keep tracker dets of correct class
            keep_tr = (tracker_classes == cls_id)
            data['tracker_ids'][t] = tracker_ids[keep_tr]
            data['tracker_keypoints'][t] = tracker_keypoints[keep_tr]
            data['tracker_confidences'][t] = tracker_confidences[keep_tr]
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])

        # Relabel IDs to contiguous
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(num_timesteps):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int_)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(num_timesteps):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int_)

        # Overview stats
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_timesteps'] = num_timesteps
        data['seq'] = raw_data['seq']

        return data


    def _calculate_similarities(self, gt_keypoints, tracker_keypoints, zero_distance=2.0):
        # gt_keypoints: (N, K, 2), tracker_keypoints: (M, K, 2)
        if gt_keypoints.shape[0] == 0 or tracker_keypoints.shape[0] == 0:
            return np.zeros((gt_keypoints.shape[0], tracker_keypoints.shape[0]))
        # If number of keypoints differs, only compare up to the minimum
        min_kps = min(gt_keypoints.shape[1], tracker_keypoints.shape[1])
        gt_kps = gt_keypoints[:, :min_kps, :]
        trk_kps = tracker_keypoints[:, :min_kps, :]
        # Compute mean Euclidean distance for each pair
        dist = np.zeros((gt_kps.shape[0], trk_kps.shape[0]))
        for i in range(gt_kps.shape[0]):
            for j in range(trk_kps.shape[0]):
                dist[i, j] = np.mean(np.linalg.norm(gt_kps[i] - trk_kps[j], axis=1))
        sim = np.maximum(0, 1 - dist / zero_distance)
        return sim
