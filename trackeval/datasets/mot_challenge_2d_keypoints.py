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
            'GT_FOLDER': os.path.join(code_path, 'data/gt/'),
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/'),
            'OUTPUT_FOLDER': None,
            'TRACKERS_TO_EVAL': None,
            'CLASSES_TO_EVAL': ['left hand', 'right hand', 'scissors', 'tweezers',
                                'needle holder', 'needle'],
            'BENCHMARK': 'MOT17',
            'SPLIT_TO_EVAL': 'train',
            'INPUT_AS_ZIP': False,
            'PRINT_CONFIG': True,
            'DO_PREPROC': True,
            'TRACKER_SUB_FOLDER': '',
            'OUTPUT_SUB_FOLDER': '',
            'TRACKER_DISPLAY_NAMES': None,
            'SEQMAP_FOLDER': None,
            'SEQMAP_FILE': None,
            'SEQ_INFO': None,
            'GT_LOC_FORMAT': '{gt_folder}/{seq}.txt',
            'TRACKER_LOC_FORMAT': '{trackers_folder}/{seq}_pred.txt',    # other options include: 'tracker' for multiple trackers and {tracker_sub_fol} for subfolder
            'SKIP_SPLIT_FOL': True,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
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

        # Get classes to eval
        self.class_name_to_class_id = {
            'left hand': 0,
            'right hand': 1,
            'scissors': 2,
            'tweezers': 3,
            'needle holder': 4,
            'needle': 5
        }
        self.valid_classes = list(self.class_name_to_class_id.keys())
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. Only left hand, right hand, scissors, tweezers, needle holder, and needle classes are valid.')
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
                if not os.path.isfile(curr_file):
                    print('GT file not found ' + curr_file)
                    raise TrackEvalException('GT file not found for sequence: ' + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, 'data.zip')
            if not os.path.isfile(curr_file):
                print('GT file not found ' + curr_file)
                raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    print('Tracker file not found: ' + curr_file)
                    raise TrackEvalException(
                        'Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = curr_file = self.config["TRACKER_LOC_FORMAT"].format(trackers_folder=self.tracker_fol, tracker=tracker, tracker_sub_fol=self.tracker_sub_fol,seq=seq)
                    # curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '_pred.txt')
                    if not os.path.isfile(curr_file):
                        print('Tracker file not found: ' + curr_file)
                        raise TrackEvalException(
                            'Tracker file not found: ' + curr_file)


    def _get_seq_info(self):
        seq_list = []
        seq_lengths = {}
        seq_frame_rates = {}

        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]

            # If sequence length is 'None' tries to read sequence length from .ini files.
            for seq, seq_length in seq_lengths.items():
                if isinstance(seq_length, dict): # added to include frame rate which is needed for parsing raw data file
                    seq_list.append(seq)
                    seq_lengths[seq] = seq_length.get('length')
                    seq_frame_rates[seq] = seq_length.get('frame_rate')
                else:
                    seq_list.append(seq)
                    seq_lengths[seq] = seq_length
                    seq_frame_rates[seq] = None

                if seq_length is None:
                    ini_data = self._get_ini_data(seq)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
                    seq_frame_rates[seq] = int(ini_data['Sequence']['frameRate'])
        else:
            if self.config["SEQMAP_FILE"]:
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                if self.config["SEQMAP_FOLDER"] is None:
                    seqmap_file = os.path.join(self.config['GT_FOLDER'], 'seqmaps', self.gt_set + '.txt')
                else:
                    seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], self.gt_set + '.txt')
            if not os.path.isfile(seqmap_file):
                print('no seqmap found: ' + seqmap_file)
                raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == '':
                        continue
                    seq = row[0]
                    seq_list.append(seq)
                    ini_data = self._get_ini_data(seq)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
                    seq_frame_rates[seq] = int(ini_data['Sequence']['frameRate'])
        self.seq_frame_rates = seq_frame_rates
        return seq_list, seq_lengths

    def _get_ini_data(self, seq):
        ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
        if not os.path.isfile(ini_file):
            raise TrackEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
        ini_data = configparser.ConfigParser()
        ini_data.read(ini_file)
        return ini_data

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


    def get_preprocessed_seq_data(self, raw_data, cls):
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
            'gt_dets', 'tracker_dets',
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
            gt_keypoints = raw_data['gt_dets'][t]  # shape: (num_gt, num_kp, 2)
            # Only keep gt of correct class
            keep_gt = (gt_classes == cls_id)
            data['gt_ids'][t] = gt_ids[keep_gt]
            data['gt_dets'][t] = gt_keypoints[keep_gt]
            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            num_gt_dets += len(data['gt_ids'][t])

            # Tracker
            tracker_ids = raw_data['tracker_ids'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_keypoints = raw_data['tracker_dets'][t]  # shape: (num_tr, num_kp, 2)
            tracker_confidences = raw_data['tracker_confidences'][t]  # shape: (num_tr, num_kp)
            # Only keep tracker dets of correct class
            keep_tr = (tracker_classes == cls_id)
            data['tracker_ids'][t] = tracker_ids[keep_tr]
            data['tracker_dets'][t] = tracker_keypoints[keep_tr]
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
