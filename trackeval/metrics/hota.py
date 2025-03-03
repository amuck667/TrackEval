
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class HOTA(_BaseMetric):
    """Class which implements the HOTA metrics.
    See: https://link.springer.com/article/10.1007/s11263-020-01375-2
    """

    def __init__(self, config=None):
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.05, 0.99, 0.05)  # alpha values for HOTA
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields
        self.summary_fields = self.float_array_fields + self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the HOTA metrics for one sequence"""

        # Initialise results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=np.float)
        for field in self.float_fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA(0)'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float)
            res['LocA(0)'] = 1.0
            return res

        # Variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the potential matches between ids in each timestep
            # These are normalised, weighted by the match similarity.
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]

        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue

            # Get matching scores between pairs of dets for optimizing HOTA
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Calculate and accumulate basic statistics
            for a, alpha in enumerate(self.array_labels):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

        # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res['HOTA_TP'][a])

        # Calculate final scores
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_array_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items()
                     if (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()}, field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)

        for field in self.float_fields + self.float_array_fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in all_res.values() if
                                      (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()],
                                     axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['OWTA'] = np.sqrt(res['DetRe'] * res['AssA'])

        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)']*res['LocA(0)']
        return res

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'b--', 'b:', 'g--', 'g:', 'm']
        for name, style in zip(self.float_array_fields, styles_to_plot):
            plt.plot(self.array_labels, res[name], style)
        plt.xlabel('alpha')
        plt.ylabel('score')
        plt.title(tracker + ' - ' + cls)
        plt.axis([0, 1, 0, 1])
        legend = []
        for name in self.float_array_fields:
            legend += [name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')']
        plt.legend(legend, loc='lower left')
        out_file = os.path.join(output_folder, cls + '_plot.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.pdf', '.png'))
        plt.clf()

class KP_HOTA(HOTA):
    """Class which implements the HOTA metrics with keypoint distance matching.
    See: https://link.springer.com/article/10.1007/s11263-020-01375-2
    """

    def __init__(self, config=None):
        super().__init__(config)

    @_timing.time
    def eval_sequence(self, data, sigma = 10):
        """
        Evaluates a tracking sequence using the HOTA metric with keypoint distance matching.

        This function computes tracking accuracy by matching ground truth (GT) detections
        with tracker predictions using Euclidean distances between keypoints instead of IoU.
        The Hungarian algorithm is applied to find the best matches for each timestep,
        and various tracking metrics (HOTA, LocA, AssA, AssRe, AssPr) are computed.

        Parameters:
        -----------
        data : dict
            A dictionary containing tracking data for a sequence with the following keys:
            - 'num_tracker_dets' (int): Total number of tracker detections in the sequence.
            - 'num_gt_dets' (int): Total number of ground truth detections in the sequence.
            - 'num_gt_ids' (int): Number of unique ground truth object IDs.
            - 'num_tracker_ids' (int): Number of unique tracker object IDs.
            - 'gt_ids' (list of arrays): GT object IDs at each timestep.
            - 'tracker_ids' (list of arrays): Tracker object IDs at each timestep.
            - 'gt_keypoints' (list of (N, 2) arrays): GT keypoints at each timestep.
            - 'tracker_keypoints' (list of (M, 2) arrays): Tracker keypoints at each timestep.
            - 'confidence_matrix' (list of arrays): Confidence matrix at each timestep. (frame × gt_ids × tracker_ids)

        Returns:
        --------
        res : dict
            A dictionary containing computed HOTA-related metrics as arrays calculated for each alpha value.:
            - 'HOTA_TP', 'HOTA_FP', 'HOTA_FN' (arrays): True/False Positives and False Negatives.
            - 'LocA' (array): Localization accuracy based on keypoint similarity.
            - 'AssA', 'AssRe', 'AssPr' (arrays): Association metrics.
            - 'LocA(0)' (float): Localization accuracy at threshold 0.
        """

        # Initialise metric results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=float)
        for field in self.float_fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=float)
            res['LocA(0)'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=float)
            res['LocA(0)'] = 1.0
            return res

        # Init variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))

        # First loop through each timestep and accumulate global track information.
        assert(len(data['gt_ids']) == len(data['tracker_ids']))  # Ensure same number of frames, if not the data was not loaded correctly
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])): # this basically loops through the frames, if there are no detections in a frame, there's an empty array
            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
                continue

            # Compute keypoint distance-based similarity matrix
            similarity = self.compute_similarity_from_distance(data['gt_keypoints'][t],
                                                               data['tracker_keypoints'][t], sigma)

            # Accumulate global potential matches count
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += similarity

            # Count detections per GT and tracker ID
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Calculate overall Jaccard alignment score (before unique matching)
        # JAS evaluates how well a tracker maintains correct associations between detections across frames. This helps balance local (frame-wise) and global (over time) associations.
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)

        # Initialize variables for counting matches
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]

        # Calculate scores for each timestep, local track information
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue

            # Compute keypoint distance-based similarity matrix
            similarity = self.compute_similarity_from_distance(data['gt_keypoints'][t], data['tracker_keypoints'][t], sigma)

            # Get final matching scores - optimize per-frame matches when computing HOTA scores: consistency across frames * local (frame-wise) correctness
            score_mat = global_alignment_score[gt_ids_t[:, None], tracker_ids_t[None, :]] * similarity

            # Hungarian algorithm for optimal assignment
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Calculate and accumulate basic statistics
            for a, alpha in enumerate(self.array_labels):
                matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo(float).eps  # optimal matches between the ground truth and tracker objects, which were found using linear_sum_assignment (eps ensures truly >= alpha w numeric precision)
                # indices for correct matches row - gt, col - tracker
                alpha_match_rows = match_rows[matched_mask]
                alpha_match_cols = match_cols[matched_mask]

                num_matches = len(alpha_match_rows)
                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_keypoints_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_keypoints_t) - num_matches

                if num_matches > 0:
                    res['LocA'][a] += np.sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

        # Calculate association scores (AssA, AssRe, AssPr) for each alpha
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res['HOTA_TP'][a])

        # Final location accuracy calculation
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)

        return res

    def compute_similarity_from_distance(self, gt_kps, pred_kps, sigma):
        # Extract keypoints
        gt_keypoints_t = gt_kps
        tracker_keypoints_t = pred_kps
        assert (gt_keypoints_t.shape == tracker_keypoints_t.shape)  # Ensure same shape, missing detections should have empty arrays

        # Compute keypoint distance matrix
        dist_matrix = self.compute_keypoint_distances(gt_keypoints_t, tracker_keypoints_t)
        # Convert distance to similarity (Gaussian similarity) under consideration of confidence scores
        confidence_matrix = data['confidence_matrix'][t]  # ensure that uncertain keypoints don’t dominate matching, leave out if confidence too noisy
        similarity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2)) * confidence_matrix  # lower distances gives higher similarity score

        return similarity

    def compute_keypoint_distances(self, gt_keypoints, pred_keypoints):
        """
        Compute Euclidean distances between sets of keypoints.
        :param gt_keypoints: (N, K, 2) array of GT objects with K keypoints.
        :param pred_keypoints: (M, K, 2) array of predicted objects with K keypoints.
        :return: (N, M) distance matrix (average keypoint distance per object).
        """
        N, K, _ = gt_keypoints.shape  # N GT objects, K keypoints per object
        M, _, _ = pred_keypoints.shape  # M predicted objects, K keypoints per object

        dist_matrix = np.zeros((N, M))  # Create an empty distance matrix (N, M) to store the average keypoint distances between each pair of GT and predicted objects.
        for i in range(N):
            for j in range(M):
                # Compute mean Euclidean distance across keypoints
                dist_matrix[i, j] = np.mean(np.linalg.norm(gt_keypoints[i] - pred_keypoints[j], axis=1))

        return dist_matrix