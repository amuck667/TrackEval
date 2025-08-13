import csv
import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DKeypoints.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config1 = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    dataset_config2 = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # hands and tools must be evaluated separately
    sequence_info = { 'P11H': {'length': 1740, 'frame_rate': 29}, 'E66F': {'length': 1827, 'frame_rate': 29}, }
    dataset_config1["SEQ_INFO"] = sequence_info.copy()
    dataset_config1['CLASSES_TO_EVAL'] = ['scissors', 'tweezers', 'needle holder', 'needle']
    dataset_config1['PREFILTER_RAW'] = True
    dataset_config1['OUTPUT_SUB_FOLDER'] = 'tools'
    dataset_config2["SEQ_INFO"] = sequence_info.copy()
    dataset_config2['CLASSES_TO_EVAL'] = ['left hand', 'right hand']
    dataset_config2['PREFILTER_RAW'] = True
    dataset_config2['OUTPUT_SUB_FOLDER'] = 'hands'

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DKeypoints(dataset_config1)]
    dataset_list2 = [trackeval.datasets.MotChallenge2DKeypoints(dataset_config2)]
    metrics_list = []

    if 'HOTA' in metrics_config['METRICS']:
        metrics_list.append(trackeval.metrics.HOTA(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    results = evaluator.evaluate(dataset_list, metrics_list)[0]['MotChallenge2DKeypoints']
    results2 = evaluator.evaluate(dataset_list2, metrics_list)[0]['MotChallenge2DKeypoints']
    # combine results from hands and tools and save to file
    outfile = os.path.join(default_dataset_config['OUTPUT_FOLDER'], 'hota', 'mot_challenge_kp_results.csv')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    combined_seq = {}
    with open(outfile, 'w', newline='') as csvfile:
        fieldnames = ['TRACKER', 'SEQ', 'HOTA']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tracker in results.keys():
            assert tracker in results2
            tracker_results1 = results[tracker]
            tracker_results2 = results2[tracker]
            sequences = tracker_results1.keys()
            assert sequences == tracker_results2.keys(), "Mismatch in sequences between two datasets"
            for seq in sequences:
                combined_seq1 = tracker_results1[seq]
                combined_seq2 = tracker_results2[seq]
                combined_seq = {**combined_seq1, **combined_seq2}
                combined_hota_seq = {cls_key: cls_value['HOTA'] for cls_key, cls_value in combined_seq.items()}
                final_res_all = metrics_list[0].combine_classes_class_averaged(combined_hota_seq)
                result_hota = sum(final_res_all['HOTA'])/len(final_res_all['HOTA'])
                if seq == "COMBINED_SEQ": print(f"Combined HOTA for tracker {tracker}: {result_hota:.4f}")
                writer.writerow({'TRACKER': tracker, 'SEQ': seq, 'HOTA': f"{result_hota}"})
