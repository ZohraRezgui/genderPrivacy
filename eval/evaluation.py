import logging
import os
import argparse
import torch
import sys
import numpy as np
import csv
import glob
from evaluation_utils import get_model, calculate_pic, read_metrics_prepivacy
sys.path.append('/home/rezguiz/genderPrivacy')

from utils.utils_callbacks import CallBackEvaluation
from utils.utils_logging import init_logging
from config.config import config as cfg

def evaluate_model(log_root, model_root, experiment_name, reference_pth):
    
    os.makedirs(os.path.join(log_root, experiment_name), exist_ok=True)
    experiment_folder = os.path.join(model_root, experiment_name)
    all_experiments = os.listdir(experiment_folder)
    device = torch.device('cuda:0')
    root_logger = logging.getLogger()
    init_logging(root_logger, os.path.join(log_root, experiment_name), 'evaluate.log')
    callback_eval = CallBackEvaluation(-1, cfg.test_targets, cfg.rec)
    for exp in all_experiments:
        output_folder = os.path.join(experiment_folder, exp)

        logging.info(f'Evaluating-{exp}')

        csv_path = os.path.join(log_root, experiment_name, exp + "_metrics.csv")
        weights= [os.path.basename(w) for w in glob.glob(output_folder +  "/*ftn_layers.pth")]
        print(output_folder)
        for w in weights:
            net = get_model(cfg.output_ori, cfg.network, os.path.join(output_folder, w),device)
            step = int(w.split("backbone")[0])
            eers, tars, gender_baccs = callback_eval(step, net, forced=True, get_metrics=True)
            header = ['Step', 'Dataset', 'EER', 'TAR@FAR=0.1%', 'BACC-Gender-LogReg', 'BACC-Gender-SVM', 'BACC-Gender-RBF','BACC-Gender-AVG', 'PIC']
                
            with open(csv_path, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for k, ds in enumerate(cfg.test_targets):
                    pre_privacy = read_metrics_prepivacy(ds, reference_pth)
                    eer = eers[k]
                    tar = tars[k]
                    gender_baccs_ds = gender_baccs[k]

                    next_row = [w, ds, eer, tar]
                    for _, classifier in enumerate(gender_baccs_ds.keys()):
                        logging.info("Average CV balanced accuracy on [%s] - [%s]:[%1.5f]" % (ds, classifier, gender_baccs_ds[classifier]))
                        next_row.append(gender_baccs_ds[classifier])
                    avg_bacc = np.mean([*gender_baccs_ds.values()]) 
                    next_row.append(avg_bacc)

                    pic = calculate_pic(float(pre_privacy['EER']), eer, float(pre_privacy['BACC-Gender-AVG']), avg_bacc)
                    next_row.append(pic)
                    writer.writerow(next_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument("--log-root", required=True, help="Root directory for logs.")
    parser.add_argument("--model-root", default="no/path", required=True, help="Root directory for models.")
    parser.add_argument("--experiment-name", required=True, help="Name of the experiment.")
    parser.add_argument("--reference-pth", default="", help="Path to the reference csv results.")
    args = parser.parse_args()

    evaluate_model(args.log_root, args.model_root, args.experiment_name, args.reference_pth)