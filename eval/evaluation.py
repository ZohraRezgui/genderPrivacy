import logging
import os
import argparse
import torch
import numpy as np
import csv
from evaluation_utils import _get_model, _get_model_ft, get_data, evaluate_gender, calculate_pic, read_metrics_prepivacy
from utils.utils_callbacks import CallBackVerificationFT, CallBackVerification
from utils.utils_logging import init_logging
from config.config import config as cfg

def evaluate_model(log_root, model_root, experiment_name, reference_pth, pretrained_pth, ft=True, gpu_id=0):
    # Set up experiment folder and logging
    os.makedirs(os.path.join(log_root, experiment_name), exist_ok=True)
    experiment_folder = os.path.join(model_root, experiment_name)
    all_models = os.listdir(experiment_folder)

    for m in all_models:
        log_root = logging.getLogger()
        init_logging(log_root, 0, experiment_folder, 'evaluate.log')

        logging.info(f'Evaluating-{m}')
        if experiment_name == 'ReferenceEvaluations':
            output_folder = experiment_folder
        else:
            output_folder = os.path.join(experiment_folder, m)

        csv_path = os.path.join(log_root, experiment_name, m + "_metrics.csv")
        if ft:
            callback_verification = CallBackVerificationFT(1, cfg.test_targets, cfg.rec)
        else:
            callback_verification = CallBackVerification(1, cfg.test_targets, cfg.rec)

        weights = sorted(os.listdir(output_folder))
        res_dict = {}
        for w in weights:
            if "backbone" in w:

                if ft:
                    frozen, layer = _get_model_ft(gpu_id, os.path.join(output_folder, w), pretrained_pth)
                    modelb = torch.nn.DataParallel(frozen, device_ids=[gpu_id])
                    fc = torch.nn.DataParallel(layer, device_ids=[gpu_id])

                    for ds in cfg.test_targets:
                        img_files, id_labels, gender_labels = get_data(ds)

                        accuracies, balanced = evaluate_gender(img_files, gender_labels, id_labels, net=modelb, ft_layer=fc, ft=ft)
                        res_dict[ds] = [accuracies, balanced]

                    step = int(w.split("backbone")[0])
                    eers = callback_verification(step, modelb, fc, 0, forced=True, get_metrics=True)

                    header = ['Step', 'Dataset', 'EER', 'ACC-Gender-LogReg', 'BACC-Gender-LogReg', 'ACC-Gender-SVM', 'BACC-Gender-SVM', 'ACC-Gender-RBF', 'BACC-Gender-RBF', 'PIC']
                    
                    with open(csv_path, 'a', encoding='UTF8') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)

                        for k, ds in enumerate(res_dict.keys()):
                            pre_privacy = read_metrics_prepivacy(ds, reference_pth)
                            eer = eers[k]

                            next_row = [w, ds, eer]
                            bacc_genders = []
                            for i, classifier in enumerate(res_dict[ds][0].keys()):
                                logging.info("Average CV B-ACC {} - {}:{}".format(ds, classifier, res_dict[ds][1][classifier]))
                                next_row.append(res_dict[ds][0][classifier])
                                next_row.append(res_dict[ds][1][classifier])
                                bacc_genders.append(res_dict[ds][1][classifier])
                            bacc_gender_avg = np.mean(bacc_genders)

                            pic = calculate_pic(float(pre_privacy['EER']), eer, float(pre_privacy['BACC-Gender-AVG']), bacc_gender_avg)

                            next_row.append(pic)
                            writer.writerow(next_row)

                else:
                    csv_path = os.path.join(log_root, experiment_name, "metrics.csv")
                    frozen = _get_model(gpu_id, os.path.join(output_folder, w))
                    model = torch.nn.DataParallel(frozen, device_ids=[gpu_id])

                    for ds in cfg.test_targets:
                        img_files, id_labels, gender_labels = get_data(ds)

                        accuracies, balanced = evaluate_gender(img_files, gender_labels, id_labels, net=model, ft=ft)
                        res_dict[ds] = accuracies, balanced

                    step = int(w.split("backbone")[0])
                    eers = callback_verification(step, model, 0, forced=True, get_metrics=True)

                    header = ['Step', 'Dataset', 'EER', 'ACC-Gender-LogReg', 'BACC-Gender-LogReg', 'ACC-Gender-SVM', 'BACC-Gender-SVM', 'ACC-Gender-RBF', 'BACC-Gender-RBF']

                    with open(csv_path, 'a', encoding='UTF8') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)

                        for k, ds in enumerate(res_dict.keys()):
                            eer = eers[k]
                            next_row = [w, ds, eer]
                            for i, classifier in enumerate(res_dict[ds][0].keys()):
                                logging.info("Average CV B-ACC {} - {}:{}".format(ds, classifier, res_dict[ds][1][classifier]))
                                next_row.append(res_dict[ds][0][classifier])
                                next_row.append(res_dict[ds][1][classifier])

                            writer.writerow(next_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    
    parser.add_argument("--log-root", required=True, help="Root directory for logs.")
    parser.add_argument("--model-root", required=True, help="Root directory for models.")
    parser.add_argument("--experiment-name", required=True, help="Name of the experiment.")
    parser.add_argument("--reference-pth", default="", help="Path to the reference csv results.")
    parser.add_argument("--pretrained-pth", default="", help="Path to the pretrained model.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--ft", action="store_true", help="Enable fine-tuning.")

    args = parser.parse_args()

    evaluate_model(args.log_root, args.model_root, args.experiment_name, args.reference_pth, args.pretrained_pth, args.ft, args.gpu_id)