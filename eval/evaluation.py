import logging
import os
import argparse
import torch
import sys
import numpy as np
import csv
import glob
from evaluation_utils import _get_model, _get_model_ft,load_test_data,  get_data, evaluate_gender, calculate_pic, read_metrics_prepivacy
sys.path.append('/home/rezguiz/genderPrivacy')

from utils.data_utils import  create_directory
from utils.utils_callbacks import CallBackVerificationFT, CallBackVerification
from utils.utils_logging import init_logging
from config.config import config as cfg

def evaluate_model(log_root, model_root, experiment_name, reference_pth, ft=True, gpu_id=0):
    # Set up experiment folder and logging
    os.makedirs(os.path.join(log_root, experiment_name), exist_ok=True)
    experiment_folder = os.path.join(model_root, experiment_name)
    all_models = os.listdir(experiment_folder)
    data_dict = load_test_data()


    for m in all_models:

        if ft == False:
            output_folder = experiment_folder
        else:
            output_folder = os.path.join(experiment_folder, m)

        root_logger = logging.getLogger()
        init_logging(root_logger, experiment_folder, 'evaluate.log')

        logging.info(f'Evaluating-{m}')

        csv_path = os.path.join(log_root, experiment_name, m + "_metrics.csv")
        if ft:
            callback_verification = CallBackVerificationFT(1, cfg.test_targets, cfg.rec)
        else:
            callback_verification = CallBackVerification(1, cfg.test_targets, cfg.rec)

        weights= [os.path.basename(w) for w in glob.glob(output_folder +  "/*backbone.pth")]
        res_dict = {}
        for w in weights:
                
            if ft:
                print("Finetuned model")
                frozen, layer = _get_model_ft(gpu_id, os.path.join(output_folder, w), os.path.join(cfg.output_ori, "backbone.pth"), cfg.network)
                modelb = torch.nn.DataParallel(frozen, device_ids=[gpu_id])
                fc = torch.nn.DataParallel(layer, device_ids=[gpu_id])

                for ds in cfg.test_targets:
                    img_files, id_labels, gender_labels = get_data(ds, data_dict)

                    acc_bacc_res = evaluate_gender(img_files, gender_labels, id_labels, net=modelb, ft_layer=fc)
                    res_dict[ds] = acc_bacc_res

                step = int(w.split("backbone")[0])
                eers, tars = callback_verification(step, modelb, fc, 0, forced=True, get_metrics=True)

                header = ['Step', 'Dataset', 'EER', 'TAR@FAR=0.1%', 'ACC-Gender-LogReg', 'BACC-Gender-LogReg', 'ACC-Gender-SVM', 'BACC-Gender-SVM', 'ACC-Gender-RBF', 'BACC-Gender-RBF','BACC-Gender-AVG', 'PIC']
                
                with open(csv_path, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                    for k, ds in enumerate(res_dict.keys()):
                        pre_privacy = read_metrics_prepivacy(ds, reference_pth)
                        eer = eers[k]
                        tar = tars[k]

                        next_row = [w, ds, eer, tar]
                        bacc_genders = []
                        for i, classifier in enumerate(res_dict[ds].keys()):
                            logging.info("Average CV B-ACC {} - {}:{}".format(ds, classifier, res_dict[ds][classifier]['balanced']))
                            next_row.append(res_dict[ds][classifier]['accuracies'])
                            next_row.append(res_dict[ds][classifier]['balanced'])
                            bacc_genders.append(res_dict[ds][classifier]['balanced'])
                        bacc_gender_avg = np.mean(bacc_genders)
                        next_row.append(bacc_gender_avg)

                        pic = calculate_pic(float(pre_privacy['EER']), eer, float(pre_privacy['BACC-Gender-AVG']), bacc_gender_avg)

                        next_row.append(pic)
                        writer.writerow(next_row)

            else:
                print("Reference Model")
                csv_path = os.path.join(log_root, experiment_name, "metrics.csv")
                frozen = _get_model(gpu_id, os.path.join(output_folder, w), cfg.network)
                model = torch.nn.DataParallel(frozen, device_ids=[gpu_id])

                for ds in cfg.test_targets:
                    img_files, id_labels, gender_labels = get_data(ds, data_dict)

                    acc_bacc_res = evaluate_gender(img_files, gender_labels, id_labels, net=model)
                    res_dict[ds] = acc_bacc_res
                try:
                    step = int(w.split("backbone")[0])
                except:
                    step = 0
                eers, tars = callback_verification(step, model, 0, forced=True, get_metrics=True)
                

                header = ['Step', 'Dataset', 'EER','TAR@FAR=0.1%', 'ACC-Gender-LogReg', 'BACC-Gender-LogReg', 'ACC-Gender-SVM', 'BACC-Gender-SVM', 'ACC-Gender-RBF', 'BACC-Gender-RBF','BACC-Gender-AVG']

                with open(csv_path, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                    for k, ds in enumerate(res_dict.keys()):
                        eer = eers[k]
                        tar = tars[k]

                        next_row = [w, ds, eer, tar]
                        bacc_genders = []
                        for i, classifier in enumerate(res_dict[ds].keys()):
                            logging.info("Average CV B-ACC {} - {}:{}".format(ds, classifier, res_dict[ds][classifier]['balanced']))
                            next_row.append(res_dict[ds][classifier]['accuracies'])
                            next_row.append(res_dict[ds][classifier]['balanced'])
                            bacc_genders.append(res_dict[ds][classifier]['balanced'])
                        bacc_gender_avg = np.mean(bacc_genders)
                        next_row.append(bacc_gender_avg)
                        writer.writerow(next_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    
    parser.add_argument("--log-root", required=True, help="Root directory for logs.")
    parser.add_argument("--model-root", required=True, help="Root directory for models.")
    parser.add_argument("--experiment-name", required=True, help="Name of the experiment.")
    parser.add_argument("--reference-pth", default="", help="Path to the reference csv results.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--ft", default=True, help="Enable fine-tuning.")

    args = parser.parse_args()

    evaluate_model(args.log_root, args.model_root, args.experiment_name, args.reference_pth, args.ft, args.gpu_id)