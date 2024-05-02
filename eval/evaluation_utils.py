

import numpy as np
import os
import sys


import torch
import torch.nn.functional as F
import csv

sys.path.append('/home/rezguiz/genderPrivacy')
from backbones.iresnet import ProjectionLayer, FinetunedModel, iresnet100, iresnet50

from config.config import config as cfg

def read_metrics_prepivacy(ds, reference_pth):
    csv_reader = csv.DictReader(open(reference_pth), delimiter=',')
    for row in csv_reader:
        if row['Dataset'] == ds:
            row['BACC-Gender-AVG'] = np.mean([float(row['BACC-Gender-LogReg']), float(row['BACC-Gender-SVM']), float(row['BACC-Gender-RBF'])])
            return row

def calculate_pic(EER_v_pre, EER_v_post, ACC_g_pre, ACC_g_post):
    gender_supp = ((1.0-ACC_g_post) - (1.0 - ACC_g_pre))/(1.0-ACC_g_pre)
    identity_loss = (EER_v_post - EER_v_pre)/EER_v_pre
    pic = gender_supp - identity_loss
    return pic


def get_model(bck_pth, bck_arch, finetune_pth, device):
    backbone = eval(bck_arch)().to(device)
    weights_bck = torch.load(bck_pth)
    backbone.load_state_dict(weights_bck)
    if os.path.isfile(finetune_pth):
        layer = ProjectionLayer(in_features=512, out_features=512, n_hidden=2).to(device)
        weights_ft = torch.load(finetune_pth)
        layer.load_state_dict(weights_ft)
        net = FinetunedModel(backbone,layer)
    else:
        net = backbone
        net.eval()
    return net
    









