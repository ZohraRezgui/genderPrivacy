
from sklearn.preprocessing import normalize

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

import numpy as np
import os
import sys

import cv2
import torch
import torch.nn.functional as F
import csv
import xml.etree.ElementTree as ET
from collections import namedtuple
sys.path.append('/home/rezguiz/genderPrivacy')
from backbones.iresnet import ProjectionLayer, iresnet50, iresnet100
from backbones.sfnet import sfnet20

from config.config import config as cfg

def read_metrics_prepivacy(ds, reference_pth):
    csv_reader = csv.DictReader(open(reference_pth), delimiter=',')
    for row in csv_reader:
        if row['Dataset'] == ds:
            row['BACC-Gender-AVG'] = np.mean([float(row['BACC-Gender-LogReg']), float(row['BACC-Gender-SVM']), float(row['BACC-Gender-RBF'])])
            return row

def calculate_pic(EER_v_pre, EER_v_post, ACC_g_pre, ACC_g_post):
    gender_supp = ((1-ACC_g_post) - (1 - ACC_g_pre))/ (1-ACC_g_pre)
    identity_loss = (EER_v_post - EER_v_pre)/EER_v_pre
    pic = gender_supp - identity_loss
    return pic

def _get_model(ctx, model_path, net):
    if 'backbone' in os.path.basename(model_path):
        weight = torch.load(model_path)
    else:
        weight = torch.load(model_path)['backbone']
    if net== "iresnet50":
        backbone = iresnet50().to(f"cuda:{ctx}")
    elif net == "iresnet100":
        backbone = iresnet100().to(f"cuda:{ctx}")
    elif net == "sfnet20":
        backbone = sfnet20().to(f"cuda:{ctx}")

    try:
        backbone.load_state_dict(weight)
    except:
        for key in list(weight.keys()):
            weight[key.replace('module.', '')] = weight.pop(key)
        backbone.load_state_dict(weight)

    backbone.eval()
    return backbone

def _get_model_ft(ctx, model_path, pretrained_pth, net):
    if 'backbone' in os.path.basename(pretrained_pth):
        weight = torch.load(pretrained_pth)
    else:
        weight = torch.load(pretrained_pth)['backbone']
    if 'backbone' in os.path.basename(model_path):
        weight_proj = torch.load(model_path)
    else:
        weight_proj = torch.load(model_path)['backbone']
    if net== "iresnet50":
        frozen = iresnet50().to(f"cuda:{ctx}")
    elif net == "iresnet100":
        frozen = iresnet100().to(f"cuda:{ctx}")
    elif net == "sfnet20":
        frozen = sfnet20().to(f"cuda:{ctx}")
    project = ProjectionLayer(in_features=cfg.embedding_size, out_features=cfg.embedding_size, n_hidden=2).to(f"cuda:{ctx}")
    try:
        frozen.load_state_dict(weight)
        print("backbone loaded !")
    except:
        for key in list(weight.keys()):
            weight[key.replace('module.', '')] = weight.pop(key)
        frozen.load_state_dict(weight)

    try:
        project.load_state_dict(weight_proj)
        print("projection layer loaded !")
    except:
        for key in list(weight_proj.keys()):
            weight_proj[key.replace('module.', '')] = weight_proj.pop(key)
        project.load_state_dict(weight_proj)


    frozen.eval()
    project.eval()
    return frozen, project


def read_lfw_data(img_dir, attribute_dir):
    id_folders = os.listdir(img_dir)
    with(open(os.path.join(attribute_dir, 'female_names.txt'), 'r')) as f:
        female = f.readlines()
        female = [n.strip() for n in female]
        female_id = [f.split('.jpg')[0][:-5] for f in female]
    with(open(os.path.join(attribute_dir, 'male_names.txt'), 'r')) as f:
        male = f.readlines()
        male = [n.strip() for n in male]
        male_id = [m.split('.jpg')[0][:-5] for m in male]
    
    id_label = []
    gender= []
    img_file = []
    for id in id_folders:
        img_folder = os.listdir(os.path.join(img_dir, id))
        for img_name in img_folder:
            img_file.append(os.path.join(img_dir,id, img_name))
            # img_file.append(img_name)
            id_label.append(id)
            if id in female_id:
                gender.append(0)
            elif id in male_id:
                gender.append(1)
            elif id == "Tara_Kirk": # originally was not labelled. FIXED
                gender.append(0)

    id_label = np.array(id_label)
    gender = np.array(gender)

    print("loaded lfw for gender classification...")
    return img_file, id_label, gender



def read_agedb_data(img_dir):
    img_names = os.listdir(img_dir)
    count = 0
    id_label = []
    gender= []
    img_file = []
    for i, im_name in enumerate(img_names):
        imgpth=os.path.join(img_dir, im_name)
        id_l= im_name.split('_')[1]
        gender_l = im_name.split('_')[3].split('.jpg')[0]
        if i==0:
            print(gender_l)
        if gender_l=="f":
            img_file.append(imgpth)
            id_label.append(id_l)

            gender.append(0)
        elif gender_l=="m":
            img_file.append(imgpth)
            id_label.append(id_l)
            gender.append(1)

        else:
            count+=1

    id_label = np.array(id_label)
    gender = np.array(gender)
    print("loaded agedb for gender classification...")
    return img_file, id_label, gender

def read_vggface2_data(root_dir,attribute_dir):
    lb = 0
    count = 0
    id_label = []
    gender= []
    img_file = []
    att_file = os.path.join(attribute_dir, 'annotations_train.csv')
    att_reader = csv.DictReader(open(att_file), delimiter=',')
    gender_dict={}
    for row in att_reader:
        if row['Male'] == '-1':
            gender_dict[row['Filename'].rstrip()] = 0
        elif row["Male"] == '1':
            gender_dict[row['Filename'].rstrip()] = 1
    dir_list = os.listdir(root_dir)
    dir_list.sort()

    for id_dir in dir_list:
        lb +=1
        images=os.listdir(os.path.join(root_dir,id_dir))
        for img in images:
            if (os.path.join(id_dir,img) in gender_dict.keys()):
                img_file.append(os.path.join(root_dir,id_dir,img))
                id_label.append(lb)
                gender.append(gender_dict[os.path.join(id_dir,img)])
            else:
                count+=1
    
    id_label = np.array(id_label)
    gender = np.array(gender)
    print("loaded vggface2 for gender classification...")

    return img_file, id_label, gender

def read_colorferet_data(root_dir, attribute_pth):
    lb = 0
    id_label = []
    gender= []
    img_file = []
    mytree =  ET.parse(attribute_pth)
    gen = mytree.iter(tag='Subject')
    gender_dict = {}

    for element in gen:
        subject = element.attrib['id'][4:]
        sex = element[0].attrib['value']
        if sex == "Female":

            gender_dict[subject] = 0
        else:
            gender_dict[subject] = 1
    
    dir_list = os.listdir(root_dir)
    dir_list.sort()

    for id_dir in dir_list:
        lb +=1
        if id_dir in  gender_dict.keys():
            images=os.listdir(os.path.join(root_dir,id_dir))
            for img in images:
                
                img_file.append(os.path.join(root_dir,id_dir,img))
                id_label.append(lb)
                gender.append(gender_dict[id_dir])
    
    id_label = np.array(id_label)
    gender = np.array(gender)  
    print("loaded colorferet for gender classification...")
          
    return img_file, id_label, gender


def load_csv(filename: str, header: int = None,):
    CSV = namedtuple("CSV", ["header", "index", "data"])

    with open(os.path.join(filename)) as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

    if header is not None:
        headers = data[header]
        data = data[header + 1 :]
    else:
        headers = []

    indices = [row[0] for row in data]
    data = [row[1:] for row in data]
    data_int = [list(map(int, i)) for i in data]

    return CSV(headers, indices, torch.tensor(data_int))



@torch.no_grad()
def _getFeatureBlob(input_blob, model):
    imgs = torch.Tensor(input_blob).cuda()
    imgs.div_(255).sub_(0.5).div_(0.5)
    feat = model(imgs)
    return feat.cpu().numpy()
    

@torch.no_grad()
def _getFeatureBlob_ft(input_blob, model, ft_layer):
    imgs = torch.Tensor(input_blob).cuda()
    imgs.div_(255).sub_(0.5).div_(0.5)
    featb = model(imgs)
    featb =F.normalize(featb)
    feat = ft_layer(featb)
    return feat.cpu().numpy()

@torch.no_grad()
def get_batch_feature(image_path_list, model, batch_size=64, ft_layer=None):

    count = 0
    num_batch = int(len(image_path_list) / batch_size)
    features = []
    for i in range(0, len(image_path_list), batch_size):
        if count < num_batch:
            tmp_list = image_path_list[i: i + batch_size]
        else:
            tmp_list = image_path_list[i:]
        count += 1
        images = []
        for image_path in tmp_list:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (112, 112))
            a = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            a = np.transpose(a, (2, 0, 1))
            images.append(a)

        if ft_layer is None:
            emb =  _getFeatureBlob(images, model)
        else:
            emb = _getFeatureBlob_ft(images, model, ft_layer)
        features.append(emb)
        # print("batch" + str(i))

    features = np.vstack(features)
    features = normalize(features)
    print('output features:', features.shape)
    return features

def load_test_data():
    data_readers = {
        "lfw": read_lfw_data(os.path.join(cfg.data_dir, "lfw_aligned"), os.path.join(cfg.data_dir,'LFW_gender')),
        "agedb_30": read_agedb_data(os.path.join(cfg.data_dir, 'age_db_mtcnn')),
        "custom_agedb30": read_agedb_data(os.path.join(cfg.data_dir, 'age_db_mtcnn')),
        "vggface2": read_vggface2_data(os.path.join(cfg.data_dir,'samples/vggface2'), os.path.join(cfg.data_dir,"VGGFace2/metadata/")),
        "colorferet": read_colorferet_data(os.path.join(cfg.data_dir, 'ColorFeret/ColorFeret_aligned'), os.path.join(cfg.data_dir,"colorferet/dvd1/data/ground_truths/xml/subjects.xml"))
    }
    return data_readers

def get_data(test_set, data_readers):
    if test_set in data_readers:
        img_files, id_labels, gender_labels = data_readers[test_set]
    else:
        raise NotImplementedError(f"Data loading for '{test_set}' is not implemented.")

    return img_files, id_labels, gender_labels

@torch.no_grad()
def evaluate_gender(img_files, gender_labels, id_labels, net=None, ft_layer=None):
    
    features = get_batch_feature(img_files, net, 64, ft_layer)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=False)

    models = {
        'LogReg': LogisticRegression(),
        'SVM': SVC(kernel='linear', probability=True),
        'RBF': SVC(kernel='rbf', probability=True)
    }

    results = {model_name: {'accuracies': [], 'balanced': []} for model_name in models}

    for i, (train_idxs, test_idxs) in enumerate(cv.split(img_files, gender_labels, id_labels)):
        for model_name, model in models.items():
            model_fit = model.fit(features[train_idxs], gender_labels[train_idxs])
            model_acc = model_fit.score(features[test_idxs], gender_labels[test_idxs])
            model_preds = model_fit.predict(features[test_idxs])
            model_balanced_acc = balanced_accuracy_score(gender_labels[test_idxs], model_preds)

            results[model_name]['accuracies'].append(model_acc)
            results[model_name]['balanced'].append(model_balanced_acc)

    for model_name in results.keys():
        results[model_name]['accuracies'] = sum(results[model_name]['accuracies']) / len(results[model_name]['accuracies'])
        results[model_name]['balanced'] = sum(results[model_name]['balanced']) / len(results[model_name]['balanced'])

    return results



# def check_folds(folds): # folds = {0: {"data": .. , "id_label": .. , "gender": ...}, 1:{..},..., n_fold:{...}}
#     for i in folds.keys():
#         # Check if ids are not overlapping in the folds
#         for j in folds.keys():
#             if i !=j:
#                 assert len(np.intersect1d(folds[i]['id'], folds[j]['id'])) == 0 , "IDs overlapping in fold {} and {}".format(i, j)
                
#         print("------Stats for fold {} : ------".format(i))
#         print("Number of images : {}".format(len(folds[i]['data'])))
#         print("Number of ids : {}".format(len(np.unique(folds[i]['id']))))
#         print("Number of female images: {}".format((folds[i]['gender'] == 0).sum()))
#         print("Number of male images: {}".format((folds[i]['gender'] == 1).sum()))

#         f_ids = folds[i]['id'][np.where(folds[i]['gender'] == 0)]
#         m_ids = folds[i]['id'][np.where(folds[i]['gender'] == 1)]
#         print("Number of female ids : {}".format(len(np.unique(f_ids))))
#         print("Number of male ids: {}".format(len(np.unique(m_ids))))


