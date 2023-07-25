import os
import csv

import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import  Dataset
from torchvision import transforms
import cv2




        
class VGGDataset(Dataset):
    def __init__(self, root_dir, attribute):
        super(VGGDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.attribute_dir=attribute
        self.atribute=self.getGender(self.attribute_dir)
        self.imgidx, self.labels, self.gender_attribute, self.num_classes=self.scan(root_dir, self.atribute)

    def getGender(self,attribute_dir):
        att_file = os.path.join(attribute_dir, 'annotations_train.csv')
        att_reader = csv.DictReader(open(att_file), delimiter=',')
        gender_dict={}
        for row in att_reader:
            if row['Male'] == '-1':
                gender_dict[row['Filename'].rstrip()] = 0
            elif row["Male"] == '1':
                gender_dict[row['Filename'].rstrip()] = 1
        return gender_dict

    def scan(self,root, attributes):
        imgidex=[]
        labels=[]
        gender_attribute=[]
        lb=-1
        list_dir=os.listdir(root)
        num_ids = len(list_dir)
        list_dir.sort()
        count=0
        for l in list_dir:
            images=os.listdir(os.path.join(root,l))
            lb += 1
            for img in images:
                if (os.path.join(l,img) in attributes.keys()):
                    imgidex.append(os.path.join(l,img))
                    labels.append(lb)
                    gender_attribute.append(attributes[os.path.join(l,img)])
                else:
                    count+=1
        print("Number of skipped images"+ str(count))
        return imgidex,labels, gender_attribute, num_ids

    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img=self.readImage(path)
        label = self.labels[index]
        gender=self.gender_attribute[index]
        label = torch.tensor(label, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)

        sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, gender

    def __len__(self):
        return len(self.imgidx)


class AgeDBDataset(Dataset):
    def __init__(self, root_dir):
        super(AgeDBDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.imgidx, self.labels, self.gender_attribute, self.num_classes=self.scan(root_dir)

    def scan(self, img_dir):
        img_names = os.listdir(img_dir)
        img_names.sort()
        count = 0
        imgidex=[]
        labels=[]
        gender_attribute=[]
        id_names ={}
        lb = -1
        for i, im_name in enumerate(img_names):
            imgpth=os.path.join(img_dir, im_name)
            id_l= im_name.split('_')[1]
            if id_l not in id_names.keys():
                lb += 1
                id_names[id_l] = lb
                            

            gender_l = im_name.split('_')[3].split('.jpg')[0]
            if i==0:
                print(gender_l)
            if gender_l=="f":
                imgidex.append(imgpth)
                labels.append(id_names[id_l])
                gender_attribute.append(0)
            elif gender_l=="m":                
                imgidex.append(imgpth)
                labels.append(id_names[id_l])
                gender_attribute.append(1)

            else:
                count+=1
        num_ids = len(id_names.keys())
        print("Images without gender labels : {}".format(count))
        print("Number of Images {}".format(len(imgidex)))
        
        return imgidex, labels, gender_attribute, num_ids


    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img=self.readImage(path)
        label = self.labels[index]
        gender=self.gender_attribute[index]
        label = torch.tensor(label, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)

        sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, gender

    def __len__(self):
        return len(self.imgidx)



class ColorFeretDataset(Dataset):
    def __init__(self, root_dir, attribute):
        super(ColorFeretDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.attribute_dir=attribute
        self.atribute=self.parse_xml_dict(self.attribute_dir, attribute_list=['gender'])
        self.imgidx, self.labels, self.gender_attribute, self.num_classes=self.scan(root_dir, self.atribute)


    def parse_xml_dict(self,xml_pth, attribute_list=['gender']):
        mytree =  ET.parse(xml_pth)
        gen = mytree.iter(tag='Subject')
        gender_dict = {}
        race_dict = {}
        age_dict = {}
        
        for element in gen:
            subject = element.attrib['id'][4:]
            if 'gender' in attribute_list:
                gender = element[0].attrib['value']
                if gender == 'Male':
                    gender_dict[subject] = 1
                else:
                    gender_dict[subject] = 0
            if 'age' in attribute_list:
                age = element[1].attrib['value']
                age_dict[subject] = age
            if 'race' in attribute_list:
                race = element[2].attrib['value']
                race_dict[subject]=race
        return gender_dict
             
    def scan(self,root, attributes):
        imgidex=[]
        labels=[]
        gender_attribute=[]
        lb=-1
        list_dir=os.listdir(root)
        num_ids = len(list_dir)
        list_dir.sort()
        count=0
        for l in list_dir:
            images=os.listdir(os.path.join(root,l))
            lb += 1
            for img in images:
                if l in attributes.keys():
                    imgidex.append(os.path.join(l,img))
                    labels.append(lb)
                    gender_attribute.append(attributes[l])
                else:
                    # print("file to be skipped",os.path.join(l,img))
                    count+=1
        print("Number of skipped images"+ str(count))
        return imgidex,labels, gender_attribute, num_ids

    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img=self.readImage(path)
        label = self.labels[index]
        gender=self.gender_attribute[index]
        label = torch.tensor(label, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)

        sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, gender

    def __len__(self):
        return len(self.imgidx)

class LFWDataset(Dataset):
    def __init__(self, root_dir, attribute):
        super(LFWDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        self.root_dir = root_dir
        self.attribute_dir = attribute
        self.attribute = self.getGenderLFW(self.attribute_dir)
        self.imgidx, self.labels, self.gender_attribute, self.num_classes=self.scan(root_dir, self.attribute)
    
    def getGenderLFW(self, attribute_dir):   

        att_file_f = os.path.join(attribute_dir, 'female_names.txt')
        att_file_m = os.path.join(attribute_dir, 'male_names.txt')
        gender_dict = {}
        id_folders = os.listdir(self.root_dir)
        with(open(att_file_f, 'r')) as file:
            female = file.readlines()
            female = [n.strip() for n in female]
            female_id = [f.split('.jpg')[0][:-5] for f in female]
        with(open(att_file_m, 'r')) as file:
            male = file.readlines()
            male = [n.strip() for n in male]
            male_id = [m.split('.jpg')[0][:-5] for m in male]        
        # print("female_IDs", female_id[:5])
        for id in id_folders:
            img_folder = os.listdir(os.path.join(self.root_dir, id))
            for img_name in img_folder:
                
                if id in female_id:
                    gender_dict[img_name] = 0
                elif id in male_id:                    
                    gender_dict[img_name] = 1                    
                elif id == "Tara_Kirk": # originally was not labelled. FIXED
                    gender_dict[img_name] = 0

        print(list(gender_dict.items())[:20]) 
        return gender_dict

    def scan(self,root, attributes):
        imgidex=[]
        labels=[]
        gender_attribute=[]
        lb=-1
        list_dir=os.listdir(root)
        num_ids = len(list_dir)
        list_dir.sort()
        count=0
        for l in list_dir:
            images=os.listdir(os.path.join(root,l))
            lb += 1
            for img in images:
                if img in attributes.keys():
                    imgidex.append(os.path.join(l,img))
                    labels.append(lb)
                    gender_attribute.append(attributes[img])
                else:
                    print("file to be skipped",os.path.join(l,img))
                    count+=1
        print("Number of skipped images"+ str(count))
        return imgidex,labels, gender_attribute, num_ids

    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img= self.readImage(path)
        label = self.labels[index]
        gender= self.gender_attribute[index]
        label = torch.tensor(label, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)

        sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, gender


    
    def __len__(self):
        return len(self.imgidx)

class TrainFoldsDataset(Dataset):
    def __init__(self, root_dir, fold):
        super(TrainFoldsDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        self.root_dir = root_dir
        self.fold = fold
        self.fold_dict = self.define_folds()
        self.imgidx, self.labels, self.gender_attribute, self.num_classes=self.scan(self.root_dir, self.fold)
        print(np.unique(self.gender_attribute))

    def define_folds(self):
        # fold_evalagedb = ['lfw']
        fold_evalagedb = ['lfw', 'colorferet']
        fold_evallfw = ['agedb', 'colorferet']
        fold_evalcolorferet = ['lfw', 'agedb']
        fold_evalvgg = ['lfw', 'agedb', 'colorferet']

        dict_folds = {0:fold_evalagedb, 1:fold_evallfw, 2:fold_evalcolorferet, 3:fold_evalvgg}
        return dict_folds
    
    def scan(self, root_dir, fold):
        imgidex=[]
        labels=[]
        gender_attribute=[]
        lb = -1
        all_ids = os.listdir(root_dir)
        count = 0
        male_ids =0
        female_ids = 0
        for id_folder in all_ids:
            ds_src = id_folder.split('_')[-1]
            if ds_src in self.fold_dict[fold]:
                lb +=1
                gender = int(id_folder.split('_')[-2])
                if gender == 1:
                    male_ids+=1
                elif gender==0:
                    female_ids+=1
                else:
                    print(gender)
                id_folder_pth = os.path.join(root_dir, id_folder)
                img_names = os.listdir(id_folder_pth)
                for im in img_names:
                    imgidex.append(os.path.join(id_folder_pth, im))
                    gender_attribute.append(gender)
                    labels.append(lb)
            else:
                count+=1
        n_ids = len(all_ids) - count
        print(f"Fold {fold} - Number of Identities = {n_ids}")
        print(f'Number of Images = {len(imgidex)}')
        print(f'Number of M identities = {male_ids}')
        print(f'Number of F identities = {female_ids}')
        print(f'Number of M images = {len([i for i in gender_attribute if i==1])}')
        print(f'Number of F images = {len([i for i in gender_attribute if i==0])}')
        return imgidex, labels, gender_attribute, n_ids
    
    
    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img= self.readImage(path)
        label = self.labels[index]
        gender= self.gender_attribute[index]
        label = torch.tensor(label, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)

        sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, gender


    
    def __len__(self):
        return len(self.imgidx)




