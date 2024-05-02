import os
import torch
from torch.utils.data import  Dataset
from torchvision import transforms
import cv2



class FaceDataset(Dataset):
    def __init__(self, root_dir):
        super(FaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.imgidx, self.id_labels, self.gender, self.num_classes = self.scan(root_dir, self.attribute)


    def scan(self,root):
            imgidex = []
            labels = []
            gender_attribute = []
            lb=-1
            list_dir = os.listdir(root)
            num_ids = len(list_dir)
            list_dir.sort()
            for id_dir  in list_dir:
                images=os.listdir(os.path.join(root,id_dir))
                lb += 1
                gender = id_dir.split('_')[-1]
                for img in images:
                    imgidex.append(os.path.join(id_dir,img))
                    labels.append(lb)
                    gender_attribute.append(gender)
            return imgidex, labels, gender_attribute, num_ids

    def readImage(self,path):
            return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.readImage(path)
        label = self.id_labels[index]
        gender = self.gender[index]
        label = torch.tensor(label, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)

        sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, gender

    def __len__(self):
        return len(self.imgidx)
        
