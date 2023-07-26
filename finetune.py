
import csv
import logging
import os
import time
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, sampler
from config.config import config as cfg
from utils import losses

from utils.dataset import VGGDataset, LFWDataset, ColorFeretDataset, AgeDBDataset, TrainFoldsDataset
from utils.utils_callbacks import CallBackVerificationFT, CallBackLogging, CallBackModelCheckpoint
from utils.data_utils import make_weights_for_balanced_classes, get_gender, get_unique_gender
from utils.utils_logging import AverageMeter, init_logging
from backbones.iresnet import  ProjectionLayer, iresnet100, iresnet50

def train():
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
    csvpath = os.path.join(cfg.log_dir,cfg.experiment, cfg.data, 'config.csv')


    # Create output folder if it is not exist
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    elif os.path.exists(cfg.output):
        for file in os.listdir(cfg.output):
            os.remove(os.path.join(cfg.output, file))
        time.sleep(2)

    # Saving hyperparameters in csv
    header =  ['output','TrainData',	'Arcloss-M',	'Arcloss-S'	,'lr-network',	'lr-header',	'epochs', 'alpha', 'beta', 'batchSize']
    file_exists = os.path.isfile(csvpath)
    with open(csvpath, 'a', encoding='UTF8') as f:
        cwriter = csv.writer(f)
        if not file_exists:
            cwriter.writerow(header)
        info = [cfg.output, cfg.data, cfg.m, cfg.s, cfg.lr, cfg.headid_lr, cfg.num_epoch, cfg.alpha, cfg.beta, cfg.batch_size]
        cwriter.writerow(info)

    # intiate logger to write the output of log file
    log_root = logging.getLogger()
    init_logging(log_root,  cfg.log_dir)

    # create instance of dataset
    if cfg.data == "VGGFace2":
        trainset = VGGDataset(root_dir=os.path.join(cfg.data_dir,'samples/vggface2'),attribute=os.path.join(cfg.data_dir,"VGGFace2/metadata/"))
    elif cfg.data=="LFW":
        trainset = LFWDataset(root_dir=os.path.join(cfg.data_dir, "lfw_aligned") ,attribute=os.path.join(cfg.data_dir,"LFW_gender" ))
    elif cfg.data == "ColorFeret":
        trainset= ColorFeretDataset(root_dir=os.path.join(cfg.data_dir, 'ColorFeret_aligned'), attribute=os.path.join(cfg.data_dir,"colorferet/dvd1/data/ground_truths/xml/subjects.xml"))
    elif cfg.data =="AgeDB":
        trainset = AgeDBDataset(root_dir=os.path.join(cfg.data_dir, 'age_db_aligned'))
    elif cfg.data =="LFW-ColorFeret":
        trainset = TrainFoldsDataset(root_dir=os.path.join(cfg.data_dir, 'balanced_train_folds'), fold=0 )
    elif cfg.data == "AgeDB-ColorFeret":
        trainset = TrainFoldsDataset(root_dir=os.path.join(cfg.data_dir, 'balanced_train_folds'), fold=1 )
    elif cfg.data == "LFW-AgeDB":
        trainset = TrainFoldsDataset(root_dir=os.path.join(cfg.data_dir, 'balanced_train_folds'), fold=2 )
    else:
        raise NotImplementedError



    # Creating trainloader with balanced batches
    weights = make_weights_for_balanced_classes(trainset.imgidx, trainset.gender_attribute, nclasses=2)  
    weights = torch.DoubleTensor(weights)                                       
    trainsampler = sampler.WeightedRandomSampler(weights, len(weights))    
    train_loader = DataLoader(dataset=trainset, batch_size=cfg.batch_size, sampler=trainsampler, num_workers=16, pin_memory=True, drop_last=True)

    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(device)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(device)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()
    try:
        projection = ProjectionLayer(in_features=cfg.embedding_size, out_features=cfg.embedding_size, n_hidden=2).to(device)
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("Error initializing network !")
    label_f , label_m = get_unique_gender(trainset.labels, trainset.gender_attribute)
    
    try:
        backbone_pth = os.path.join(cfg.output_ori, '153250' + "backbone.pth")
        backbone.load_state_dict(torch.load(backbone_pth, map_location=device))
        logging.info("backbone resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("load backbone resume init, failed!")


    # Initialize identity weights and recognition loss type
    header = losses.ArcFace(in_features=cfg.embedding_size, out_features=trainset.num_classes, s=cfg.s, m=cfg.m).to(device)
    constraint_one = losses.P1Loss()
    constraint_two = losses.P2Loss(label_f, label_m)
    criterion = CrossEntropyLoss()


    # initializing optimizers 
    opt_header = torch.optim.Adam(
        params=[{'params': header.parameters()}],
        lr=cfg.headid_lr)
    opt_layer = torch.optim.Adam(
        params=[{'params': projection.parameters()}],
        lr=cfg.lr)
       
    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)
 
    logging.info("Total Step is: %d" % total_step)



    # verification, logging and checkpoint
    callback_verification = CallBackVerificationFT(cfg.eval_step, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(120, total_step, cfg.batch_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(cfg.output)


    loss_r_meter = AverageMeter()
    loss_p_meter=AverageMeter()

    global_step = cfg.global_step

    # freezing backbone and setting the finetuning layers and identity class weights to train mode
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    header.train()
    projection.train()


    print("Number of iterations {}".format(len(train_loader)))

    for epoch in range(start_epoch, cfg.num_epoch):

        for i, (img, label, gender) in enumerate(train_loader):
            global_step += 1
            projection.zero_grad()
            header.zero_grad()

            img = img.to(device)
            label = label.to(device)
            gender = gender.to(device)

            features = F.normalize(backbone(img))
            proj_features = F.normalize(projection(features))


            # ----------- RECOGNITION LOSS--------------
           

            thetas_r = header(proj_features, label)
            loss_r = criterion(thetas_r, label.detach())
                
                   
            male_idx, female_idx = get_gender(gender=gender.detach())


            #--------------PRIVACYLOSS--------------
                
            loss_p1 = constraint_one(proj_features, female_idx, male_idx)
            loss_p2 = constraint_two(header, proj_features, female_idx, male_idx)
            loss_p = cfg.alpha*loss_p1 + cfg.beta*loss_p2
            loss_t =  loss_r + loss_p 


            loss_t.backward()
            clip_grad_norm_(projection.parameters(), max_norm=5, norm_type=2)


            opt_header.step()
            opt_layer.step()


            loss_r_meter.update(loss_t.item(), 1)         
            loss_p_meter.update(loss_p.item(), 1)       
            
            callback_logging(global_step, epoch, loss_r_meter, loss_p_meter)
            callback_verification(global_step, backbone, projection, epoch)


        
        callback_checkpoint(global_step, projection, header)


if __name__ == "__main__":

    train()

