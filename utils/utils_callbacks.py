import logging
import os
import time
from typing import List
import numpy as np
import torch

from eval import verification
from utils.utils_logging import AverageMeter


class CallBackEvaluation(object):
    def __init__(self, frequent, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.eer:  List[float] = []
        self.tar: List[float] = []
        self.gender_baccs: List[dict] = []

        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
       
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, eer, val, far, gender_acc = verification.test(
                self.ver_list[i], backbone, 10, 10, 3)
            self.tar.append(val)
            self.eer.append(eer)
            self.gender_baccs.append(gender_acc)

            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Verif-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Verif-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            logging.info('[%s][%d]Accuracy-Gender-Avg: %1.5f' % (self.ver_name_list[i], global_step, np.mean([*gender_acc.values()])))

            logging.info('[%s][%d]EER: %1.5f' % (self.ver_name_list[i], global_step, eer))
            logging.info('[%s][%d]TAR@FAR = %1.5f : %1.5f' % (self.ver_name_list[i], global_step, far, val))



    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")

            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)
    

    def __call__(self, num_update, backbone: torch.nn.Module, forced: bool=False, get_metrics: bool=False):
        self.eer=[]
        self.tar=[]
        self.gender_baccs = []
        if num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()
            if get_metrics:
                return self.eer, self.tar, self.gender_baccs
        elif forced:
            backbone.eval()
            self.ver_test(backbone, num_update)
            if get_metrics:
                return self.eer, self.tar, self.gender_baccs




class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, resume=0, rem_total_steps=None):
        self.frequent: int = frequent
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size

        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, epoch: int, loss_verif: AverageMeter, loss_privacy: AverageMeter = None):
        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
            
                speed_total: float = self.frequent * self.batch_size / (time.time() - self.tic)


                time_now = (time.time() - self.time_start) / 3600
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now

                
                msg = "Speed %.2f samples/sec   Loss %.4f Loss %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                    speed_total, loss_verif.avg, loss_privacy.avg, epoch, global_step, time_for_end
                )

                logging.info(msg)
                loss_verif.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

            


class CallBackModelCheckpoint(object):
    def __init__(self, output="./"):
        self.output: str = output
    def __call__(self, global_step, ftn_layers: torch.nn.Module, header: torch.nn.Module = None):

        if global_step > 100 :
            torch.save(ftn_layers.state_dict(), os.path.join(self.output, str(global_step)+ "ftn_layers.pth"))
            if header:
                torch.save(header.state_dict(), os.path.join(self.output, str(global_step)+ "header.pth"))

        
