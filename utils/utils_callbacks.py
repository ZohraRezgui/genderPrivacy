import logging
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter


class CallBackVerification(object):
    def __init__(self, frequent, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.eer:  List[float] = []
        self.img_list : List[str] = []
        self.gender_list : List[str] = []
        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int, epoch: int, writer=None):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, eer, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
           
            self.eer.append(eer)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            logging.info('[%s][%d]EER: %1.5f' % (self.ver_name_list[i], global_step, eer))
            results.append(acc2)
            if writer:
                writer.add_scalar("Verification Accuracy [%s]" % self.ver_name_list[i], acc2, epoch)


    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")

            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)
    

    def __call__(self, num_update, backbone: torch.nn.Module,  epoch: int, forced: bool=False, writer=None, get_metrics: bool=False):
        if num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update, epoch, writer)
            backbone.train()
            if get_metrics:
                return self.eer
        elif forced:
            backbone.eval()
            self.ver_test(backbone, num_update, epoch, writer)
            backbone.train()
            if get_metrics:
                return self.eer

class CallBackVerificationFT(object):
    def __init__(self, frequent, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.eer:  List[float] = []
        self.img_list : List[str] = []
        self.gender_list : List[str] = [] 
        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test_ft(self, backbone: torch.nn.Module,  layer:torch.nn.Module,  global_step: int, epoch: int, writer=None):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, eer, embeddings_list = verification.testft(
                self.ver_list[i], backbone,layer, 32, 10)
            self.eer.append(eer)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            logging.info('[%s][%d]EER: %1.5f' % (self.ver_name_list[i], global_step, eer))
            results.append(acc2)
            if writer:
                writer.add_scalar("Verification Accuracy [%s]" % self.ver_name_list[i], acc2, epoch)


    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)
    

    def __call__(self, num_update, backbone: torch.nn.Module, layer:torch.nn.Module, epoch: int, forced: bool=False, writer=None, get_metrics: bool=False):
        self.eer=[]
        if num_update > 0 and num_update % self.frequent == 0:
            layer.eval()
            self.ver_test_ft(backbone, layer, num_update, epoch, writer)
            layer.train()
            
        elif forced:
            layer.eval()
            self.ver_test_ft(backbone, layer,num_update, epoch, writer)
            layer.train()

        if get_metrics:
            return self.eer




class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, writer=None, resume=0, rem_total_steps=None):
        self.frequent: int = frequent
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.writer = writer
        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, epoch: int, loss: AverageMeter, loss_2: AverageMeter = None):
        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
            
                speed_total: float = self.frequent * self.batch_size / (time.time() - self.tic)


                time_now = (time.time() - self.time_start) / 3600
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('loss_r', loss.avg, global_step)
                    if loss_2:
                        self.writer.add_scalar('loss_p', loss_2.avg, global_step)
                if loss_2:
                    msg = "Speed %.2f samples/sec   Loss %.4f Loss %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                        speed_total, loss.avg,loss_2.avg, epoch, global_step, time_for_end
                    )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f  Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                        speed_total, loss.avg, epoch, global_step, time_for_end
                    )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

            


class CallBackModelCheckpoint(object):
    def __init__(self, output="./"):
        self.output: str = output
    def __call__(self, global_step, backbone: torch.nn.Module, header: torch.nn.Module = None):

        if global_step > 100 :
            torch.save(backbone.state_dict(), os.path.join(self.output, str(global_step)+ "backbone.pth"))
        if global_step > 100 and header is not None:
            torch.save(header.state_dict(), os.path.join(self.output, str(global_step)+ "header.pth"))
        
