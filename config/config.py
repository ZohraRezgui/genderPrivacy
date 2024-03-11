from easydict import EasyDict as edict
import os

config = edict()
config.embedding_size = 512 # embedding size of model


config.lr = 0.01     
config.headid_lr = 0.01  

config.data = 'LFW-AgeDB'#ColorFeret, LFW, AgeDB, VGGFace2
config.pretrained = "ElasticFace" #ArcFace, ElasticFace, SphereFace,
config.experiment = "FoldsDataset"
config.loss="ArcFace"  

config.s=64.0
config.m=0
config.alpha=20
config.beta=0
config.batch_size =128 

# train model output folder
config.output = "/home/rezguiz/genderPrivacy/models/" + config.pretrained + "/" + config.experiment + "/" + config.data + "/" +str(config.alpha) + "L1" +"_" + str(config.beta) + "L2"   # train model output folder
config.data_dir = "/home/rezguiz/datasets"
config.log_dir = "/home/rezguiz/MyLogs/MyExperimentsLogsNew"
if os.path.isdir(config.output):
    config.output += '_extra'


config.output_ori = "/home/rezguiz/genderPrivacy/models/" + config.pretrained+"/reference"
config.global_step = 0 # step to resume



# type of network to train [iresnet100 | iresnet50 | sfnet20]
config.network = "iresnet100"
config.SE=False # SEModule




config.rec = "/home/rezguiz/datasets/faces_bins_112x112" # path to the val folder
config.num_epoch =  100
config.warmup_epoch = -1
config.val_targets =  ["lfw",  "agedb_30"]
config.test_targets = ["vggface2", "lfw", "agedb_30", "colorferet"]
config.eval_step= 1000


