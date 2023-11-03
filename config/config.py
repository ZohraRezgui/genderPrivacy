from easydict import EasyDict as edict
import os

config = edict()
config.embedding_size = 512 # embedding size of model


config.lr = 0.01     
config.headid_lr = 0.01  

config.data = 'LFW'
config.experiment = "OneDataset"
config.loss="ArcFace"  

config.s=64.0
config.m=0
config.alpha=20
config.beta=1
config.batch_size =128 

# train model output folder
config.output = "/home/rezguiz/genderPrivacy/models/" + config.experiment + "/" + config.data + "/" +str(config.alpha) + "L1" +"_" + str(config.beta) + "L2" + "_lr" + str(config.lr).replace('.','') + "_m" + str(config.m) + "_s" +str(config.m)   # train model output folder
config.data_dir = "/home/rezguiz/datasets"
config.log_dir = "/home/rezguiz/logs"
if os.path.isdir(config.output):
    config.output += '_extra'


config.output_ori = "/home/rezguiz/genderPrivacy/models/reference"
config.global_step = 0 # step to resume



# type of network to train [iresnet100 | iresnet50]
config.network = "iresnet50"
config.SE=False # SEModule




config.rec = "/home/rezguiz/datasets/faces_webface_112x112" # path to the val folder
config.num_epoch =  100
config.warmup_epoch = -1
config.val_targets =  ["lfw",  "agedb_30"]
config.test_targets = ["lfw", "agedb_30","vggface2"]
config.eval_step= 1000


