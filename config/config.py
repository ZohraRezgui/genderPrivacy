from easydict import EasyDict as edict
import os

config = edict()
config.embedding_size = 512 # embedding size of model


config.lr = 0.01     
config.headid_lr = 0.01  

config.data = 'AgeDB'# ColorFeret, LFW, AgeDB, VGGFace2
config.pretrained = "ArcFace" # ArcFace, ElasticFace, SphereFace,
config.experiment = "OneDataset"
config.loss="ArcFace"  

config.s=64.0
config.m=0.0
config.alpha=20
config.beta=1

config.batch_size = 128 
config.checkpoint_directory = "_".join([str(config.alpha) + "L1", str(config.beta) + "L2" ])
config.checkpoint_root = "/home/rezguiz/genderPrivacy/models"

# train model output folder
config.output = os.path.join(config.checkpoint_root, config.pretrained, config.experiment, config.data, config.checkpoint_directory)
if os.path.isdir(config.output):
    config.output += '_extra'

config.data_dir = "/home/rezguiz/datasets"
config.log_dir = "/home/rezguiz/MyLogs/MyExperimentsLogsNew"



config.output_ori = os.path.join(config.checkpoint_root, config.pretrained, 'reference', 'backbone.pth')
config.global_step = 0 # step to resume



# type of network to train [iresnet100 | iresnet50]
config.network = "iresnet50"
config.SE=False # SEModule




config.rec = "/home/rezguiz/datasets/bins_verification_gender" # path to the val folder
config.num_epoch =  100
config.warmup_epoch = -1
config.val_targets =  ["lfw",  "agedb_30"]
config.test_targets = ["vggface2", "lfw", "agedb_30", "colorferet"]
config.eval_step= 1000


