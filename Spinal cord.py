import sys

sys.path.append("..")
from utilize_190805 import *
from data_regulation_190805 import *
import os
Target_Organ = "Spinal cord"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

train_stage2(Organ_config_stage2[Target_Organ]+[1],epochs=200,batch_size=1,organ=Target_Organ)
