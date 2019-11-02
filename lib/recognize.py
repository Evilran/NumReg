import os
from PIL import Image
import lib.generate as g
from sklearn.externals import joblib

modelPath = os.getcwd() + "/data/data_models/"


def recognizeNumber(path, flag):
    num = g.getFeature(Image.open(path))
    if flag is 1:
        LR(num)
    elif flag is 2:
        LSVC(num)
    elif flag is 3:
        MLPC(num)
    elif flag is 4:
        SGDC(num)

# LR


def LR(num):
    LR = joblib.load(modelPath + "model_LR.m")
    predict_LR = LR.predict([num])
    print("[LR] Handwritten number is closest to", predict_LR[0])

# LSVC


def LSVC(num):
    LSVC = joblib.load(modelPath + "model_LSVC.m")
    predict_LSVC = LSVC.predict([num])
    print("[LSVC] Handwritten number is closest to", predict_LSVC[0])

# MLPC


def MLPC(num):
    MLPC = joblib.load(modelPath + "model_MLPC.m")
    predict_MLPC = MLPC.predict([num])
    print("[MLPC] Handwritten number is closest to", predict_MLPC[0])

# SGDC


def SGDC(num):
    SGDC = joblib.load(modelPath + "model_SGDC.m")
    predict_SGDC = SGDC.predict([num])
    print("[SGDC] Handwritten number is closest to", predict_SGDC[0])
