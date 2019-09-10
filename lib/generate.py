import os
import csv
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

imgPath = os.getcwd() + "/data/data_images/"
csvPath = os.getcwd() + "/data/data_csvs/"
modelPath = os.getcwd() + "/data/data_models/"

# generate folders to store the image data


def generateFolders():
    for i in range(10):
        if os.path.isdir(imgPath + "num_" + str(i)):
            for i in range(10):
                list = os.listdir(imgPath+"num_"+str(i))
                for cnt in list:
                    if cnt in list:
                        print("Delete pic ", imgPath +
                              "num_" + str(i) + "/"+cnt)
                        os.remove(imgPath + "num_" + str(i) + "/"+cnt)
            pass
        else:
            print("Creating folder:", imgPath + "num_"+str(i)+'/')
            os.mkdir(imgPath + "num_"+str(i))

# generate a single distorted digital image


def generateNum(font):
    blank = Image.new('RGB', (50, 50), (255, 255, 255))
    draw = ImageDraw.Draw(blank)
    num = str(random.randint(0, 9))
    # default font is 20px
    font = ImageFont.truetype(os.getcwd() + '/lib/' + font, 20)
    draw.text(xy=(15, 7), font=font, text=num, fill=(0, 0, 0))
    # random rotated -10-10 angle
    random_angle = random.randint(-10, 10)
    rotated = blank.rotate(random_angle)
    # graphic distortion parameter
    params = [1 - float(random.randint(1, 2)) / 100,
              0,
              0,
              0,
              1 - float(random.randint(1, 10)) / 100,
              float(random.randint(1, 2)) / 500,
              0.001,
              float(random.randint(1, 2)) / 500]
    transformed = rotated.transform((50, 50), Image.PERSPECTIVE, params)
    # generate 30*30 blank pics
    img = transformed.crop([10, 10, 40, 40])
    return img, num

# extract the feature of a single image


def getFeature(img):
    pixels = []
    height, width = 30, 30
    for y in range(height):
        pixel = 0
        for x in range(width):
            if img.getpixel((x, y)) == 0:  # if it is black
                pixel += 1
        pixels.append(pixel)

    for x in range(width):
        pixel = 0
        for y in range(height):
            if img.getpixel((x, y)) == 0:
                pixel += 1
        pixels.append(pixel)
    return pixels


def generateNums(nums, font):
    generateFolders()
    cnt = []
    for i in range(10):
        cnt.append(0)
    # generate times: nums
    for m in range(1, nums+1):
        img, num = generateNum(font)
        gray = img.convert('1')
        for j in range(10):
            if num == str(j):
                cnt[j] = cnt[j]+1
                print("Draw pic " + imgPath + "num_"+str(j) +
                      "/" + str(j) + "_" + str(cnt[j]) + ".png")
                gray.save(imgPath + "num_"+str(j) + "/" +
                          str(j) + "_" + str(cnt[j]) + ".png")

    # output the distribution of 0-9
    print("\nGenerate 0-9 handwritten numbers result：")
    for k in range(10):
        print("Number", k, "-", cnt[k], "pics")
    print('\nGet the features of handwritten numbers...')
    sum = 0
    # save the features in csv file
    with open(csvPath + "tmp.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(10):
            list = os.listdir(imgPath + "num_" + str(i))
            if os.path.isdir(imgPath + "num_" + str(i)):
                sum += len(list)
                for j in range(len(list)):
                    img = Image.open(imgPath + "num_" + str(i)+"/" + list[j])
                    pixels = getFeature(img)
                    pixels.append(list[j][0])
                    writer.writerow(pixels)

    if "data_"+str(sum)+".csv" in os.listdir(csvPath):
        os.remove(csvPath+"data_" + str(sum) + ".csv")
    os.rename(csvPath+"tmp.csv", csvPath+"data_"+str(sum)+".csv")

    print('Training...')
    list = os.listdir(csvPath)
    for dir in list:
        print("Analysing " + dir + "...")
        col = []
        for i in range(0, 60):
            col.append("feature_" + str(i))
        col.append("true_number")
        data = pd.read_csv(csvPath + dir, names=col)
        x_train, x_test, y_train, y_test = train_test_split(
            data[col[0:60]], data[col[60]], test_size=0.2, random_state=0)
        LR(x_train, x_test, y_train, y_test)
        MLPC(x_train, x_test, y_train, y_test)
        LSVC(x_train, x_test, y_train, y_test)
        SGDC(x_train, x_test, y_train, y_test)
        print("\n")
    print("Completed Successfully!")


def LR(x_train, x_test, y_train, y_test):
    LR = LogisticRegression()
    LR.fit(x_train, y_train)
    y_predict = LR.predict(x_test)
    score = LR.score(x_test, y_test)
    print("The accurary of LR: ", score)
    joblib.dump(LR, modelPath + "model_LR.m")


def MLPC(x_train, x_test, y_train, y_test):
    MLPC = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
    MLPC.fit(x_train, y_train)
    y_predict = MLPC.predict(x_test)
    score = MLPC.score(x_test, y_test)
    print("The accurary of MLPC: ", score)
    joblib.dump(MLPC, modelPath + "model_MLPC.m")


def LSVC(x_train, x_test, y_train, y_test):
    LSVC = LinearSVC()
    LSVC.fit(x_train, y_train)
    y_predict = LSVC.predict(x_test)
    score = LSVC.score(x_test, y_test)
    print("The accurary of LSVC: ", score)
    joblib.dump(LSVC, modelPath + "model_LSVC.m")


def SGDC(x_train, x_test, y_train, y_test):
    SGDC = SGDClassifier(max_iter=5)
    SGDC.fit(x_train, y_train)
    y_predict = SGDC.predict(x_test)
    score = SGDC.score(x_test, y_test)
    print("The accurary of SGDC: ", score)
    joblib.dump(SGDC, modelPath + "model_SGDC.m")
