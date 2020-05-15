## EXTERNAL
from collections import defaultdict

import pandas as pd
import numpy as np
import pickle
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import sklearn
import time
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def prepare_data_lstm(x, y, time_steps, steps_ahead, steps_move, is_train):
    x_lstm = list()
    y_lstm = list()
    start = 0
    # if train mode, the sliding window is 1, else as parameter steps_move
    if is_train:
        steps_move = 1
        start = time_steps
    for i in range(0, len(x) - time_steps - steps_ahead + 1, steps_move):
        x_lstm.append(x[i:i + time_steps])
    for i in range(start, len(y) - steps_ahead + 1, steps_move):
        y_lstm.append(y[i:i + steps_ahead])
    return x_lstm, y_lstm


# 这里ExampleDataset继承了Dataset类
class ExampleDataset(Dataset):
    def __init__(self, x, y, batchsize):
        self.datalist = x
        self.target = y
        self.batchsize = batchsize
        self.length = 0
        self.length = len(x)

    def __len__(self):
        return int(self.length / self.batchsize + 1)

    def __getitem__(self, idx):
        x = self.datalist[idx * self.batchsize:(idx + 1) * self.batchsize]
        y = self.target[idx * self.batchsize:(idx + 1) * self.batchsize]
        sample = {'x': x, 'y': y}

        return sample


def save_checkpoint(state, directory, fileName, isBest):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    if isBest:
        fileName = directory + 'best.pth.tar'
    else:
        fileName = directory + 'last.pth.tar'
    torch.save(state, fileName)


# save the running parameters to txt
def save_parameters(path, params):
    if not os.path.exists(path):
        os.makedirs(path)
    filePath = path + '/Params.txt'
    fo = open(filePath, 'w')
    for key in params:
        fo.write(str(key) + '=' + str(params[key]) + '\n')
    fo.close()


def deleteFile(filePath):
    if os.path.exists(filePath):
        # 删除文件，可使用以下两种方法。
        os.remove(filePath)
    else:
        print('no such file:%s' % filePath)


def normalLizeData(data, isReshape, scaler):
    if isReshape:
        data = np.reshape(data, (data.shape[0], 1))
        data = scaler.fit_transform(data)
        data = data.flatten()
    else:
        data = scaler.fit_transform(data)
    return data


def inverseData(data, isReshape, scaler):
    if isReshape:
        data = np.reshape(data, (data.shape[0], 1))
        data = scaler.inverse_transform(data)
        data = data.flatten()
    else:
        data = scaler.inverse_transform(data)
    return data


# calculate the MAPE
def getMAPE(actual, forecast):
    length = len(actual)
    tempSum = 0
    for i in range(length):
        tempSum += abs((actual[i] - forecast[i]) / actual[i])
    MAPE = tempSum / length
    return MAPE


# calculate the R
def getR(actual, forecast):
    meanA = np.mean(actual)
    meanF = np.mean(forecast)
    length = len(actual)
    tempSum1 = 0
    tempSumA = 0
    tempSumB = 0
    for i in range(length):
        tempSum1 += (actual[i] - meanA) * (forecast[i] - meanF)
        tempSumA += np.square(actual[i] - meanA)
        tempSumB += np.square(forecast[i] - meanF)
    R = tempSum1 / np.sqrt(tempSumA * tempSumB)
    return R


# calculate the Theil
def getTheilU(actual, forecast):
    tempSum1 = 0
    tempSumA = 0
    tempSumF = 0
    length = len(actual)
    for i in range(length):
        tempSum1 += np.square(actual[i] - forecast[i])
        tempSumA += np.square(actual[i])
        tempSumF += np.square(forecast[i])
    TheilU = np.sqrt(tempSum1) / (np.sqrt(tempSumA) + np.sqrt(tempSumF))
    return TheilU


# calculate the RMSE
def getRMSE(actual, forecast):
    length = len(actual)
    tempSum = 0
    for i in range(length):
        tempSum += np.square(actual[i] - forecast[i])
    RMSE = np.sqrt(tempSum / length)
    return RMSE


# calculate the MAE
def getMAE(actual, forecast):
    length = len(actual)
    tempSum = 0
    for i in range(length):
        tempSum += abs(actual[i] - forecast[i])
    MAE = tempSum / length
    return MAE


# calculate the RMSE
def getMultiRMSE(actual, forecast):
    length = actual.shape[0]
    dim = actual.shape[1]
    tempSum = 0
    allSum = 0
    for i in range(dim):
        for j in range(length):
            tempSum += np.square(actual[j, i] - forecast[j, i])
        tempSum /= length
        allSum += tempSum
    RMSE = np.sqrt(allSum / dim)
    return RMSE


# calculate the MAE
def getMultiMAE(actual, forecast):
    length = actual.shape[0]
    dim = actual.shape[1]
    tempSum = 0
    allSum = 0
    for i in range(dim):
        for j in range(length):
            tempSum += abs(actual[j, i] - forecast[j, i])
        tempSum /= length
        allSum += tempSum
    MAE = tempSum / dim
    return MAE


def SaveResult(actual, forecast, path):
    if not os.path.exists(path):
        os.makedirs(path)

    # plot the true series and prediction series
    plt.figure(figsize=(12, 8))
    plt.plot(actual, color='blue')
    plt.plot(forecast, color='red')
    plt.savefig(path + "actual-predict.jpg")

    # save the evaluation indicators
    MAE = getMAE(actual, forecast)
    R = getR(actual, forecast)
    TheilU = getTheilU(actual, forecast)
    RMSE = getRMSE(actual, forecast)
    resultA = [MAE, R, TheilU, RMSE]
    resultA = np.vstack(resultA).T
    df = pd.DataFrame(resultA, columns=['MAE', 'R', 'TheilU', 'RMSE'])
    df.to_csv(path + 'resultsErr.csv', index=None)

    # save the prediction series
    resultPre = [actual, forecast]
    resultPre = np.vstack(resultPre).T
    df = pd.DataFrame(resultPre, columns=['actual', 'forecast'])
    df.to_csv(path + 'resultsTs.csv', index=None)


def MultiSaveResult(actual, forecast, nameList, path):
    dim = actual.shape[1]
    if not os.path.exists(path):
        os.makedirs(path)
    # plot the actual and forecast
    plt.figure(figsize=(40, 30))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                        wspace=None, hspace=0.4)
    fontSize = 30
    lineWidth = 4
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(dim):
        ax = plt.subplot(511 + i)
        ax.spines['left'].set_linewidth(lineWidth)
        ax.spines['right'].set_linewidth(lineWidth)
        ax.spines['bottom'].set_linewidth(lineWidth)
        ax.spines['top'].set_linewidth(lineWidth)
        plt.plot(actual[:, i], label='actual', linewidth=lineWidth)
        plt.plot(forecast[:, i], label='forecast', linewidth=lineWidth)
        plt.title(nameList[i], fontsize=fontSize)
        plt.legend(loc='best', fontsize=15)
        if i == 2:
            plt.ylim(ymax=34)
        if i == 3:
            plt.ylim(ymax=24)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
    plt.savefig(path + 'compare.jpg')
    plt.show()
    # save the evaluation indicators
    MAE = getMultiMAE(actual, forecast)
    RMSE = getMultiRMSE(actual, forecast)
    resultA = [MAE, RMSE]
    resultA = np.vstack(resultA).T
    df = pd.DataFrame(resultA, columns=['MAE', 'RMSE'])
    df.to_csv(path + 'resultsErr.csv', index=None)

    # save the prediction series
    dfActual = pd.DataFrame(actual)
    dfForecast = pd.DataFrame(forecast)
    dfActual.to_csv(path + 'actual.csv', index=None)
    dfForecast.to_csv(path + 'forecast.csv', index=None)


def reformatData(filePath, dataName):
    data = pd.read_csv(filePath).iloc[26304:, 1:]

    oriLen = len(data)
    # # drop NA value
    # data.dropna(axis=0, how='any', inplace=True)
    # data.reset_index(inplace=True, drop=True)

    # 将NA值设置为前面的值
    for i in range(len(data)):
        for j in data.columns:
            if pd.isnull(data.loc[i, j]):
                # 先找前面不为0的替代
                flag = i - 1
                while flag >= 0:
                    if not pd.isnull(data.loc[flag, j]):
                        data.loc[i, j] = data.loc[flag, j]
                        break
                    flag -= 1
                # 前面找不到找后面的
                if pd.isnull(data.loc[i, j]):
                    flag = i + 1
                    while flag < len(data):
                        if not pd.isnull(data.loc[flag, j]):
                            data.loc[i, j] = data.loc[flag, j]
                            break
                        flag += 1
    # 将风向换为枚举值
    for i in range(len(data)):
        if data.loc[i, 'cbwd'] == 'SE':
            data.loc[i, 'cbwd'] = 1
        elif data.loc[i, 'cbwd'] == 'SW':
            data.loc[i, 'cbwd'] = 2
        elif data.loc[i, 'cbwd'] == 'NE':
            data.loc[i, 'cbwd'] = 3
        elif data.loc[i, 'cbwd'] == 'NW':
            data.loc[i, 'cbwd'] = 4
        elif data.loc[i, 'cbwd'] == 'cv':
            data.loc[i, 'cbwd'] = 5
        else:
            data.loc[i, 'cbwd'] = 6
    nowLen = len(data)
    print('oriLen:%s,nowLen:%s,totolNum%s' % (str(oriLen), str(nowLen), str(oriLen - nowLen)))
    # data.to_csv('../data/FiveCitiePMData/' + dataName + '/formatAll.csv')


def GetTrainValTestData(path, savePath, time_steps, steps_ahead, num_sub_series, method, No):
    # read data
    feats = pd.read_csv(path)

    # the test data
    df_test = feats[-720:]

    # get the train data
    sub_series_list = list()
    sub_series_len = time_steps + steps_ahead
    for i in range(0, len(feats) - sub_series_len - 720 + 1, sub_series_len):
        sub_series_list.append(feats.iloc[i:i + sub_series_len, :])

    # get num_sub_series  sub_series
    randomList = random.sample(range(0, len(sub_series_list)), num_sub_series)

    # concat sub_series
    feats = sub_series_list[randomList[0]]
    for i in range(1, len(randomList)):
        feats = pd.concat([feats, sub_series_list[randomList[i]]])

    feats = pd.concat([feats, df_test])
    feats.reset_index(drop=True, inplace=True)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    feats.to_csv(savePath + '%s-step%s-ahead%s-No%s.csv' % (method, str(time_steps), str(steps_ahead), No))


def GetBaggingResult(pathList, savePath):
    data = list()
    actual = list()
    for path in pathList:
        forecast = pd.read_csv(path)['forecast'].values
        actual = pd.read_csv(path)['actual'].values
        data.append(forecast)
    BaggingForecast = data[0]
    for i in range(1, len(data)):
        BaggingForecast += data[i]
    BaggingForecast = BaggingForecast / len(data)
    SaveResult(actual, BaggingForecast, savePath)
    return BaggingForecast


def PlotPic(TsList, directory):
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                        wspace=None, hspace=0.4)
    fontSize = 15
    lineWidth = 2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(TsList)):
        ax = plt.subplot(121 + i)
        ax.spines['left'].set_linewidth(lineWidth)
        ax.spines['right'].set_linewidth(lineWidth)
        ax.spines['bottom'].set_linewidth(lineWidth)
        ax.spines['top'].set_linewidth(lineWidth)
        plt.plot(TsList[i]['actual'], label='实际序列', linewidth=lineWidth)
        plt.plot(TsList[i]['forecast'], label='预测序列', linewidth=lineWidth)
        plt.legend(loc='best', fontsize=15)
        if i == 0:
            dataName = 'DA-RNN'
        else:
            dataName = 'UM-CPA-EASL'
        plt.title(dataName, fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        plt.xlabel('时间(h)', fontsize=fontSize)
        plt.ylabel('PM2.5(ug/m^3)', fontsize=fontSize)
    plt.savefig(directory + "step3.png")
    plt.show()


def Plot3D():
    # 定义坐标轴
    from mpl_toolkits.mplot3d import Axes3D
    # 定义图像和三维格式坐标轴
    fig = plt.figure()
    ax = Axes3D(fig)
    z = random.sample(range(50, 250), 50)
    x = random.sample(range(50, 250), 50)
    y = random.sample(range(50, 250), 50)

    z1 = random.sample(range(10, 50), 5)
    x1 = random.sample(range(10, 50), 5)
    y1 = random.sample(range(10, 50), 5)

    z2 = random.sample(range(250, 300), 5)
    x2 = random.sample(range(250, 300), 5)
    y2 = random.sample(range(250, 300), 5)

    z3 = random.sample(range(10, 50), 5)
    x3 = random.sample(range(300, 310), 5)
    y3 = random.sample(range(300, 310), 5)

    # zd = 13 * np.random.random(100)
    # xd = 5 * np.sin(zd)
    # yd = 5 * np.cos(zd)
    # ax.scatter3D(xd, yd, zd, cmap='Blues')  # 绘制散点图
    ax.scatter3D(x, y, z)
    ax.scatter3D(x1, y1, z1)
    ax.scatter3D(x2, y2, z2)
    ax.scatter3D(x3, y3, z3)
    plt.show()


def PlotDiffColor():
    from matplotlib.collections import LineCollection
    import numpy as np
    import math
    path = '../data/FiveCitiePMData/Beijing/step3/TAD-step3-ahead3-No1.csv'
    y = pd.read_csv(path)['PM_US Post'].values[:600]
    x = np.linspace(0, 600, 600)
    color = []
    for i in range(len(y)):
        if i < 450:
            color.append('blue')
        else:
            color.append('red')
    plt.figure(figsize=(20, 8))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=2, color=color)

    ax = plt.axes()
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.add_collection(lc)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'AR PL UKai CN'
    plt.rcParams['axes.unicode_minus'] = False
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def PlotPvalue(TsList, directory):
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                        wspace=None, hspace=0.4)
    fontSize = 20
    lineWidth = 2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(TsList)):
        ax = plt.subplot(211 + i)
        ax.spines['left'].set_linewidth(lineWidth)
        ax.spines['right'].set_linewidth(lineWidth)
        ax.spines['bottom'].set_linewidth(lineWidth)
        ax.spines['top'].set_linewidth(lineWidth)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        if i == 0:
            plt.xlabel('时间(h)', fontsize=fontSize)
            plt.ylabel('PM2.5(ug/m^3)', fontsize=fontSize)
            plt.plot(TsList[i], linewidth=lineWidth, color='blue')
            plt.title("PM_US Post时间序列", fontsize=fontSize)
        else:
            plt.xlabel('时间(h)', fontsize=fontSize)
            plt.ylabel('p_value', fontsize=fontSize)
            plt.plot(TsList[i], linewidth=lineWidth, color='orange')
            plt.plot([0.5] * len(TsList[i]), linewidth=lineWidth, color='red', linestyle='--')
            plt.title("p_value", fontsize=fontSize)

    plt.savefig(directory + "compare.png")
    plt.show()


def aa(TsList):
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                        wspace=None, hspace=0.4)
    fontSize = 15
    lineWidth = 2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(TsList)):
        ax = plt.subplot(111 + i)
        ax.spines['left'].set_linewidth(lineWidth)
        ax.spines['right'].set_linewidth(lineWidth)
        ax.spines['bottom'].set_linewidth(lineWidth)
        ax.spines['top'].set_linewidth(lineWidth)
        plt.plot(TsList[i], linewidth=lineWidth)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        plt.xlabel('时间(h)', fontsize=fontSize)
        plt.ylabel('PM2.5($\mu$g/m^3)', fontsize=fontSize)
    plt.savefig("../result/resultOri.png")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('../data/FiveCitiePMData/Beijing/formatAll.csv')['PM_US Post'].values[1000:2000]
    aa([data])
    # drop NaN data and transform the non-value data to value data
    # path = '../data/FiveCitiePMData/BeijingPM20100101_20151231.csv'
    # dataset = 'Guangzhou'
    # reformatData(path, dataset)
    #
    # # get the train/val/test data for bagging
    # dataset = 'Guangzhou'
    # time_steps = 15
    # steps_ahead = 6
    # num_subSeries = 128
    # No = 1
    # method = 'TA'
    # path = '../data/FiveCitiePMData/Guangzhou/formatAll.csv'
    # path = '../runs/seq2seq/Guangzhou/onlyNoiseReduction/denoiseSeries.csv'
    # path = '../runs/seq2seq/Guangzhou/anomalyDetection/2020-03-24-10-46-55/result/recover.csv'
    # path = '../runs/seq2seq/%s/detectionAndNoiseReduction/denoiseSeries.csv' % dataset
    # savePath = '../data/FiveCitiePMData/%s/step%s/' % (dataset, str(steps_ahead))
    # GetTrainValTestData(path, savePath, time_steps, steps_ahead, num_subSeries, method, No)

    # filePath = 'runs/seq2seq/Guangzhou/onlyNoiseReduction/denoiseSeries.csv'
    # No1 = '2020-03-23-00-55-40_step3_No1'
    # No2 = '2020-03-23-00-56-19_step3_No2'
    # No3 = '2020-03-23-00-54-48_step3_No3'
    # savePath = '../runs/seq2seq/Guangzhou/BaggingSeq2Seq/T-BaggingResult/'
    # pathList = [filePath % No1, filePath % No2, filePath % No3]
    # GetBaggingResult(pathList, savePath)

    # dataset = 'Guangzhou/'

    # dataset = 'Guangzhou/'
    # path = '../result/predict/' + dataset
    # DA_RNN_step3 = pd.read_csv(path + 'test.csv')
    # data = DA_RNN_step3['forecast'].values[:300]
    # for i in range(len(data)):
    #     if data[i] < 50:
    #         data[i] += 50
    #     if data[i] > 200:
    #         data[i] -= 50
    # data[3] = 2
    # data[25] = 467
    # data[87] = 392
    # data[101] = 5
    # data[102] = 7
    # data[103] = 19
    # data[292] = 332
    # data[289] = 301
    # color = []
    # for i in range(len(data)):
    #     if i > len(data)-50:
    #         color.append('red')
    #     else:
    #         color.append('blue')
    # plt.figure(figsize=(20, 10))
    # plt.plot(data, linewidth=3, color=color)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.show()

    # DA_RNN_step3 = pd.read_csv(path + 'DA-RNN-TAD-step15-3.csv')
    #
    # DA_RNN_step9 = pd.read_csv(path + 'DA-RNN-TAD-step15-9.csv')
    # # DA_RNN_step9 = pd.read_csv(path + 'DA-RNN-TAD-step15-9.csv')
    #
    # ourModel_step3 = pd.read_csv(path + 'ourModel-TAD-step9-3.csv')
    # # ourModel_step3 = pd.read_csv(path + 'ourModel-TAD-step9-3.csv')
    #
    # ourModel_step9 = pd.read_csv(path + 'ourModel-TAD-step24-9.csv')
    # # ourModel_step9 = pd.read_csv(path + 'ourModel-TAD-step24-9.csv')
    #
    # dataList = [DA_RNN_step3, ourModel_step3]
    # PlotPic(dataList, path)

    # Plot3D()
    # PlotDiffColor()

    # about detection
    # path = '../runs/seq2seq/Guangzhou/anomalyDetection/2020-03-24-10-46-55/result/'
    # fileName = 'pvalues.csv'
    # data = pd.read_csv(path + fileName)
    # dataList = [data['post'].values[:3000], data['p_value'].values[:3000]]
    # PlotPvalue(dataList, path)
