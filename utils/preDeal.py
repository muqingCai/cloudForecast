import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def Plot(TsList, NameList, path):
    plt.figure(figsize=(20, 30))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                        wspace=None, hspace=0.4)
    fontSize = 25
    lineWidth = 4
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(TsList)):
        ax = plt.subplot(611 + i)
        ax.spines['left'].set_linewidth(lineWidth)
        ax.spines['right'].set_linewidth(lineWidth)
        ax.spines['bottom'].set_linewidth(lineWidth)
        ax.spines['top'].set_linewidth(lineWidth)
        if i < len(TsList) - 1:
            plt.plot(TsList[i], linewidth=lineWidth)
        else:
            plt.plot([0.3] * len(TsList[i]), linewidth=lineWidth, color='red', linestyle='--')
            plt.plot(TsList[i], color='orange', linewidth=lineWidth)
        if i == 2:
            plt.ylim(ymax=34)
        if i == 3:
            plt.ylim(ymax=24)
        plt.title(NameList[i], fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
    plt.savefig('%spredict_detectAll.jpg' % path)
    plt.show()


def PlotOri(TsList, NameList):
    plt.figure(figsize=(50, 70))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                        wspace=None, hspace=0.4)
    fontSize = 60
    lineWidth = 4
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(TsList)):
        ax = plt.subplot(511 + i)
        ax.spines['left'].set_linewidth(lineWidth)
        ax.spines['right'].set_linewidth(lineWidth)
        ax.spines['bottom'].set_linewidth(lineWidth)
        ax.spines['top'].set_linewidth(lineWidth)
        plt.plot(TsList[i], linewidth=lineWidth)
        plt.title(NameList[i], fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
    plt.savefig('../result/cloudDetection/oriAll.png')
    plt.show()


def Plot2(actualList, forecastList, pvalueList, NameList, path):
    plt.figure(figsize=(30, 40))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                        wspace=None, hspace=0.4)
    fontSize = 30
    lineWidth = 4
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    length = len(NameList)
    TsLen = actualList.shape[1]
    for i in range(length):
        ax = plt.subplot(100 * length + 11 + i)
        ax.spines['left'].set_linewidth(lineWidth)
        ax.spines['right'].set_linewidth(lineWidth)
        ax.spines['bottom'].set_linewidth(lineWidth)
        ax.spines['top'].set_linewidth(lineWidth)
        if i < length - 2:
            plt.plot(actualList[:, i], linewidth=lineWidth)
            plt.plot(forecastList[:, i], linewidth=lineWidth)
        else:
            plt.plot([0.3] * len(pvalueList[i - TsLen]), linewidth=lineWidth, color='red', linestyle='--')
            plt.plot([0.4] * len(pvalueList[i - TsLen]), linewidth=lineWidth, color='red', linestyle='--')
            plt.plot([0.5] * len(pvalueList[i - TsLen]), linewidth=lineWidth, color='red', linestyle='--')
            plt.plot(pvalueList[i - TsLen], color='orange', linewidth=lineWidth)
        if i == 2:
            plt.ylim(ymax=34)
        if i == 3:
            plt.ylim(ymax=24)
        plt.title(NameList[i], fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
    plt.savefig('%s/result/predict_actual_detectAll.jpg' % path)
    plt.show()


def myTest():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    d = b.T
    c = np.concatenate((a, b.T), axis=1)
    print(c)


def getMachineData(dataPath, saveDir):
    data = pd.read_csv(dataPath).values
    machineUsageList = list()
    machineId = 'm_1831'
    for i in range(len(data)):
        if data[i][0] == machineId:
            machineUsageList.append(data[i])
    NameList = ['machine_id', 'time_stamp', 'cpu_util_precent', 'mem_util_percent', 'mem_gps', 'mkpi', 'net_in', 'net_out', 'disk_io_percent']
    machineUsagePd = pd.DataFrame(machineUsageList, columns=NameList)
    machineUsagePd.to_csv('%s/%s.csv' % (saveDir, machineId), index=None)


def getContainerStatus(dataPath):
    data = pd.read_csv(dataPath).values
    length = len(data)
    for i in range(length):
        if data[i, 1] == 'm_1':
            print(data[i])


def getInstanceStatus(dataPath):
    data = pd.read_csv(dataPath).values
    length = len(data)
    for i in range(length):
        if data[i, 4] == 'Failed' and data[i, 7] == 'm_1':
            print(data[i])


def getSortMachineData(dataPath, saveDir):
    data = pd.read_csv(dataPath)
    data = data.sort_values(by="start_time")
    data.to_csv('%s/%s.csv' % (saveDir, 'sortM1831'), index=None)


def TTT(dataPath):
    data = pd.read_csv(dataPath).values
    length = len(data)
    statusList = list()
    for i in range(length):
        flag = 1
        for status in statusList:
            if data[i, 6] == status:
                flag = 0
                break
        if flag == 1:
            print(data[i, 6])
            statusList.append(data[i, 6])


if __name__ == '__main__':
    # dataPath = '../data/cloudData/batchInstance/m_1831.csv'
    # dataPath = '../data/cloudData/machine_usage.csv'
    # saveDir = '../data/cloudData/batchInstance'
    # getMachineData(dataPath, saveDir)
    # getSortMachineData(dataPath, saveDir)

    dataPath = '../data/cloudData/machine_meta.csv'
    TTT(dataPath)

    # path = '../data/cloudData/machineData/sortM1831.csv'
    # data = pd.read_csv(path).iloc[:5000, :]
    # data.to_csv('../data/cloudData/machineData/sortM1831_5000.csv')

    # data = pd.read_csv('../data/cloudData/machineData/sortM1831_5000.csv')
    # TsList = [data.iloc[:, 3], data.iloc[:, 4], data.iloc[:, 5], data.iloc[:, 6], data.iloc[:, 7]]
    # NameList = ['cpu_util_precent', 'mem_util_percent', 'net_in', 'net_out', 'disk_io_percent']
    # PlotOri(TsList, NameList)

    # path = '../runs/cloud/sortM1_5000/2020-05-03-10-05-31-detection/'
    # data = pd.read_csv('%sresult/predict_pvalues.csv' % path).values
    # TsList = list()
    # for i in range(data.shape[1]):
    #     TsList.append(data[:, i])
    # NameList = ['cpu_util_precent', 'mem_util_percent', 'net_in', 'net_out', 'disk_io_percent', 'p_values']
    # Plot(TsList, NameList, path)

    # predictPath = '../runs/cloud/sortM1_5000/2020-05-03-09-29-04_step12_ahead3'
    # actual = pd.read_csv('%s/result/actual.csv' % predictPath).values
    # predict = pd.read_csv('%s/result/forecast.csv' % predictPath).values
    # detectionPath = '../runs/cloud/sortM1_5000/2020-05-03-10-05-31-detection'
    # pvaluesActual = pd.read_csv('%s/result/actual_pvalues.csv' % detectionPath)['p_values'].values
    # pvaluesPredict = pd.read_csv('%s/result/forecast_pvalues.csv' % detectionPath)['p_values'].values
    # pvaluesList = [pvaluesActual, pvaluesPredict]
    # NameList = ['cpu_util_precent', 'mem_util_percent', 'net_in', 'net_out', 'disk_io_percent', 'Actual_p_values', 'predict_p_values']
    # Plot2(actual, predict, pvaluesList, NameList, detectionPath)

    # get container status
    # getContainerStatus('../data/cloudData/sectionData/container_meta.csv')
    # getInstanceStatus('../data/cloudData/batch_instance.csv')
