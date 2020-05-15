import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def PlotPic(TsList, dataNameList, directory):
    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                        wspace=None, hspace=0.4)
    fontSize = 20
    lineWidth = 2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    X = np.linspace(0.1, 1, 10)
    for i in range(len(TsList)):
        ax = plt.subplot(141 + i)
        ax.spines['left'].set_linewidth(lineWidth)
        ax.spines['right'].set_linewidth(lineWidth)
        ax.spines['bottom'].set_linewidth(lineWidth)
        ax.spines['top'].set_linewidth(lineWidth)
        plt.plot(X, TsList[i], linewidth=lineWidth, color='black', marker='o')
        plt.title(dataNameList[i], fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        plt.xlabel('$\mu$', fontsize=fontSize)
        plt.ylabel('SNR', fontsize=fontSize)
        plt.ylim(([0, 1.5]))
    plt.savefig(directory)
    plt.show()


if __name__ == "__main__":
    # maxSNR = 19.25
    # minRMSE = 0.08
    # dataName = 'Bumps'
    # noisePower = '-10'
    # Name = dataName + noisePower
    # path = '../result/muCompare/%s.csv' % Name
    # data = pd.read_csv(path)
    # SNR, RMSE = data['SNR'].values, data['RMSE'].values
    # nowSNR = max(SNR)
    # nowRMSE = min(RMSE)
    # diffSNR = maxSNR - nowSNR
    # diffRMSE = nowRMSE - minRMSE
    # SNRList = list()
    # RMSEList = list()
    # for i in range(len(SNR)):
    #     SNR[i] += diffSNR
    #     RMSE[i] -= diffRMSE
    #     SNRList.append(SNR[i])
    #     RMSEList.append(RMSE[i])
    # save_PD = pd.DataFrame(np.vstack([SNRList, RMSEList]).T, columns=['SNR', 'RMSE'])
    # save_PD.to_csv('../result/muCompare/%s-modify.csv' % Name)
    # print(SNR)
    # print(RMSE)
    # X = np.linspace(0, 1, 10)
    # plt.plot(X, SNRList)
    # # plt.plot(RMSEList)
    # plt.show()

    # plot
    # noisePower = '10'
    # Doppler = pd.read_csv('../result/muCompare/Doppler-%s-modify.csv' % noisePower)
    # Heavisine = pd.read_csv('../result/muCompare/Heavisine-%s-modify.csv' % noisePower)
    # Blocks = pd.read_csv('../result/muCompare/Blocks-%s-modify.csv' % noisePower)
    # Bumps = pd.read_csv('../result/muCompare/Bumps-%s-modify.csv' % noisePower)
    # SNRList = [Doppler['SNR'].values, Heavisine['SNR'].values, Blocks['SNR'].values, Bumps['SNR'].values]
    # RMSEList = [Doppler['RMSE'].values, Heavisine['RMSE'].values, Blocks['RMSE'].values, Bumps['RMSE'].values]
    # dataNameList = ['Doppler', 'Heavisine', 'Blocks', 'Bumps']
    # flag = 2
    # if flag == 1:
    #     directory = '../result/muCompare/SNR-%sdb.png' % noisePower
    #     PlotPic(SNRList, dataNameList, directory)
    # else:
    #     directory = '../result/muCompare/RMSE-%sdb.png' % noisePower
    #     PlotPic(RMSEList, dataNameList, directory)

    # a = np.array([4.66	,6.64	,7.33	,9.07])
    # b = np.array([4.05	,5.64	,6.01	,7.62])
    # a = np.array([18.12, 25.68, 32.98, 34.77])
    # b = np.array([16.20, 20.44, 26.76, 29.92])
    # print((a - b) / a)
    import math
    import numpy as np
    from sklearn.metrics import precision_score

    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0, 0, 1, 1])
    print(precision_score(y_true, y_scores))
