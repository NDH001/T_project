import pandas as pd
from eda import cleaning
from model.trainEva import TrainEva
from model.customDataSet import ChurnData
from model.mlp import MLPReg
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def getCleanedData(path):
    return cleaning(pd.read_csv(path))


def getCustomedData(df):
    return ChurnData(getCleanedData(df))


def setUp(path, splits):
    df = getCustomedData(path)
    kfold = KFold(n_splits=splits, shuffle=True)
    model = MLPReg(df)

    return df, kfold, model


def showplt(data, title, n):
    fig, ax = plt.subplots(3, figsize=(5, 12))

    for i in range(len(data)):
        ax[i].plot(data[i])
        ax[i].set_title(title[i])

    plt.show()


def main(path, epochs=30, l_r=0.001, splits=5, report=False):
    df, kfold, model = setUp(path, splits)
    l, a, v = [], [], []
    for _, (train_ids, test_ids) in enumerate(kfold.split(df)):
        train_set = SubsetRandomSampler(train_ids)
        test_set = SubsetRandomSampler(test_ids)
        train_loader = DataLoader(df, batch_size=64, sampler=train_set)
        test_loader = DataLoader(df, batch_size=64, sampler=test_set)
        tne = TrainEva(model, epochs, lr=l_r)
        l_t, a_t, v_t = tne.train(train_loader, test_loader)
        l.append(sum(l_t) / len(l_t))
        a.append(sum(a_t) / len(a_t))
        v.append(sum(v_t) / len(v_t))

    if report:
        showplt([l, a, v], ["loss", "acc", "val"], splits)


if __name__ == "__main__":
    main(
        r"C:\Users\ZhiJun\Desktop\T_project\WA_Fn-UseC_-Telco-Customer-Churn.csv",
        1,
        0.001,
        2,
        True,
    )
