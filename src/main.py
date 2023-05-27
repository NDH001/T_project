import pandas as pd
from eda import cleaning
from model.trainEva import TrainEva
from model.customDataSet import ChurnData
from model.mlp import MLPReg
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def getCleanedData(path):
    return cleaning(pd.read_csv(path))


def setUp(ori, splits):
    df = ori.copy()
    df = ChurnData(df)
    kfold = KFold(n_splits=splits, shuffle=True)
    model = MLPReg(df)

    return df, kfold, model


def showplt(data, title, n):
    fig, ax = plt.subplots(3, figsize=(5, 12))

    for i in range(len(data)):
        ax[i].plot(data[i])
        ax[i].set_title(title[i])

    plt.show()


def manualTrain(ori, epochs, l_r, splits):
    df, kfold, model = setUp(ori, splits)
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

    showplt([l, a, v], ["loss", "acc", "val"], splits)

    x = torch.tensor(ori.drop(columns=["Churn"]).to_numpy()).float()
    y = torch.tensor(ori["Churn"].to_numpy())
    pred = torch.argmax(model(x), 1)
    print(f"MLP training f1 score --> {f1_score(pred, y)}")
    res = confusion_matrix(pred, y)
    plot(res)


def plot(cf):
    ax = sns.heatmap(cf, annot=True, fmt="d")
    ax.set_xlabel("Predicted")
    ax.xaxis.set_ticklabels(["Negative", "Positive"])

    ax.set_ylabel("Actual")
    ax.yaxis.set_ticklabels(["Negative", "Positive"])
    plt.show()


def rfTrain(train_x, test_x, train_y, test_y):
    rf = RandomForestClassifier(n_estimators=1000)
    scores = cross_val_score(rf, test_x, test_y, cv=5)
    print(f"rf training acc --> {sum(scores) / 5}")
    rf_res = rf.fit(train_x, train_y)
    pred = rf_res.predict(test_x)
    print(f"rf training f1 score --> {f1_score(pred, test_y)}")
    cf_rf = confusion_matrix(pred, test_y)
    plot(cf_rf)


def lgTrain(train_x, test_x, train_y, test_y):
    lr = LogisticRegression(max_iter=200)
    scores = cross_val_score(lr, test_x, test_y, cv=5)
    print(f"lg training acc --> {sum(scores) / 5}")
    lr_res = lr.fit(train_x, train_y)
    pred = lr_res.predict(test_x)
    print(f"lg training f1 score --> {f1_score(pred, test_y)}")
    cf_lr = confusion_matrix(pred, test_y)
    plot(cf_lr)


def main(path, epochs=30, l_r=0.001, splits=5, report=False):
    # MLP training
    df = getCleanedData(path)
    manualTrain(df, epochs, l_r, splits)

    # rf and lg training
    train_x, test_x, train_y, test_y = train_test_split(
        df.drop(columns=["Churn"]), df["Churn"], test_size=0.2, random_state=42
    )
    rfTrain(train_x, test_x, train_y, test_y)
    lgTrain(train_x, test_x, train_y, test_y)


if __name__ == "__main__":
    main(
        r"WA_Fn-UseC_-Telco-Customer-Churn.csv",
        30,
        0.001,
        5,
        True,
    )
