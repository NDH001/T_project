import pandas as pd


def cleaning(df):
    df.drop(columns="customerID", inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"].replace("Yes", 1, inplace=True)
    df["Churn"].replace("No", 0, inplace=True)
    df["Churn"] = df["Churn"].astype("int")
    df = pd.get_dummies(df, dtype=int)
    return df
