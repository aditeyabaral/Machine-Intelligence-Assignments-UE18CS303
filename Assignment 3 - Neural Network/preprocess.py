import numpy as np
import pandas as pd


def getMissingValueReplacer(x, metric):
    '''
    Obtain the replacing value in a series by passing the replacment metric. 
    Accepted metrics are median and mode
    '''
    if metric == "median":
        replacing_value = np.nanmedian(x)
    if metric == "mode":
        replacing_value = max(x, key=x.tolist().count)
    return replacing_value


def preprocessData(df):
    '''
    Preprocessing pipeline for a dataframe
    '''
    replacement_type = {    # Assigning replacement metrics for each column
        "Community": "mode",
        "Age": "median",
        "Residence": "mode",
        "Education": "mode",
        "BP": "median",
        "HB": "median",
        "Delivery phase": "mode",
        "Weight": "median"
    }

    '''Iterate through each column and replace NaN values using the associated metric'''

    for col in df.columns:
        if df[col].isnull().any():
            replacing_value = getMissingValueReplacer(
                df[col].values, replacement_type[col])
            df[col].fillna(replacing_value, inplace=True)

    '''Encoding Delivery phase as binary encoded integers'''

    df["Delivery phase"] = df["Delivery phase"].apply(
        lambda x: x == 1).astype(int)
    df["Residence"] = df["Residence"].apply(lambda x: x == 1).astype(int)

    '''Encoding Community as one hot encoded integers'''

    original = df["Community"].values-1
    encoded = pd.get_dummies(original, prefix="Community")
    df2 = pd.concat((df, encoded), axis=1)

    '''
    Education is dropped because all the rows have the same value and hence offer 0 variance to the model.
    Community is dropped since it has already been encoded as one-hot variables.
    '''
    df2.drop(columns=["Community", "Education"], inplace=True)
    df2.to_csv("../data/[CLEANED]LBW_Dataset.csv", index=False)

    print("PREPROCESSING COMPLETED. FILE SAVED SUCCESSFULLY.")


#   Read DataFrame and preprocess it
df = pd.read_csv("../data/LBW_Dataset.csv")
preprocessData(df)
