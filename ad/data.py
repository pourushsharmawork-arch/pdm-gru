import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

COLUMNS_TO_DROP = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]

def load_fd001(data_dir:str, subset:str="FD001")->pd.DataFrame:
    train_data = pd.read_csv(os.path.join(data_dir,"train",f"train_{subset}.txt"),sep=r"\s+",header=None)
    # print(train_data.info())
    return train_data

def add_rul(data:pd.DataFrame)->pd.DataFrame:
    data = data.copy()
    max_cycles = data.groupby(0)[1].transform("max")
    data["RUL"]=max_cycles-data[1]
    return data

def split_engines(data:pd.DataFrame, train_fraction:float=0.70, val_fraction:float=0.15, seed:int=42):
    engine_ids = data[0].unique()
    print(f"{len(engine_ids)} unique engines.")

    rng = np.random.default_rng(seed)
    engine_ids = rng.permutation(engine_ids)
    train_end = int(len(engine_ids)*train_fraction)
    val_end = train_end + int(len(engine_ids)*val_fraction)
    train_ids = engine_ids[:train_end]
    val_ids = engine_ids[train_end:val_end]
    test_ids = engine_ids[val_end:]
    return train_ids, val_ids, test_ids

def split_data(data,train_ids,val_ids,test_ids):
    train_data = data[data[0].isin(train_ids)].copy()
    val_data = data[data[0].isin(val_ids)].copy()
    test_data = data[data[0].isin(test_ids)].copy()
    return train_data, val_data, test_data

def select_healthy(data,healthy_rul_threshold=100):
    healthy_data = data[
        data["RUL"] >= healthy_rul_threshold
    ].copy()
    return healthy_data

def preprocess(train_data,val_data,test_data):
    if len(train_data) == 0:
        raise ValueError("Healthy training data is empty")

    if len(val_data) == 0:
        raise ValueError("Healthy validation data is empty")

    if len(test_data) == 0:
        raise ValueError("Healthy test data is empty")

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_data.drop(columns=COLUMNS_TO_DROP + ["RUL"])
    )

    val_scaled = scaler.transform(
        val_data.drop(columns=COLUMNS_TO_DROP + ["RUL"])
    )

    test_scaled = scaler.transform(
        test_data.drop(columns=COLUMNS_TO_DROP + ["RUL"])
    )

    train_data = pd.DataFrame(
        np.c_[train_data[0].values,train_data[1].values,train_scaled]
    )

    val_data = pd.DataFrame(
        np.c_[val_data[0].values,val_data[1].values,val_scaled]
    )

    test_data = pd.DataFrame(
        np.c_[test_data[0].values,test_data[1].values,test_scaled]
    )

    return train_data,val_data,test_data,scaler

def create_windows(data,window_length,shift):
    windows = []
    for engine_id in data[0].unique():
        engine_data = data[data[0] == engine_id]
        sensor_data = engine_data.drop(columns=[0, 1]).values
        engine_windows = process_input(input_data=sensor_data,target_data=None,window_length=window_length,shift=shift)
        windows.append(engine_windows)
    return np.concatenate(windows,axis=0)

def filter_eligible_engines(data,healthy_rul_threshold=100,window_length=30):
    required_rul = healthy_rul_threshold + window_length - 1
    max_rul = data.groupby(0)["RUL"].max()
    eligible_ids = max_rul[max_rul >= required_rul].index
    return data[data[0].isin(eligible_ids)].copy()

def process_input(input_data:np.array, target_data:np.array, window_length:int, shift:int)->np.array:
    if input_data.ndim != 2:
        raise ValueError(f"input data must be 2-dimensional, received:{input_data.shape}")
    num_batches = int(np.floor((len(input_data) - window_length)/shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches*window_length*num_features).reshape(
        num_batches,
        window_length,
        num_features
    )

    if target_data is None:
        for batch in range(num_batches):
            output_data[batch,:,:]=input_data[(0+shift*batch):(0+shift*batch+window_length),:]
        return output_data
    else:
        if target_data.ndim != 1:
            raise ValueError(f"target data must be 1-dimensional, received {target_data.shape}")
        if len(input_data) != len(target_data):
            raise ValueError(f"input and target data must have the same length, received: input:{len(input_data)}, target:{len(target_data)}")

        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch,:,:]=input_data[(0+shift*batch):(0+shift*batch+window_length),:]
            output_targets[batch]=target_data[(shift*batch + (window_length-1))]
        return output_data, output_targets

def prepare_data(data_dir,subset="FD001",healthy_rul_threshold=100,window_length=30,shift=1,seed=42):

    data = load_fd001(data_dir,subset)
    data = add_rul(data)

    data = filter_eligible_engines(data,healthy_rul_threshold, window_length)
    train_ids,val_ids,test_ids = split_engines(data,seed=seed)

    train_data, val_data, test_data = (split_data(data,train_ids,val_ids,test_ids))
    train_data = select_healthy(train_data,healthy_rul_threshold)
    val_data = select_healthy(val_data,healthy_rul_threshold)
    test_data = select_healthy(test_data,healthy_rul_threshold)

    print("Healthy train rows:",len(train_data))
    print("Healthy val rows:",len(val_data))
    print("Healthy test rows:",len(test_data))

    print("Train engines:",train_data[0].nunique())
    print("Val engines:",val_data[0].nunique())
    print("Test engines:",test_data[0].nunique())

    (train_data,val_data,test_data,scaler) = preprocess(train_data,val_data,test_data)

    X_train = create_windows(train_data,window_length,shift)
    X_val = create_windows(val_data,window_length,shift)
    X_test_normal = create_windows(test_data,window_length,shift)

    return (X_train,X_val,X_test_normal,scaler)
