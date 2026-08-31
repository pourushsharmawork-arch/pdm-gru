import os
import numpy as np

def inject_bias(window,magnitude=2.0,rng=None):
    window = window.copy()
    if rng is None:
        rng = np.random.default_rng()

    sensor_id = rng.integers(window.shape[1])
    direction = rng.choice([-1.0,1.0])
    window[:,sensor_id] += direction*magnitude
    return window

def inject_spike(window,magnitude=4.0,rng=None):
    window = window.copy()
    if rng is None:
        rng = np.random.default_rng()

    time_id = rng.integers(window.shape[0])
    sensor_id = rng.integers(window.shape[1])
    direction = rng.choice([-1.0,1.0])
    window[time_id,sensor_id] += direction*magnitude
    return window

def inject_drift(window,magnitude=2.0,rng=None):
    window = window.copy()
    if rng is None:
        rng = np.random.default_rng()

    sensor_id = rng.integers(window.shape[1])
    direction = rng.choice([-1.0,1.0])
    drift = np.linspace(0,direction*magnitude,window.shape[0])
    window[:,sensor_id] += drift
    return window

def inject_noise(window,std=1.0,rng=None):
    window = window.copy()
    if rng is None:
        rng = np.random.default_rng()

    sensor_id = rng.integers(window.shape[1])
    noise = rng.normal(loc=0.0,scale=std,size=window.shape[0])
    window[:,sensor_id] += noise
    return window

def generate_anomaly(window,rng):
    anomaly_type = rng.choice(
        ["bias","spike","drift","noise"])
    if anomaly_type == "bias":
        anomaly = inject_bias(window,rng=rng)
    elif anomaly_type == "spike":
        anomaly = inject_spike(window,rng=rng)
    elif anomaly_type == "drift":
        anomaly = inject_drift(window,rng=rng)
    else:
        anomaly = inject_noise(window,rng=rng)
    return anomaly,anomaly_type

def generate_synthetic_data(X_normal,num_samples=None,seed=42):
    rng = np.random.default_rng(seed)
    if num_samples is None:
        num_samples = len(X_normal)

    if num_samples % 2 != 0:
        num_samples -= 1

    num_normal = num_samples//2
    num_anomalies = num_samples//2

    if num_normal > len(X_normal) or num_anomalies > len(X_normal):
        raise ValueError(
            f"Requested {num_samples} synthetic samples needs {num_normal} normal and "
            f"{num_anomalies} anomalous windows, but only {len(X_normal)} normal windows "
            f"are available. Pass num_samples <= {2*len(X_normal)}."
        )

    normal_ids = rng.choice(len(X_normal),size=num_normal,replace=False)
    anomaly_ids = rng.choice(len(X_normal),size=num_anomalies,replace=False)
    normal_data = X_normal[normal_ids].copy()
    anomaly_data = []
    anomaly_types = []

    for index in anomaly_ids:
        anomaly,anomaly_type = generate_anomaly(X_normal[index],rng)
        anomaly_data.append(anomaly)
        anomaly_types.append(anomaly_type)

    anomaly_data = np.stack(anomaly_data)
    X_synthetic = np.concatenate(
        [normal_data,anomaly_data],
        axis=0)
    y_synthetic = np.concatenate([np.zeros(num_normal,dtype=np.int64),np.ones(num_anomalies,dtype=np.int64)])
    sample_types = np.array(["normal"]*num_normal+ anomaly_types)
    shuffle_ids = rng.permutation(num_samples)
    X_synthetic = X_synthetic[shuffle_ids]
    y_synthetic = y_synthetic[shuffle_ids]
    sample_types = sample_types[shuffle_ids]
    return (X_synthetic,y_synthetic,sample_types)

def save_synthetic_data(X_synthetic,sample_types,output_dir,subset="FD001"):
    os.makedirs(output_dir,exist_ok=True)

    num_samples,window_length,num_features = X_synthetic.shape
    units = np.repeat(np.arange(1,num_samples+1),window_length)
    cycles = np.tile(np.arange(1,window_length+1),num_samples)
    features = X_synthetic.reshape(num_samples*window_length,num_features)
    rows = np.column_stack([units,cycles,features])

    data_path = os.path.join(output_dir,f"synthetic_{subset}.txt")
    labels_path = os.path.join(output_dir,f"labels_{subset}.txt")

    fmt = ["%d","%d"] + ["%.6f"]*num_features
    np.savetxt(data_path,rows,fmt=fmt)
    np.savetxt(labels_path,sample_types,fmt="%s")

    return data_path,labels_path

if __name__ == "__main__":
    from ad.data import prepare_data

    X_train,X_val,X_test_normal,scaler = prepare_data(data_dir="data",subset="FD001",healthy_rul_threshold=100,window_length=30,shift=1,seed=42)
    num_samples = min(4000, 2*len(X_test_normal))
    X_synthetic,y_synthetic,anomaly_types = generate_synthetic_data(X_test_normal,num_samples=num_samples,seed=42)

    print("X:",X_synthetic.shape)
    print("Normal:",np.sum(y_synthetic == 0))
    print("Anomaly:",np.sum(y_synthetic == 1))

    print(np.unique(anomaly_types,return_counts=True))

    data_path,labels_path = save_synthetic_data(X_synthetic,anomaly_types,output_dir="data/synthetic",subset="FD001")
    print("Saved data to:",data_path)
    print("Saved labels to:",labels_path)