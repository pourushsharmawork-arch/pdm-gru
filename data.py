import numpy as np

def process_targets(data_length:int, max_rul=None)->np.array:
    if max_rul is not None:
        duration = data_length-max_rul
        if duration <= 0:
            return np.arange(data_length-1, -1, -1)
        else:
            return np.append(max_rul*np.ones(shape = (duration,)), np.arange(max_rul-1, -1, -1))
    else:
        return np.arange(data_length-1, -1, -1)

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

def process_test(test_data:np.array, window_length:int, shift:int, num_windows:int=1):
    max_batches = int(np.floor((len(test_data) - window_length)/shift)) + 1
    if max_batches < num_windows:
        req_len = (max_batches -1)* shift + window_length
        batched_data = process_input(input_data=test_data[-req_len:,:], target_data=None, window_length=window_length, shift=shift)
        return batched_data, max_batches
    else:
        req_len = (num_windows-1)*shift + window_length
        batched_data = process_input(input_data=test_data[-req_len:, :], target_data=None, window_length=window_length, shift=shift)
        return batched_data, num_windows