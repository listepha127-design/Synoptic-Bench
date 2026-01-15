import h5py
import pandas as pd
import os

def filter_and_unify_hdf5_chunked(input_file_path, output_file_path, chunk_size=5000):
    """
    Reads a large HDF5 file, filters for complete samples,
    and saves them in chunks to a new, unified HDF5 file.
    """
    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists. Skipping filtering.")
        return

    print(f"Opening input file: {input_file_path}")
    
    with h5py.File(input_file_path, 'r') as hf_in:
        all_keys = sorted([key for key in hf_in.keys() if key.startswith('sample_') and isinstance(hf_in[key], h5py.Group)])
        total_samples = len(all_keys)
        print(f"Found {total_samples} sample groups.")
        
        master_columns_set = set(['afd_text', 'time'])
        print("Building master column list from a large sample set...")
        for key in all_keys[:5000]:
            sample_group = hf_in[key]
            if 'forecasts' in sample_group:
                forecasts_group = sample_group['forecasts']
                for hour_key in forecasts_group.keys():
                    hour_group = forecasts_group[hour_key]
                    for var_name in hour_group.keys():
                        master_columns_set.add(f"{hour_key}_{var_name}")
        
        master_columns = sorted(list(master_columns_set))
        if len(master_columns) <= 2:
            raise ValueError("Could not find any forecast data to build the master column list.")
        print(f"Master list of columns determined: {master_columns}")

        chunk_counter = 0
        all_complete_dfs = []
        
        for i, key in enumerate(all_keys):
            sample_group = hf_in[key]
            data_dict = {}

            if 'afd_text' in sample_group:
                data_dict['afd_text'] = sample_group['afd_text'][()].decode('utf-8')
            if 'time' in sample_group:
                data_dict['time'] = sample_group['time'][()].decode('utf-8')

            has_forecasts = 'forecasts' in sample_group
            if has_forecasts:
                forecasts_group = sample_group['forecasts']
                for hour_key in sorted(forecasts_group.keys()):
                    hour_group = forecasts_group[hour_key]
                    for var_name in sorted(hour_group.keys()):
                        data_dict[f"{hour_key}_{var_name}"] = hour_group[var_name][()]

            if has_forecasts:
                df_sample = pd.DataFrame([data_dict], columns=master_columns)
                all_complete_dfs.append(df_sample)
            
            # Write to file in chunks to prevent memory errors
            if (i + 1) % chunk_size == 0 or i == total_samples - 1:
                if all_complete_dfs:
                    chunk_df = pd.concat(all_complete_dfs, ignore_index=True)
                    
                    mode = 'a'
                    if not os.path.exists(output_file_path):
                        mode = 'w'
                    
                    min_itemsize = {'afd_text': 50000, 'time': 50}
                    
                    chunk_df.to_hdf(
                        output_file_path, 
                        key='data', 
                        mode=mode, 
                        format='table', 
                        append=True, 
                        data_columns=True, 
                        min_itemsize=min_itemsize
                    )
                    
                    print(f"Saved chunk {chunk_counter} with {len(chunk_df)} samples to {output_file_path}.")
                    all_complete_dfs = [] # Clear the list after saving
                    chunk_counter += 1

    print("Data filtering and unification complete.")

input_file = '/scratch/hay3fm/gfs_zarr/Multimodal/final_training_data_multi_forecast.hdf5'
output_file = '/scratch/hay3fm/gfs_zarr/Multimodal/filtered_training_data.hdf5'
filter_and_unify_hdf5_chunked(input_file, output_file)