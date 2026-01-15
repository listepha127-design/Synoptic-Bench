import argparse
import os
import re
import numpy as np
from datetime import datetime, timedelta
import h5py
import pickle
import gc
import sys
import glob

print("=== SCRIPT STARTING ===", flush=True)
print(f"Python path: {sys.executable}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)

# --- CONFIGURATION ---
# Update these paths to match your actual data locations
AFD_DIRECTORY = '/scratch/hay3fm/afd_data_cleaned/'
GFS_DIRECTORY = '/scratch/hay3fm/gfs_zarr/Multimodal/gfs_2019.h5'  # Directory containing your individual .h5 files
CHECKPOINT_DIR = '/scratch/hay3fm/checkpoints/'
OUTPUT_DIR = '/scratch/hay3fm/gfs_zarr/Multimodal/'

# Map raw variables in your .h5 files to the names used in your other data
# Adjust the keys (left side) if your raw files have different names
VAR_MAPPING = {
    'HGT_L100': 'z',        # Geopotential Height
    'TMP_L103': 't2m',      # 2m Temperature
    'U_GRD_L100': 'u850',   # U Wind
    'V_GRD_L100': 'v850',   # V Wind
    'PRATE_L1_Avg_1': 'tp'  # Total Precip
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process weather data for specific date range')
    parser.add_argument('--start_date', type=str, default='2019-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default='2020-01-01', help='End date YYYY-MM-DD')
    return parser.parse_args()

def save_checkpoint(data, checkpoint_name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(checkpoint_name):
    path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"Checkpoint loaded: {path}")
        return data
    return None

def parse_afd_filename(filename):
    """Parses AFD filenames: AFD_STATION_..._YYYYMMDDHHMM.txt"""
    parts = filename[:-4].split('_')
    if (len(parts) >= 5 and parts[0].startswith('AFD') and len(parts[4]) == 12 and parts[4].isdigit()):
        return parts[0][3:], parts[4]  # station_id, timestamp
    return None, None

def load_afd_files_chunked(chunk_size=10000, start_date_str=None, end_date_str=None):
    """Loads and filters AFD files by date."""
    print("Loading AFD files...")
    
    gfs_start = datetime.strptime(start_date_str, '%Y-%m-%d')
    gfs_end = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    checkpoint_name = f'afd_files_{gfs_start.strftime("%Y%m%d")}_{gfs_end.strftime("%Y%m%d")}'
    checkpoint = load_checkpoint(checkpoint_name)
    if checkpoint: return checkpoint

    filtered_files = []
    
    # Walk through directory
    for root, _, files in os.walk(AFD_DIRECTORY):
        for file in files:
            if file.endswith('.txt'):
                station_id, timestamp = parse_afd_filename(file)
                if station_id and timestamp:
                    try:
                        dt = datetime.strptime(timestamp, '%Y%m%d%H%M')
                        if gfs_start <= dt <= gfs_end:
                            filtered_files.append({
                                'station_id': station_id,
                                'run_time': dt,
                                'file_path': os.path.join(root, file)
                            })
                    except:
                        continue
    
    print(f"Found {len(filtered_files):,} AFD files in date range.")
    save_checkpoint(filtered_files, checkpoint_name)
    return filtered_files

def index_gfs_files(gfs_dir):
    """
    Scans the GFS directory and builds an index:
    Index structure: { run_time_datetime: { forecast_hour_int: file_path } }
    """
    print(f"Indexing GFS files in {gfs_dir}...")
    gfs_index = {}
    
    # Regex to find timestamp (10 digits) and forecast hour (fXXX)
    # Assumes filename format like: ...2018112100...f012...
    pattern = re.compile(r'(\d{10}).*f(\d{3})')
    
    files = glob.glob(os.path.join(gfs_dir, "*.h5"))
    print(f"Found {len(files)} .h5 files. Parsing filenames...")
    
    for fpath in files:
        fname = os.path.basename(fpath)
        match = pattern.search(fname)
        if match:
            try:
                dt_str = match.group(1)
                hour_str = match.group(2)
                
                run_time = datetime.strptime(dt_str, '%Y%m%d%H')
                forecast_hour = int(hour_str)
                
                if run_time not in gfs_index:
                    gfs_index[run_time] = {}
                
                gfs_index[run_time][forecast_hour] = fpath
            except ValueError:
                continue
                
    print(f"Indexed {len(gfs_index)} unique GFS run times.")
    return gfs_index

def find_matching_pairs(afd_list, gfs_index):
    """Matches AFDs to the closest available GFS run time."""
    print("Matching AFDs to GFS runs...")
    matches = []
    
    # Get sorted list of available GFS run times
    available_runs = sorted(gfs_index.keys())
    if not available_runs:
        print("No GFS runs found in index!")
        return []

    # Simple logic: Find closest run time within 12 hours
    # Since runs are sorted, we can use simple comparisons or binary search.
    # Here we iterate for simplicity as N is small.
    
    processed = 0
    for afd in afd_list:
        afd_time = afd['run_time']
        
        # Find closest run
        closest_run = min(available_runs, key=lambda x: abs(x - afd_time))
        diff = abs(afd_time - closest_run)
        
        if diff <= timedelta(hours=12):
            matches.append({
                'station_id': afd['station_id'],
                'afd_time': afd['run_time'],
                'file_path': afd['file_path'],
                'gfs_run_time': closest_run,
                'time_diff_hours': diff.total_seconds() / 3600
            })
        
        processed += 1
        if processed % 10000 == 0:
            print(f"Matched {processed}/{len(afd_list)} AFDs...")

    print(f"Found {len(matches):,} valid matches.")
    return matches

def group_matches_by_forecast(matches):
    """Groups matches by their assigned GFS run time to optimize I/O."""
    groups = {}
    for m in matches:
        run_time = m['gfs_run_time']
        if run_time not in groups:
            groups[run_time] = []
        groups[run_time].append(m)
    return groups

def get_station_coordinates():
    # Condensed version of your station dictionary
    return {
        'ABR': {'lat': 45.5, 'lon': 261.5}, 'ALY': {'lat': 42.75, 'lon': 286.25}, 'ABQ': {'lat': 35, 'lon': 253.25}, 'AMA': {'lat': 35.25, 'lon': 258.25}, 'APX': {'lat': 45, 'lon': 275.25}, 'ARX': {'lat': 43.75, 'lon': 268.75}, 'AKQ': {'lat': 37, 'lon': 283}, 'EWX': {'lat': 30.25, 'lon': 262.25}, 'LWX': {'lat': 39.25, 'lon': 283.5}, 'BYZ': {'lat': 45.75, 'lon': 251.5}, 'BGM': {'lat': 42, 'lon': 284}, 'BMX': {'lat': 33.5, 'lon': 273.25}, 'BIS': {'lat': 46.75, 'lon': 259.25}, 'RNK': {'lat': 37.25, 'lon': 279.5}, 'BOI': {'lat': 43.5, 'lon': 243.75}, 'BOX': {'lat': 42.5, 'lon': 289}, 'BRO': {'lat': 26, 'lon': 262.5}, 'BUF': {'lat': 43, 'lon': 281}, 'BTV': {'lat': 44.5, 'lon': 286.75}, 'CAR': {'lat': 46.75, 'lon': 292}, 'CHS': {'lat': 32.75, 'lon': 280}, 'RLX': {'lat': 38.25, 'lon': 278.5}, 'CYS': {'lat': 41.25, 'lon': 255.25}, 'LOT': {'lat': 42, 'lon': 272.5}, 'CLE': {'lat': 41.5, 'lon': 278.25}, 'CAE': {'lat': 34, 'lon': 279}, 'CRP': {'lat': 27.75, 'lon': 262.5}, 'FWD': {'lat': 32.75, 'lon': 263.25}, 'BOU': {'lat': 39.75, 'lon': 255}, 'DMX': {'lat': 41.5, 'lon': 266.5}, 'DTX': {'lat': 42.25, 'lon': 277}, 'DDC': {'lat': 37.75, 'lon': 260}, 'DLH': {'lat': 46.75, 'lon': 267.75}, 'LKN': {'lat': 40.75, 'lon': 244.25}, 'EPZ': {'lat': 31.75, 'lon': 253.5}, 'EKA': {'lat': 40.75, 'lon': 235.75}, 'FGZ': {'lat': 35.25, 'lon': 248.25}, 'GGW': {'lat': 48.25, 'lon': 253.25}, 'GLD': {'lat': 39.25, 'lon': 258.25}, 'FGF': {'lat': 47.75, 'lon': 263}, 'GJT': {'lat': 39, 'lon': 251.5}, 'GRR': {'lat': 43, 'lon': 274.25}, 'GYX': {'lat': 44, 'lon': 289.75}, 'TFX': {'lat': 47.5, 'lon': 248.75}, 'GRB': {'lat': 44.5, 'lon': 272}, 'GSP': {'lat': 35, 'lon': 277.5}, 'GID': {'lat': 40.5, 'lon': 261.5}, 'HGX': {'lat': 29.25, 'lon': 265.25}, 'HUN': {'lat': 34.75, 'lon': 273.5}, 'IND': {'lat': 40, 'lon': 273.75}, 'JAN': {'lat': 32.25, 'lon': 269.75}, 'JKL': {'lat': 37.5, 'lon': 276.75}, 'JAX': {'lat': 30.25, 'lon': 278.5}, 'EAX': {'lat': 39, 'lon': 265.5}, 'KEY': {'lat': 24.5, 'lon': 278.25}, 'LCH': {'lat': 30.25, 'lon': 266.75}, 'VEF': {'lat': 36.25, 'lon': 244.75}, 'ILX': {'lat': 40.75, 'lon': 263.25}, 'LZK': {'lat': 34.75, 'lon': 267.75}, 'LOX': {'lat': 34, 'lon': 241.75}, 'LMK': {'lat': 38.25, 'lon': 274.25}, 'LUB': {'lat': 33.5, 'lon': 258.25}, 'MQT': {'lat': 46.5, 'lon': 272.75}, 'MFR': {'lat': 42.25, 'lon': 237.25}, 'MLB': {'lat': 28, 'lon': 279.5}, 'MEG': {'lat': 35, 'lon': 270}, 'MFL': {'lat': 25.75, 'lon': 279.75}, 'MAF': {'lat': 32, 'lon': 257.75}, 'MKX': {'lat': 43, 'lon': 272.25}, 'MSO': {'lat': 46.75, 'lon': 246}, 'MOB': {'lat': 30.75, 'lon': 272}, 'MRX': {'lat': 36.25, 'lon': 276.75}, 'PHI': {'lat': 40, 'lon': 285.25}, 'OHX': {'lat': 36.25, 'lon': 273.25}, 'LIX': {'lat': 30, 'lon': 269.75}, 'MHX': {'lat': 34.75, 'lon': 283.25}, 'OKX': {'lat': 40.75, 'lon': 286}, 'OUN': {'lat': 35.25, 'lon': 262.75}, 'IWX': {'lat': 41.25, 'lon': 274.25}, 'LBF': {'lat': 41, 'lon': 259.25}, 'OAX': {'lat': 41.25, 'lon': 264}, 'PAH': {'lat': 37, 'lon': 271.5}, 'FFC': {'lat': 33.5, 'lon': 275.5}, 'PDT': {'lat': 45.75, 'lon': 241.25}, 'PSR': {'lat': 33.5, 'lon': 247.75}, 'PBZ': {'lat': 40.5, 'lon': 280.25}, 'PIH': {'lat': 43, 'lon': 247.75}, 'PQR': {'lat': 45.5, 'lon': 237.25}, 'PUB': {'lat': 38.25, 'lon': 255.5}, 'DVN': {'lat': 41.5, 'lon': 269.5}, 'RAH': {'lat': 35.75, 'lon': 281.5}, 'UNR': {'lat': 44, 'lon': 256.75}, 'REV': {'lat': 39.5, 'lon': 240.25}, 'RIW': {'lat': 43, 'lon': 251.5}, 'STO': {'lat': 38.5, 'lon': 238.5}, 'SLC': {'lat': 40.75, 'lon': 248}, 'SJT': {'lat': 31.5, 'lon': 259.5}, 'SGX': {'lat': 32.75, 'lon': 242.75}, 'MTR': {'lat': 37.75, 'lon': 237.5}, 'HNX': {'lat': 36.25, 'lon': 240.5}, 'TJSJ': {'lat': 18.5, 'lon': 293.75}, 'SEW': {'lat': 47.75, 'lon': 237.75}, 'SHV': {'lat': 32.5, 'lon': 266.25}, 'FSD': {'lat': 43.75, 'lon': 263.25}, 'OTX': {'lat': 47.75, 'lon': 242.75}, 'SGF': {'lat': 37.25, 'lon': 266.75}, 'CTP': {'lat': 40.75, 'lon': 282.25}, 'LSX': {'lat': 38.5, 'lon': 269.75}, 'TAE': {'lat': 30.5, 'lon': 275.75}, 'TBW': {'lat': 27.75, 'lon': 277.5}, 'TOP': {'lat': 39, 'lon': 264.25}, 'TWC': {'lat': 32.25, 'lon': 249.25}, 'TSA': {'lat': 36, 'lon': 264.25}, 'MPX': {'lat': 45, 'lon': 266.75}, 'ICT': {'lat': 37.75, 'lon': 262.75}, 'ILM': {'lat': 34.25, 'lon': 282}, 'ILN': {'lat': 39.5, 'lon': 276.25
    }}

def process_and_save_chunks(forecast_groups, gfs_index, output_path, chunk_size=500):
    """Processes groups and saves to HDF5 incrementally."""
    
    # Required forecast hours
    forecast_hours = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72]
    
    nws_stations = get_station_coordinates()
    
    # Initialize Global Lat/Lon from the very first valid file found
    first_run = list(forecast_groups.keys())[0]
    first_file = gfs_index[first_run].get(forecast_hours[0])
    
    if not first_file or not os.path.exists(first_file):
        print("Error: Could not find a valid GFS file to read global coordinates.")
        return

    with h5py.File(first_file, 'r') as f:
        # Assuming standard names 'lat'/'lon' or 'latitude'/'longitude'
        # Adjust 'lat'/'lon' keys if your raw files use different names
        if 'lat' in f:
            lat_global = f['lat'][:]
            lon_global = f['lon'][:]
        elif 'latitude' in f:
            lat_global = f['latitude'][:]
            lon_global = f['longitude'][:]
        else:
            # Fallback for checking shapes if keys are weird
            print("Warning: Could not auto-detect lat/lon names. Checking keys:", list(f.keys()))
            return

    # Create output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as hf:
        # Save Global Metadata
        hf.create_dataset('lat_global', data=lat_global.astype(np.float32))
        hf.create_dataset('lon_global', data=lon_global.astype(np.float32))
        hf.attrs['creation_date'] = datetime.now().isoformat()
        hf.attrs['variables'] = list(VAR_MAPPING.values())

        sample_counter = 0
        
        # Iterate over GFS Runs (Groups)
        sorted_runs = sorted(forecast_groups.keys())
        
        for run_idx, run_time in enumerate(sorted_runs):
            afds_in_run = forecast_groups[run_time]
            print(f"Processing Run {run_time} ({run_idx+1}/{len(sorted_runs)}) - {len(afds_in_run)} AFDs")
            
            # 1. Load all required forecast steps for this run into memory
            # This prevents opening/closing files 1000 times for 1000 AFDs
            run_data_cache = {} # { hour: { var_name: numpy_array } }
            valid_run = True
            
            for hour in forecast_hours:
                file_path = gfs_index[run_time].get(hour)
                if not file_path or not os.path.exists(file_path):
                    valid_run = False
                    print(f"  Missing file for f{hour:03d}, skipping run.")
                    break
                
                try:
                    with h5py.File(file_path, 'r') as src:
                        step_data = {}
                        for raw_name, out_name in VAR_MAPPING.items():
                            if raw_name in src:
                                # Handle dimensions (ensure 2D lat/lon)
                                data = src[raw_name][:]
                                if data.ndim == 4: data = data[0,0,:,:]
                                elif data.ndim == 3: data = data[0,:,:]
                                step_data[out_name] = data.astype(np.float32)
                        run_data_cache[hour] = step_data
                except Exception as e:
                    print(f"  Error reading {os.path.basename(file_path)}: {e}")
                    valid_run = False
                    break
            
            if not valid_run: continue
            
            # 2. Build Samples for this run
            for afd in afds_in_run:
                station_id = afd['station_id']
                if station_id not in nws_stations: continue
                
                # Load Text
                try:
                    with open(afd['file_path'], 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                except: continue
                
                if not text: continue
                
                # Create HDF5 Group
                gname = f'sample_{sample_counter:06d}'
                grp = hf.create_group(gname)
                
                # Metadata
                grp.create_dataset('station_id', data=station_id.encode('utf-8'))
                grp.create_dataset('afd_text', data=text.encode('utf-8'))
                grp.create_dataset('afd_time', data=str(afd['afd_time']).encode('utf-8'))
                grp.create_dataset('gfs_run_time', data=str(run_time).encode('utf-8'))
                
                # Save Forecasts
                fcst_grp = grp.create_group('forecasts')
                for hour, data_dict in run_data_cache.items():
                    h_grp = fcst_grp.create_group(f'f_{hour}')
                    for var, arr in data_dict.items():
                        h_grp.create_dataset(var, data=arr, compression='gzip', compression_opts=4)
                
                sample_counter += 1
                
                # Flush every 100 samples to keep memory low
                if sample_counter % 100 == 0:
                    hf.flush()
                    
            # Clear cache for this run
            del run_data_cache
            gc.collect()

    print(f"Processing Complete. Saved {sample_counter} samples to {output_path}")

def main():
    args = parse_arguments()
    
    # 1. Load AFDs
    afds = load_afd_files_chunked(start_date_str=args.start_date, end_date_str=args.end_date)
    if not afds: return

    # 2. Index GFS Files
    gfs_index = index_gfs_files(GFS_DIRECTORY)
    if not gfs_index: return

    # 3. Match
    matches = find_matching_pairs(afds, gfs_index)
    if not matches: return

    # 4. Group
    groups = group_matches_by_forecast(matches)

    # 5. Process and Save
    date_suffix = f"{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}"
    output_filename = f'training_data_matched_{date_suffix}.hdf5'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    process_and_save_chunks(groups, gfs_index, output_path)

if __name__ == '__main__':
    main()