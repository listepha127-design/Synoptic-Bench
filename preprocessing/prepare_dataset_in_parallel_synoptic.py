import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import os
import json
import re
from tqdm import tqdm
import argparse

# --- CONFIGURATION ---
DAYS_MAP = {
    'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6,
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
}

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def get_day_distance(issue_idx, mention_idx):
    return (mention_idx - issue_idx + 7) % 7

def determine_issue_info(afd_group):
    """
    Scans AFDs to find the Issue Day, Month, Date, and YEAR.
    Returns: (day_idx, month_idx, date_str_with_year)
    """
    station_keys = list(afd_group.keys())
    
    date_pattern = re.compile(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\s+([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})', re.IGNORECASE)
    
    for sid in station_keys:
        try:
            # Safely extract text for both old (bytes) and new (string) formats
            text_data = afd_group[sid]['afd_text'][()]
            raw_text = text_data.decode('utf-8', errors='ignore') if isinstance(text_data, bytes) else text_data
            
            header = raw_text[:500] 
            match = date_pattern.search(header)
            
            if match:
                day_str = match.group(1).lower()
                month_str = match.group(2).lower()
                day_num = match.group(3).zfill(2) # 09
                year_str = match.group(4)         # 2017
                
                day_idx = DAYS_MAP.get(day_str[:3], DAYS_MAP.get(day_str))
                month_idx = MONTH_MAP.get(month_str)
                
                date_clean = f"{month_str}{day_num}_{year_str}"
                
                if day_idx is not None and month_idx is not None:
                    return day_idx, month_idx, date_clean
        except:
            continue
            
    return None, None, "unknown_date"

def extract_synoptic_sentences_filtered(afd_text, issue_day_idx=None, max_lead_days=1):
    INCLUSION_KEYWORDS = ['trough', 'trof', 'the low', 'this low', 'upper level low', 'low pressure', 'low-pressure', 'upper low', 'cyclone', 'closed low', 'cut-off low', 'troughing', 'ridge', 'the high', 'this high', 'upper level high', 'high pressure', 'high-pressure', 'upper high', 'anticyclone', 'blocking', 'ridging', 'cold front', 'warm front', 'warmer', 'cooler', 'freezing'
    ]
    EXCLUSION_KEYWORDS = [
        'ECMWF', 'EURO', 'HRRR', 'ECCC', 'CMC', 'GEM', 'NAM', 'UKMET', 'ICON', 'RAP', 'SREF', 'HREF',
        'shortwave', 'short-wave', 'sfc trough', 'surface trough', 
        'sfc ridge', 'surface ridge', 'surface low', 'sfc low', 'surface high', 'sfc high'
    ]
    
    sentences = afd_text.replace('...', '.').split('.')
    synoptic_sentences = []
    stop_triggered = False
    
    for sentence in sentences:
        if stop_triggered: break
        clean_sentence = sentence.strip()
        if not clean_sentence: continue
        lower_sentence = clean_sentence.lower()
        
        # Temporal Sentinel
        if issue_day_idx is not None:
            mentioned_days = re.findall(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', lower_sentence)
            for day_str in mentioned_days:
                dist = get_day_distance(issue_day_idx, DAYS_MAP[day_str])
                if dist > max_lead_days:
                    stop_triggered = True; break 
            if stop_triggered: break

        if any(bad in lower_sentence for bad in EXCLUSION_KEYWORDS): continue
        if not any(good in lower_sentence for good in INCLUSION_KEYWORDS): continue
        synoptic_sentences.append(clean_sentence + '.')
        
    return " ".join(synoptic_sentences)

def generate_anomaly_plot(plot_data, lats, lons, output_path, local_extent):
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), subplot_kw={'projection': projection})
    
    synoptic_extent = [200.25, 300.00, 15.25, 65.00]
    ax.set_extent(synoptic_extent, crs=projection)
    
    # Anomaly Fill (-5 to 5) NO COLORBAR
    levels = np.linspace(-5, 5, 41)
    ax.contourf(
        lons, lats, plot_data['t2m_anomaly'], 
        cmap='coolwarm', 
        levels=levels, 
        extend='both', 
        transform=projection
    )

    # GH500 Contours
    gh_contours = ax.contour(
        lons, lats, plot_data['avg_GH500'], 
        colors='black', 
        levels=np.arange(5100, 6001, 60), 
        transform=projection
    )
    ax.clabel(gh_contours, inline=True, fmt='%d', fontsize=10)

    # Wind Barbs
    skip = 20
    ax.barbs(
        lons[::skip], lats[::skip], 
        plot_data['avg_U850'][::skip, ::skip], 
        plot_data['avg_V850'][::skip, ::skip], 
        length=6, 
        transform=projection
    )
    
    # Local Bounding Box
    ax.add_patch(mpatches.Rectangle(
        xy=[local_extent[0], local_extent[2]],
        width=(local_extent[1] - local_extent[0]),
        height=(local_extent[3] - local_extent[2]),
        edgecolor='yellow', 
        facecolor='none', 
        linewidth=2, 
        transform=projection
    ))
    
    ax.set_title('Synoptic Pattern & Temperature Anomaly (0-48h Mean)', fontsize=12)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# Helper function to extract forecast variables safely
def get_var(fcst_group, old_name, new_name):
    if old_name in fcst_group:
        return fcst_group[old_name][:]
    elif new_name in fcst_group:
        return fcst_group[new_name][:]
    else:
        raise KeyError(f"Neither {old_name} nor {new_name} found in forecast!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_file", type=str, required=True)
    parser.add_argument("--climo_file", type=str, required=True)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Climatology
    print(f"Chunk {args.chunk_id}: Loading Climatology...")
    with h5py.File(args.climo_file, 'r') as f_clim:
        climo_means = f_clim['monthly_t2m_means'][:]

    # 2. Work Items
    all_work_items = []
    with h5py.File(args.hdf5_file, 'r') as f:
        for sample_key in f.keys():
            if not sample_key.startswith('sample_'): continue
            afd_group = f[sample_key]['associated_afds']
            for afd_key in afd_group.keys():
                all_work_items.append((sample_key, afd_key))
                
    total_items = len(all_work_items)
    items_per_chunk = (total_items + args.total_chunks - 1) // args.total_chunks
    start_index = args.chunk_id * items_per_chunk
    end_index = min(start_index + items_per_chunk, total_items)
    work_for_this_job = all_work_items[start_index:end_index]
    
    if not work_for_this_job:
        print(f"Chunk {args.chunk_id} has no work.")
        exit()
        
    print(f"Chunk {args.chunk_id} processing {len(work_for_this_job)} AFDs.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    training_data_manifest = []
    
    current_sample_key = None
    current_issue_day_idx = None
    current_issue_month_idx = None
    current_date_str = "unknown_date"

    with h5py.File(args.hdf5_file, 'r') as f:
        lats = f['lat_global'][:]
        lons = f['lon_global'][:]
        
        lead_times_to_load = [f'f_{h}' for h in range(3, 49, 3)]

        for sample_key, afd_key in tqdm(work_for_this_job, desc=f"Chunk {args.chunk_id}"):
            sample = f[sample_key]
            afd_data = sample['associated_afds'][afd_key]
            
            # --- DATE DETECTION ---
            if sample_key != current_sample_key:
                current_sample_key = sample_key
                current_issue_day_idx, current_issue_month_idx, current_date_str = determine_issue_info(sample['associated_afds'])
            
            if current_issue_month_idx is None:
                continue

            # --- LOAD FORECAST ---
            try:
                raw_data = {'GH500': [], 'U850': [], 'V850': [], 't2m': []}
                valid_sample = True
                
                for lt in lead_times_to_load:
                    if lt not in sample['forecasts']:
                        valid_sample = False; break
                    grp = sample['forecasts'][lt]
                    
                    # Safely load variables using fallback names
                    raw_data['GH500'].append(get_var(grp, 'GH500', 'z'))
                    raw_data['U850'].append(get_var(grp, 'U850', 'u850'))
                    raw_data['V850'].append(get_var(grp, 'V850', 'v850'))
                    raw_data['t2m'].append(get_var(grp, 't2m', 't2m'))
                
                if not valid_sample: continue

                avg_gh = np.mean(np.stack(raw_data['GH500']), axis=0)
                avg_u = np.mean(np.stack(raw_data['U850']), axis=0)
                avg_v = np.mean(np.stack(raw_data['V850']), axis=0)
                avg_t2m_raw = np.mean(np.stack(raw_data['t2m']), axis=0)
                
            except KeyError: 
                continue

            # --- CALC ANOMALY ---
            t2m_anomaly = avg_t2m_raw - climo_means[current_issue_month_idx]
            
            plot_data = {
                'avg_GH500': avg_gh,
                'avg_U850': avg_u,
                'avg_V850': avg_v,
                't2m_anomaly': t2m_anomaly
            }

            # --- TEXT PROCESSING ---
            text_bytes = afd_data['afd_text'][()]
            afd_text = text_bytes.decode('utf-8') if isinstance(text_bytes, bytes) else text_bytes
            
            synoptic_text = extract_synoptic_sentences_filtered(
                afd_text, 
                issue_day_idx=current_issue_day_idx, 
                max_lead_days=1
            )
            
            if not synoptic_text: continue
                
            # --- STATION EXTRACTION ---
            if 'station_id' in afd_data:
                sid_bytes = afd_data['station_id'][()]
                station_id = sid_bytes.decode('utf-8') if isinstance(sid_bytes, bytes) else sid_bytes
            else:
                station_id = afd_key # Fallback for 2015 format
            
            # --- IMAGE GENERATION ---
            filename_base = f"{station_id}_{current_date_str}_{sample_key}"
            image_filename = f"{filename_base}.png"
            image_path = os.path.join(args.output_dir, image_filename)
            
            if not os.path.exists(image_path):
                station_lat = afd_data['station_lat'][()]
                station_lon = afd_data['station_lon'][()]
                buffer = 2.5
                local_extent = [station_lon - buffer, station_lon + buffer, station_lat - buffer, station_lat + buffer]
                
                generate_anomaly_plot(plot_data, lats, lons, image_path, local_extent)
            
            training_data_manifest.append({
                "id": filename_base,
                "image": image_filename, 
                "conversations": [
                    { "from": "human", "value": "Analyze this weather chart showing the synoptic pattern and temperature anomalies, then generate a forecast summary." },
                    { "from": "gpt", "value": synoptic_text }
                ]
            })

    partial_json_path = args.output_json.replace('.json', f'_part_{args.chunk_id}.json')
    with open(partial_json_path, 'w') as f:
        json.dump(training_data_manifest, f, indent=2)
        
    print(f"Chunk {args.chunk_id} finished. Saved to {partial_json_path}")