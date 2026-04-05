import re
import json
import math
import os
import time
import argparse 
from collections import defaultdict
from tqdm import tqdm
import statistics
from collections import deque, defaultdict
USE_TIME = False 
LOCATION_WINDOW = 120 

NWS_STATIONS_COORDS = {
    'ABR': {'lat': 45.25, 'lon': 261.5}, 'ALY': {'lat': 42.75, 'lon': 286.25}, 'ABQ': {'lat': 35, 'lon': 253.25}, 'AMA': {'lat': 35.25, 'lon': 258.25}, 
    'APX': {'lat': 45, 'lon': 275.25}, 'ARX': {'lat': 43.75, 'lon': 268.75}, 'AKQ': {'lat': 37, 'lon': 283}, 'EWX': {'lat': 30.25, 'lon': 262.25}, 
    'LWX': {'lat': 39.25, 'lon': 283.5}, 'BYZ': {'lat': 45.75, 'lon': 251.5}, 'BGM': {'lat': 42, 'lon': 284}, 'BMX': {'lat': 33.5, 'lon': 273.25}, 
    'BIS': {'lat': 46.75, 'lon': 259.25}, 'RNK': {'lat': 37.25, 'lon': 279.5}, 'BOI': {'lat': 43.5, 'lon': 243.75}, 'BOX': {'lat': 42.5, 'lon': 289}, 
    'BRO': {'lat': 26, 'lon': 262.5}, 'BUF': {'lat': 43, 'lon': 281}, 'BTV': {'lat': 44.5, 'lon': 286.75}, 'CAR': {'lat': 46.75, 'lon': 292}, 
    'CHS': {'lat': 32.75, 'lon': 280}, 'RLX': {'lat': 38.25, 'lon': 278.5}, 'CYS': {'lat': 41.25, 'lon': 255.25}, 'LOT': {'lat': 42, 'lon': 272.5}, 
    'CLE': {'lat': 41.5, 'lon': 278.25}, 'CAE': {'lat': 34, 'lon': 279}, 'CRP': {'lat': 27.75, 'lon': 262.5}, 'FWD': {'lat': 32.75, 'lon': 263.25}, 
    'BOU': {'lat': 39.75, 'lon': 255}, 'DMX': {'lat': 41.5, 'lon': 266.5}, 'DTX': {'lat': 42.25, 'lon': 277}, 'DDC': {'lat': 37.75, 'lon': 260}, 
    'DLH': {'lat': 46.75, 'lon': 267.75}, 'LKN': {'lat': 40.75, 'lon': 244.25}, 'EPZ': {'lat': 31.75, 'lon': 253.5}, 'EKA': {'lat': 40.75, 'lon': 235.75}, 
    'FGZ': {'lat': 35.25, 'lon': 248.25}, 'GGW': {'lat': 48.25, 'lon': 253.25}, 'GLD': {'lat': 39.25, 'lon': 258.25}, 'FGF': {'lat': 47.75, 'lon': 263}, 
    'GJT': {'lat': 39, 'lon': 251.5}, 'GRR': {'lat': 43, 'lon': 274.25}, 'GYX': {'lat': 44, 'lon': 289.75}, 'TFX': {'lat': 47.5, 'lon': 248.75}, 
    'GRB': {'lat': 44.5, 'lon': 272}, 'GSP': {'lat': 35, 'lon': 277.5}, 'GID': {'lat': 40.5, 'lon': 261.5}, 'HGX': {'lat': 29.25, 'lon': 265.25}, 
    'HUN': {'lat': 34.75, 'lon': 273.5}, 'IND': {'lat': 40, 'lon': 273.75}, 'JAN': {'lat': 32.25, 'lon': 269.75}, 'JKL': {'lat': 37.5, 'lon': 276.75}, 
    'JAX': {'lat': 30.25, 'lon': 278.5}, 'EAX': {'lat': 39, 'lon': 265.5}, 'KEY': {'lat': 24.5, 'lon': 278.25}, 'LCH': {'lat': 30.25, 'lon': 266.75}, 
    'VEF': {'lat': 36.25, 'lon': 244.75}, 'ILX': {'lat': 40.75, 'lon': 263.25}, 'LZK': {'lat': 34.75, 'lon': 267.75}, 'LOX': {'lat': 34, 'lon': 241.75}, 
    'LMK': {'lat': 38.25, 'lon': 274.25}, 'LUB': {'lat': 33.5, 'lon': 258.25}, 'MQT': {'lat': 46.5, 'lon': 272.75}, 'MFR': {'lat': 42.25, 'lon': 237.25}, 
    'MLB': {'lat': 28, 'lon': 279.5}, 'MEG': {'lat': 35, 'lon': 270}, 'MFL': {'lat': 25.75, 'lon': 279.75}, 'MAF': {'lat': 32, 'lon': 257.75}, 
    'MKX': {'lat': 43, 'lon': 272.25}, 'MSO': {'lat': 46.75, 'lon': 246}, 'MOB': {'lat': 30.75, 'lon': 272}, 'MRX': {'lat': 36.25, 'lon': 276.75}, 
    'PHI': {'lat': 40, 'lon': 285.25}, 'OHX': {'lat': 36.25, 'lon': 273.25}, 'LIX': {'lat': 30, 'lon': 269.75}, 'MHX': {'lat': 34.75, 'lon': 283.25}, 
    'OKX': {'lat': 40.75, 'lon': 286}, 'OUN': {'lat': 35.25, 'lon': 262.75}, 'IWX': {'lat': 41.25, 'lon': 274.25}, 'LBF': {'lat': 41, 'lon': 259.25}, 
    'OAX': {'lat': 41.25, 'lon': 264}, 'PAH': {'lat': 37, 'lon': 271.5}, 'FFC': {'lat': 33.5, 'lon': 275.5}, 'PDT': {'lat': 45.75, 'lon': 241.25}, 
    'PSR': {'lat': 33.5, 'lon': 247.75}, 'PBZ': {'lat': 40.5, 'lon': 280.25}, 'PIH': {'lat': 43, 'lon': 247.75}, 'PQR': {'lat': 45.5, 'lon': 237.25}, 
    'PUB': {'lat': 38.25, 'lon': 255.5}, 'DVN': {'lat': 41.5, 'lon': 269.5}, 'RAH': {'lat': 35.75, 'lon': 281.5}, 'UNR': {'lat': 44, 'lon': 256.75}, 
    'REV': {'lat': 39.5, 'lon': 240.25}, 'RIW': {'lat': 43, 'lon': 251.5}, 'STO': {'lat': 38.5, 'lon': 238.5}, 'SLC': {'lat': 40.75, 'lon': 248}, 
    'SJT': {'lat': 31.5, 'lon': 259.5}, 'SGX': {'lat': 32.75, 'lon': 242.75}, 'MTR': {'lat': 37.75, 'lon': 237.5}, 'HNX': {'lat': 36.25, 'lon': 240.5}, 
    'TJSJ': {'lat': 18.5, 'lon': 293.75}, 'SEW': {'lat': 47.75, 'lon': 237.75}, 'SHV': {'lat': 32.5, 'lon': 266.25}, 'FSD': {'lat': 43.75, 'lon': 263.25}, 
    'OTX': {'lat': 47.75, 'lon': 242.75}, 'SGF': {'lat': 37.25, 'lon': 266.75}, 'CTP': {'lat': 40.75, 'lon': 282.25}, 'LSX': {'lat': 38.5, 'lon': 269.75}, 
    'TAE': {'lat': 30.5, 'lon': 275.75}, 'TBW': {'lat': 27.75, 'lon': 277.5}, 'TOP': {'lat': 39, 'lon': 264.25}, 'TWC': {'lat': 32.25, 'lon': 249.25}, 
    'TSA': {'lat': 36, 'lon': 264.25}, 'MPX': {'lat': 45, 'lon': 266.75}, 'ICT': {'lat': 37.75, 'lon': 262.75}, 'ILM': {'lat': 34.25, 'lon': 282}, 
    'ILN': {'lat': 39.5, 'lon': 276.25}
}


STATION_NAME_MAP = {
  "ABR": "South Dakota", "ALY": "New York", "ABQ": "New Mexico", "AMA": "Texas", 
  "APX": "Michigan", "ARX": "Wisconsin", "AKQ": "Virginia", "EWX": "Texas", 
  "LWX": "Maryland", "BYZ": "Montana", "BGM": "New York", "BMX": "Alabama", 
  "BIS": "North Dakota", "RNK": "Virginia", "BOI": "Idaho", "BOX": "Massachusetts", 
  "BRO": "Texas", "BUF": "New York", "BTV": "Vermont", "CAR": "Maine", 
  "CHS": "South Carolina", "RLX": "West Virginia", "CYS": "Wyoming", "LOT": "Illinois", 
  "CLE": "Ohio", "CAE": "South Carolina", "CRP": "Texas", "FWD": "Texas", 
  "BOU": "Colorado", "DMX": "Iowa", "DTX": "Michigan", "DDC": "Kansas", 
  "DLH": "Minnesota", "LKN": "Nevada", "EPZ": "Texas", "EKA": "California", 
  "FGZ": "Arizona", "GGW": "Montana", "GLD": "Kansas", "FGF": "North Dakota", 
  "GJT": "Colorado", "GRR": "Michigan", "GYX": "Maine", "TFX": "Montana", 
  "GRB": "Wisconsin", "GSP": "South Carolina", "GID": "Nebraska", "HGX": "Texas", 
  "HUN": "Alabama", "IND": "Indiana", "JAN": "Mississippi", "JKL": "Kentucky", 
  "JAX": "Florida", "EAX": "Missouri", "KEY": "Florida", "LCH": "Louisiana", 
  "VEF": "Nevada", "ILX": "Illinois", "LZK": "Arkansas", "LOX": "California", 
  "LMK": "Kentucky", "LUB": "Texas", "MQT": "Michigan", "MFR": "Oregon", 
  "MLB": "Florida", "MEG": "Tennessee", "MFL": "Florida", "MAF": "Texas", 
  "MKX": "Wisconsin", "MSO": "Montana", "MOB": "Alabama", "MRX": "Tennessee", 
  "PHI": "New Jersey", "OHX": "Tennessee", "LIX": "Louisiana", "MHX": "North Carolina", 
  "OKX": "New York", "OUN": "Oklahoma", "IWX": "Indiana", "LBF": "Nebraska", 
  "OAX": "Nebraska", "PAH": "Kentucky", "FFC": "Georgia", "PDT": "Oregon", 
  "PSR": "Arizona", "PBZ": "Pennsylvania", "PIH": "Idaho", "PQR": "Oregon", 
  "PUB": "Colorado", "DVN": "Iowa", "RAH": "North Carolina", "UNR": "South Dakota", 
  "REV": "Nevada", "RIW": "Wyoming", "STO": "California", "SLC": "Utah", 
  "SJT": "Texas", "SGX": "California", "MTR": "California", "HNX": "California", 
  "TJSJ": "Puerto Rico", "SEW": "Washington", "SHV": "Louisiana", "FSD": "South Dakota", 
  "OTX": "Washington", "SGF": "Missouri", "CTP": "Pennsylvania", "LSX": "Missouri", 
  "TAE": "Florida", "TBW": "Florida", "TOP": "Kansas", "TWC": "Arizona", 
  "TSA": "Oklahoma", "MPX": "Minnesota", "ICT": "Kansas", "ILM": "North Carolina", 
  "ILN": "Ohio"
}


COAST_STATE_GROUPS = {
    "West Coast": ["Washington", "Oregon", "California"],
    "East Coast": ["Maine", "New Hampshire", "Massachusetts", "Rhode Island", "Connecticut", "New York", "New Jersey", "Delaware", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    "Gulf Coast": ["Texas", "Louisiana", "Mississippi", "Alabama"]
}

OCEAN_ANCHORS = {
    'NORTH_PACIFIC': {'lat': 45.0, 'lon': 220.0, 'name': 'Pacific Northwest'}, 
    'NORTH_ATLANTIC': {'lat': 40.0, 'lon': 300.0, 'name': 'Atlantic Waters'},
    'GULF_OF_MEXICO': {'lat': 25.0, 'lon': 270.0, 'name': 'Gulf of Mexico'}
}

WEATHER_PHENOMENA = {
    'precipitation': ['rain', 'rains', 'drizzle', 'showers', 'precipitation', 'precip', 'downpours', 'sprinkles', 'sleet', 'snow', 'flurries', 'hail', 'ice pellets','flooding', 'flood', 'inundation', 'flash flood'],
    'wind': ['wind', 'gusts', 'breezy', 'gale', 'squalls'],
    'temperature': ['heat', 'warmth', 'cold', 'chill'], 
    'pressure_systems': ['ridge', 'the high', 'this high', 'upper level low', 'upper level high', 'the low', 'this low', 'ridging', 'upper high', 'upper low' 'trough', 'troughing', 'high pressure', 'low pressure', 'high-pressure', 'low-pressure', 'cyclone', 'anticyclone', 'closed low', 'blocking', 'cut-off low'], 
    'clouds_visibility': ['fog', 'mist', 'haze', 'clouds', 'cloudy', 'overcast', 'visibility'],
    'severe': ['tornado', 'watersout', 'derecho', 'blizzard']
}

PRESSURE_POLARITY = {
    "HIGH": ['ridge', 'the high', 'this high', 'upper level high', 'high pressure', 'high-pressure', 'upper high', 'anticyclone', 'blocking', 'ridging'],
    "LOW": ['trough', 'trof', 'the low', 'this low', 'upper level low' 'low pressure', 'low-pressure', 'upper low', 'cyclone', 'closed low', 'cut-off low', 'troughing']
}

PROBABILITY_MODIFIERS = ['light', 'slight', 'some', 'chance', 'chances', 'possible', 'likely', 'potential', 'isolated', 'scattered']
EXCLUDED_MODELS = ['ECMWF', 'EURO', 'HRRR', 'ECCC', 'CMC', 'GEM', 'NAM', 'UKMET', 'ICON', 'RAP', 'SREF', 'HREF']
EXCLUDED_SURFACE_TERMS = ['surface', 'sfc', 'ground', '2m', '10m', 'freezing', 'freeze', 'frost', 'dewpoint', 'dew point', 'humid']
EXCLUDED_TIME_TERMS = ['next week', 'extended range', 'long range', 'day 4', 'day 5', 'day 6', 'day 7', '8-14 day', 'outlook', 'climatology']
LOCAL_CONTEXT_KEYWORDS = ["The Region", "The Area", "The CWA", "Overhead", "Forecast Area", "The Forecast Area", "Our Area", "This Area", "The Coast", "The Coastline", 'The Northeast', 'The Southeast', 'The Northwest', 'The Southwest','The North', 'The South', 'The East', 'The West', 'our north', 'our south', 'our east', 'our west', 'eastward', 'northward', 'westward', 'southward', 'move in']
CARDINAL_DIRECTIONS = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'northern', 'southern', 'eastern', 'western', 'central']
STATES = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
STATE_ABBREVIATIONS = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WI', 'WY']
STATE_ABBREVIATION_MAP = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}
MOUNTAIN_RANGES = ['Sierra Nevada', 'Cascades', 'Appalachians', 'Blue Ridge', 'Smoky Mountains', 'Adirondacks', 'Catskills', 'Cumberland Plateau', 'Green Mountains', 'Ozarks', 'Poconos', 'Sierra Madre', 'White Mountains']
MAJOR_RIVERS = ['Mississippi River', 'Missouri River', 'Colorado River', 'Ohio River', 'Hudson River', 'Atchafalaya River', 'Chattahoochee River', 'James River', 'Red River', 'Snake River', 'St. Lawrence River']
VALLEYS_BASINS_PLAINS = ['Columbia Basin', 'Connecticut River Valley', 'Great Basin', 'Hudson Valley', 'Mississippi River Valley', 'Mohawk Valley', 'Ohio River Valley', 'Sacramento Valley', 'San Joaquin Valley', 'Shenandoah Valley', 'Tennessee Valley', 'Northern Plains', 'Southern Plains', 'Snake River Plain']
GREAT_LAKES = ['Great Lakes', 'Lake Superior', 'Lake Michigan', 'Lake Huron', 'Lake Erie', 'Lake Ontario', 'Lake St. Clair', 'Lake Okeechobee']
OCEANS_BAYS = ['Atlantic', 'Pacific', 'Gulf of Mexico', 'Gulf', 'Chesapeake Bay', 'Buzzards Bay', 'Cape Cod Bay', 'Long Island Sound', 'Monterey Bay', 'Narragansett Bay', 'Puget Sound', 'Saginaw Bay', 'Strait of Juan de Fuca', 'Gulf Waters', 'Atlantic Waters', 'Gulf of Maine']
CAPES_ISLANDS_PENINSULAS = ['Block Island', 'Cape Cod', 'Cape Mendocino', 'Delmarva Peninsula', 'Florida Panhandle', 'Long Island', 'Marthas Vineyard', 'Nantucket', 'Olympic Peninsula', 'Outer Banks', 'Upper Peninsula', 'Vancouver Island', 'Yucatan Peninsula']
CANADIAN_PROVINCES = ['Ontario', 'Yukon', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba', 'Saskatchewan', 'Labrador', 'Nova Scotia']
NEIGHBORING_REGIONS = ['Baja California', 'Bahamas', 'Canada', 'Cuba', 'Mexico']
MISC_REGIONS = ['Tri-State','Twin Cities', 'Tri-Cities', 'West Coast', 'East Coast', 'New England', 'Mid-Atlantic', 'Big Bend', 'Concho Valley', 'Mississippi Valley', 'Ohio Valley', 'Carolinas', 'Downeast', 'Edwards Plateau', 'Finger Lakes', 'Four Corners', 'Hill Country', 'Intermountain West', 'Mid-South', 'Midwest', 'Missouri Bootheel', 'Mohave', 'Pacific Northwest', 'The Panhandle', 'The Thumb', 'Trans Pecos', 'CWA', 'Eastern Seaboard', 'Plains', 'Eastern CONUS', 'Western CONUS', 'Northern CONUS', 'Southern CONUS', 'Eastern U', 'Western U', 'Northern U', 'Southern U']

PRESSURE_POLARITY = {
    "HIGH": ["high", "ridge", "anticyclone", "high pressure", "ridging", "advancing high"],
    "LOW": ["low", "trough", "cyclone", "low pressure", "troughing", "shortwave", "advancing low"]
}

LOCAL_CONTEXT_KEYWORDS = {"The Region", "The Area", "The CWA", "Overhead", "Forecast Area", "The Forecast Area", "Our Area", "This Area", "The Coast", "The Coastline", 'The Northeast', 'The Southeast', 'The Northwest', 'The Southwest','The North', 'The South', 'The East', 'The West', 'our north', 'our south', 'our east', 'our west', 'eastward', 'northward', 'westward', 'southward', 'move in'}


def load_hierarchy(path="location_hierarchy.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CRITICAL: Could not find {path}. Please run your hierarchy generation script first.")
    with open(path, "r") as f:
        return json.load(f)

HIERARCHY_MAP = {}
LOC_PATTERN = None

def get_pressure_polarity(term):
    t = term.lower()
    for label, keywords in PRESSURE_POLARITY.items():
        if t in keywords: return label
    return "NEUTRAL"

def split_into_sentences(text):
    if not text: return []
    text = text.replace('\n', ' ').replace('...', '.')
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if s.strip()]

def analyze_weather_text(text):
    """
    Extracts (Phenomenon, Location) pairs.
    FILTERING: Discards any object where the location is 'Local/Here/CWA'.
    """
    if not text: return []
    
    extracted_data = []
    
    all_keywords = []
    for pol, kws in PRESSURE_POLARITY.items():
        for kw in kws:
            all_keywords.append((kw, pol))
    all_keywords.sort(key=lambda x: len(x[0]), reverse=True)
    
    sentences = split_into_sentences(text)
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        found_phenomena = []
        for kw, polarity in all_keywords:
            for match in re.finditer(r'\b' + re.escape(kw) + r'\b', sentence_lower):
                found_phenomena.append({
                    "kw": kw, 
                    "pol": polarity, 
                    "start": match.start()
                })
        
        if not found_phenomena: continue

        found_locations = []
        if LOC_PATTERN:
            for match in LOC_PATTERN.finditer(sentence):
                found_locations.append({
                    "raw": match.group(0),
                    "start": match.start()
                })

        for phenom in found_phenomena:
            best_loc_raw = None 
            min_dist = float('inf')
            
            if found_locations:
                for loc in found_locations:
                    dist = abs(phenom['start'] - loc['start'])
                    if dist < min_dist:
                        min_dist = dist
                        best_loc_raw = loc['raw']

            if best_loc_raw is None:
                continue

            if best_loc_raw.lower() in LOCAL_CONTEXT_KEYWORDS:
                continue
            if best_loc_raw == "LOCAL_CONTEXT":
                continue
            
            extracted_data.append({
                'phenomenon': phenom['kw'],
                'category': 'pressure_systems',
                'location': best_loc_raw, 
                'sentence': sentence
            })
            
    return extracted_data


def normalize_location_name(loc, hierarchy_map):
    """Fixes case sensitivity for 2-letter states and matches JSON keys."""
    loc = loc.strip()
    if len(loc) == 2:
        loc = loc.upper()
        
    if loc not in hierarchy_map:
        for k in hierarchy_map:
            if k.lower() == loc.lower():
                return k
    return loc

def calculate_graph_distance(loc1, loc2, hierarchy_map, max_hops=3, stop_nodes=None):
    """Calculates multidirectional hops while avoiding massive hub nodes."""
    if stop_nodes is None:
        stop_nodes = {"Canada", "CONUS", "Eastern Canada", "Central Canada", "Western Canada", "Eastern CONUS", "Western CONUS", "Central CONUS", "Central Plains", "Ohio Valley", "Great Lakes", "Central U.S.", "Eastern U.S.", "Western U.S.", "Central United States", "Eastern United States", "Western United States", "Central U", "Eastern U", "Western U", "Eastern US", "Western US", "Central US", "Midwest", "The Plains"}

    if loc1 == loc2: return 0
    if loc1 not in hierarchy_map and loc2 not in hierarchy_map: 
        return float('inf')

    queue = deque([(loc1, 0)])
    visited = {loc1}

    while queue:
        current_node, current_dist = queue.popleft()

        if current_node == loc2:
            return current_dist
        if current_dist >= max_hops:
            continue
        if current_node in stop_nodes and current_node != loc1:
            continue

        neighbors = set()
        neighbors.update(hierarchy_map.get(current_node, {}).get('parents', []))
        neighbors.update(hierarchy_map.get(current_node, {}).get('siblings', []))
        for potential_child, data in hierarchy_map.items():
            if current_node in data.get('parents', []):
                neighbors.add(potential_child)

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_dist + 1))

    return float('inf')

def calculate_space_aggregate_score(pred_objs_raw, ref_objs_raw):
    def clean_objects(raw_objs):
        cleaned = []
        for o in raw_objs:
            pol = get_pressure_polarity(o['phenomenon'])
            if pol not in ['HIGH', 'LOW']: continue
            if not o['location']: continue
            
            c_loc = normalize_location_name(o['location'], HIERARCHY_MAP)
            cleaned.append({'polarity': pol, 'clean_loc': c_loc, 'raw': o})
        return cleaned

    pred_objs = clean_objects(pred_objs_raw)
    ref_objs = clean_objects(ref_objs_raw)

    if not pred_objs and not ref_objs:
        return None  

    all_unique_locs = list(set([p['clean_loc'] for p in pred_objs] + [r['clean_loc'] for r in ref_objs]))
    
    DISTANCE_THRESHOLD = 2 
    parent_map = {loc: loc for loc in all_unique_locs}

    def find(loc):
        if parent_map[loc] != loc:
            parent_map[loc] = find(parent_map[loc])
        return parent_map[loc]

    def union(loc1, loc2):
        root1 = find(loc1)
        root2 = find(loc2)
        if root1 != root2:
            parent_map[root2] = root1

    STOP_LOCATIONS = {"Canada", "CONUS", "Eastern Canada", "Central Canada", "Western Canada", "Eastern CONUS", "Western CONUS", "Central CONUS", "Central Plains", "Ohio Valley", "Great Lakes", "Central U.S.", "Eastern U.S.", "Western U.S.", "Central United States", "Eastern United States", "Western United States", "Central U", "Eastern U", "Western U", "Eastern US", "Western US", "Central US", "Midwest", "The Plains"}
    for i in range(len(all_unique_locs)):
        for j in range(i + 1, len(all_unique_locs)):
            loc1 = all_unique_locs[i]
            loc2 = all_unique_locs[j]
            
            dist = calculate_graph_distance(loc1, loc2, HIERARCHY_MAP)
            
            current_threshold = DISTANCE_THRESHOLD 
            if loc1 in STOP_LOCATIONS or loc2 in STOP_LOCATIONS:
                current_threshold = 1
                
            if dist <= current_threshold:
                union(loc1, loc2)
                continue
                
            loc1_siblings = HIERARCHY_MAP.get(loc1, {}).get('siblings', [])
            loc2_siblings = HIERARCHY_MAP.get(loc2, {}).get('siblings', [])
            
            if loc2 in loc1_siblings or loc1 in loc2_siblings:
                union(loc1, loc2)

    loc_clusters = defaultdict(lambda: {'preds': [], 'refs': []})
    for p in pred_objs:
        loc_clusters[find(p['clean_loc'])]['preds'].append(p)
    for r in ref_objs:
        loc_clusters[find(r['clean_loc'])]['refs'].append(r)

    def get_low_ratio(obj_list):
        if not obj_list: return 0.5 
        low_count = sum(1 for o in obj_list if o['polarity'] == 'LOW')
        return low_count / len(obj_list)

    cluster_scores = []
    matched_objects_count = 0

    for root_key, data in loc_clusters.items():
        local_preds = data['preds']
        local_refs = data['refs']
        
        if len(local_preds) > 0 and len(local_refs) > 0:
            matched_objects_count += len(local_preds) + len(local_refs)
            p_ratio = get_low_ratio(local_preds)
            r_ratio = get_low_ratio(local_refs)
            local_score = 1.0 - abs(p_ratio - r_ratio)
            
        else:
            local_score = 0.5

        cluster_scores.append(local_score)

    match_score = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0

    total_objects = len(pred_objs) + len(ref_objs)
    coverage_ratio = (matched_objects_count / total_objects) if total_objects > 0 else 1.0

    final_space = match_score * coverage_ratio

    return final_space, match_score, coverage_ratio


def calculate_stats(values):
    """Returns (mean, stdev) for a list of numbers."""
    if not values: return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) > 1:
        stdev_val = statistics.stdev(values)
    else:
        stdev_val = 0.0
    return mean_val, stdev_val



def get_event_key(sample_id):
    parts = str(sample_id).split('_')
    if len(parts) >= 3:
        return "_".join(parts[1:4]) 
    return str(sample_id)

def main(input_file, output_file):
    global HIERARCHY_MAP, LOC_PATTERN
    
    print("Loading Hierarchy...")
    HIERARCHY_MAP = load_hierarchy()
    
    all_locs = sorted(HIERARCHY_MAP.keys(), key=len, reverse=True)
    pattern_str = r'\b(' + '|'.join(map(re.escape, all_locs)) + r')\b'
    print("Compiling Regex Pattern...")
    LOC_PATTERN = re.compile(pattern_str, re.IGNORECASE)

    print(f"Loading Data: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print("Pooling Objects by Event...")
    events_registry = defaultdict(lambda: {'pred': [], 'ref': []})
    
    for sample in tqdm(data, desc="Extracting"):
        sample_id = sample.get('id', 'UNKNOWN')
        event_key = get_event_key(sample_id)
        
        events_registry[event_key]['pred'].extend(analyze_weather_text(sample.get('prediction', '')))
        events_registry[event_key]['ref'].extend(analyze_weather_text(sample.get('reference', '')))

    print(f"Scoring {len(events_registry)} Events...")
    results = []
    ignored_events = 0
    
    for event_key, pools in tqdm(events_registry.items(), desc="Scoring"):
        metrics = calculate_space_aggregate_score(pools['pred'], pools['ref'])
        
        if metrics is None:
            ignored_events += 1
            continue
            
        space, match, cov = metrics
        
        results.append({
            "event_id": event_key,
            "score": space,
            "accuracy": match,
            "coverage": cov,
            "n_objects": len(pools['pred']) + len(pools['ref'])
        })

    space_scores = [r['score'] for r in results]
    coverage_scores = [r['coverage'] for r in results]
    match_scores = [r['accuracy'] for r in results if r['accuracy'] is not None]

    mean_space, std_space = calculate_stats(space_scores)
    mean_cov, std_cov = calculate_stats(coverage_scores)
    mean_match, std_match = calculate_stats(match_scores)

    print(f"FINAL AGGREGATE RESULTS ({len(results)} valid events)")
    print(f"SPACE Score:    {mean_space:.4f} ± {std_space:.4f}")
    print(f"Coverage Ratio: {mean_cov:.4f}   ± {std_cov:.4f}")
    print(f"Match Accuracy: {mean_match:.4f} ± {std_match:.4f}")
    print(f"Ignored Events: {ignored_events}")

    output_data = {
        "summary": {
            "mean_space": mean_space, "std_space": std_space,
            "mean_coverage": mean_cov, "std_coverage": std_cov,
            "mean_match": mean_match, "std_match": std_match,
            "num_events": len(results),
            "ignored_events": ignored_events
        },
        "details": results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Detailed results and stats saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Input JSON (predictions)")
    parser.add_argument("--output_json", required=True, help="Output JSON (scores)")
    args = parser.parse_args()
    
    main(args.input_json, args.output_json)