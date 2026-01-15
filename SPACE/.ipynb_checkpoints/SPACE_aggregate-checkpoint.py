import re
import json
import math
import os
import time
import argparse 
from collections import defaultdict
from tqdm import tqdm
import statistics


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
  "ABR": "Aberdeen, South Dakota", "ALY": "Albany, New York", "ABQ": "Albequerque, New Mexico", "AMA": "Amarillo, Texas", 
  "APX": "Gaylord, Michigan", "ARX": "La Crosse, Wisconsin", "AKQ": "Wakefield, Virginia", "EWX": "Austin/San Antonio, Texas", 
  "LWX": "Baltimore, Maryland", "BYZ": "Billings, Montana", "BGM": "Binghampton, New York", "BMX": "Birmingham, Alabama", 
  "BIS": "Bismarck, North Dakota", "RNK": "Blacksburg, Virginia", "BOI": "Boise, Idaho", "BOX": "Boston, Massachusetts", 
  "BRO": "Brownsville, Texas", "BUF": "Buffalo, New York", "BTV": "Burlington, Vermont", "CAR": "Caribou, Maine", 
  "CHS": "Charleston, South Carolina", "RLX": "Charleston, West Virginia", "CYS": "Cheyenne, Wyoming", "LOT": "Chicago, Illinois", 
  "CLE": "Cleveland, Ohio", "CAE": "Columbia, South Carolina", "CRP": "Corpus Christi, Texas", "FWD": "Dallas, Texas", 
  "BOU": "Denver, Colorado", "DMX": "Des Moines, Iowa", "DTX": "Detroit, Michigan", "DDC": "Dodge City, Kansas", 
  "DLH": "Duluth, Minnesota", "LKN": "Elko, Nevada", "EPZ": "El Paso, Texas", "EKA": "Eureka, California", 
  "FGZ": "Flagstaff, Arizona", "GGW": "Glasgow, Montana", "GLD": "Goodland, Kansas", "FGF": "Grand Forks, North Dakota", 
  "GJT": "Grand Junction, Colorado", "GRR": "Grand Rapids, Michigan", "GYX": "Gray, Maine", "TFX": "Great Falls, Montana", 
  "GRB": "Green Bay, Wisconsin", "GSP": "Greenville, South Carolina", "GID": "Hastings, Nebraska", "HGX": "Houston/Galveston, Texas", 
  "HUN": "Huntsville, Alabama", "IND": "Indianapolis, Indiana", "JAN": "Jackson, Mississippi", "JKL": "Jackson, Kentucky", 
  "JAX": "Jacksonville, Florida", "EAX": "Kansas City, Missouri", "KEY": "Key West, Florida", "LCH": "Lake Charles, Louisiana", 
  "VEF": "Las Vegas, Nevada", "ILX": "Lincoln, Illinois", "LZK": "Little Rock, Arkansas", "LOX": "Los Angeles, California", 
  "LMK": "Louisville, Kentucky", "LUB": "Lubbock, Texas", "MQT": "Marquette, Michigan", "MFR": "Medford, Oregon", 
  "MLB": "Melbourne, Florida", "MEG": "Memphis, Tennessee", "MFL": "Miami, Florida", "MAF": "Midland, Texas", 
  "MKX": "Milwaukee, Wisconsin", "MSO": "Missoula, Montana", "MOB": "Mobile, Alabama", "MRX": "Morristown, Tennessee", 
  "PHI": "Mount Holly, New Jersey", "OHX": "Nashville, Tennessee", "LIX": "New Orleans, Louisiana", "MHX": "Newport, North Carolina", 
  "OKX": "New York City, New York", "OUN": "Norman, Oklahoma", "IWX": "Northern Indiana", "LBF": "North Platte, Nebraska", 
  "OAX": "Omaha, Nebraska", "PAH": "Paducah, Kentucky", "FFC": "Peachtree City, Georgia", "PDT": "Pendleton, Oregon", 
  "PSR": "Phoenix, Arizona", "PBZ": "Pittsburgh, Pennsylvania", "PIH": "Pocatello, Idaho", "PQR": "Portland, Oregon", 
  "PUB": "Pueblo, Colorado", "DVN": "Quad Cities, Iowa/Illinois", "RAH": "Raleigh, North Carolina", "UNR": "Rapid City, South Dakota", 
  "REV": "Reno, Nevada", "RIW": "Riverton, Wyoming", "STO": "Sacramento, California", "SLC": "Salt Lake City, Utah", 
  "SJT": "San Angelo, Texas", "SGX": "San Diego, California", "MTR": "San Francisco, California", "HNX": "San Joaquin Valley, California", 
  "TJSJ": "San Juan, Puerto Rico", "SEW": "Seattle, Washington", "SHV": "Shreveport, Louisiana", "FSD": "Sioux Falls, South Dakota", 
  "OTX": "Spokane, Washington", "SGF": "Springfield, Missouri", "CTP": "State College, Pennsylvania", "LSX": "St. Louis, Missouri", 
  "TAE": "Tallahassee, Florida", "TBW": "Tampa, Florida", "TOP": "Topeka, Kansas", "TWC": "Tucson, Arizona", 
  "TSA": "Tulsa, Oklahoma", "MPX": "Twin Cities, Minnesota", "ICT": "Wichita, Kansas", "ILM": "Wilmington, North Carolina", 
  "ILN": "Wilmington, Ohio"
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

# --- GEOGRAPHY LISTS ---
WEATHER_PHENOMENA = {
    'precipitation': ['rain', 'rains', 'drizzle', 'showers', 'precipitation', 'precip', 'downpours', 'sprinkles', 'sleet', 'snow', 'flurries', 'hail', 'ice pellets','flooding', 'flood', 'inundation', 'flash flood'],
    'wind': ['wind', 'gusts', 'breezy', 'gale', 'squalls'],
    'temperature': ['heat', 'warmth', 'cold', 'chill'], 
    'pressure_systems': ['ridge', 'ridging', 'trough', 'upper high', 'upper low', 'troughing', 'high pressure', 'low pressure', 'high-pressure', 'low-pressure', 'cyclone', 'anticyclone', 'closed low', 'blocking', 'cut-off low', 'advancing low', 'advancing high'], 
    'clouds_visibility': ['fog', 'mist', 'haze', 'clouds', 'cloudy', 'overcast', 'visibility'],
    'severe': ['tornado', 'watersout', 'derecho', 'blizzard']
}

PRESSURE_POLARITY = {
    "HIGH": ['ridge', 'high pressure', 'high-pressure', 'anticyclone', 'upper high', 'blocking', 'ridging'],
    "LOW": ['trough', 'low pressure', 'low-pressure', 'upper low', 'cyclone', 'closed low', 'cut-off low', 'troughing']
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

# Synonyms for "The Local Area" - These will be IGNORED in this aggregate metric
LOCAL_CONTEXT_KEYWORDS = {"The Region", "The Area", "The CWA", "Overhead", "Forecast Area", "The Forecast Area", "Our Area", "This Area", "The Coast", "The Coastline", 'The Northeast', 'The Southeast', 'The Northwest', 'The Southwest','The North', 'The South', 'The East', 'The West', 'our north', 'our south', 'our east', 'our west', 'eastward', 'northward', 'westward', 'southward', 'move in'}

# ==========================================
# 2. HIERARCHY LOADING & HELPERS
# ==========================================

def load_hierarchy(path="location_hierarchy.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CRITICAL: Could not find {path}. Please run your hierarchy generation script first.")
    with open(path, "r") as f:
        return json.load(f)

# Global variables (loaded in main)
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
    
    # Pre-calculate keywords
    all_keywords = []
    for pol, kws in PRESSURE_POLARITY.items():
        for kw in kws:
            all_keywords.append((kw, pol))
    all_keywords.sort(key=lambda x: len(x[0]), reverse=True)
    
    sentences = split_into_sentences(text)
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # A. Find all weather phenomena
        found_phenomena = []
        for kw, polarity in all_keywords:
            for match in re.finditer(r'\b' + re.escape(kw) + r'\b', sentence_lower):
                found_phenomena.append({
                    "kw": kw, 
                    "pol": polarity, 
                    "start": match.start()
                })
        
        if not found_phenomena: continue

        # B. Find all known Hierarchy locations
        found_locations = []
        if LOC_PATTERN:
            for match in LOC_PATTERN.finditer(sentence):
                found_locations.append({
                    "raw": match.group(0),
                    "start": match.start()
                })

        # C. Match Phenomenon to Closest Location
        for phenom in found_phenomena:
            best_loc_raw = None 
            min_dist = float('inf')
            
            if found_locations:
                for loc in found_locations:
                    dist = abs(phenom['start'] - loc['start'])
                    if dist < min_dist:
                        min_dist = dist
                        best_loc_raw = loc['raw']
            
            # --- FILTERING LOGIC ---
            # 1. If no location found, it implies "local" -> SKIP
            if best_loc_raw is None:
                continue

            # 2. If location is explicitly a local synonym -> SKIP
            # Check against set
            if best_loc_raw.lower() in LOCAL_CONTEXT_KEYWORDS:
                continue
            # Check against hierarchy alias (if your hierarchy maps CWA -> LOCAL_CONTEXT)
            if best_loc_raw == "LOCAL_CONTEXT":
                continue
            
            # If we survived, it's a Synoptic Location (e.g. "Ohio Valley")
            extracted_data.append({
                'phenomenon': phenom['kw'],
                'category': 'pressure_systems',
                'location': best_loc_raw, 
                'sentence': sentence
            })
            
    return extracted_data
# ==========================================
# 4. SCORING LOGIC (AGGREGATE)
# ==========================================

def calculate_space_aggregate_score(pred_objs, ref_objs):
    # 1. FILTER: Only keep High/Low systems
    pred_objs = [o for o in pred_objs if get_pressure_polarity(o['phenomenon']) in ['HIGH', 'LOW']]
    ref_objs = [o for o in ref_objs if get_pressure_polarity(o['phenomenon']) in ['HIGH', 'LOW']]
    
    # 2. HELPER: Group by Parent Location -> RETURNS A SET
    # This enforces "One vote per location-polarity pair" (Fixes spamming)
    def get_unique_systems_and_counts(obj_list):
        unique_systems = set()
        location_counts = defaultdict(lambda: {'HIGH': 0, 'LOW': 0})
        
        for obj in obj_list:
            polarity = get_pressure_polarity(obj['phenomenon'])
            loc = obj['location']
            
            # Resolve to Parent
            parent_loc = loc
            if loc in HIERARCHY_MAP:
                parents = HIERARCHY_MAP[loc].get('parents', [])
                if parents: parent_loc = parents[0]
            else:
                # Case-insensitive check
                for k in HIERARCHY_MAP:
                    if k.lower() == loc.lower():
                        parents = HIERARCHY_MAP[k].get('parents', [])
                        if parents: parent_loc = parents[0]
                        break
            
            # Add to Set (implicitly handles duplicates/spam)
            unique_systems.add((parent_loc, polarity))
            # Keep counts for the secondary match_score metric
            location_counts[parent_loc][polarity] += 1
            
        return unique_systems, location_counts

    pred_set, pred_counts = get_unique_systems_and_counts(pred_objs)
    ref_set, ref_counts = get_unique_systems_and_counts(ref_objs)

    # 3. CALCULATE SYMMETRIC COVERAGE (Jaccard Index on Sets)
    intersection = pred_set.intersection(ref_set)
    union = pred_set.union(ref_set)
    
    if len(union) == 0:
        # Both empty -> Perfect Match
        coverage_ratio = 1.0 
    else:
        coverage_ratio = len(intersection) / len(union)

    # 4. CALCULATE MATCH SCORE (Spatial/Polarity Accuracy)
    # Calculated only on locations where BOTH have activity (Intersection)
    # This preserves symmetry because Intersection(A,B) == Intersection(B,A)
    
    location_accuracies = []
    
    # Identify locations present in both (using the keys from the counts)
    common_locations = set(pred_counts.keys()) & set(ref_counts.keys())
    
    for loc in common_locations:
        p_data = pred_counts[loc]
        r_data = ref_counts[loc]
        
        n_p = p_data['HIGH'] + p_data['LOW']
        n_r = r_data['HIGH'] + r_data['LOW']
        
        # Safety check (though keys appearing in both usually implies count > 0)
        if n_p > 0 and n_r > 0:
            ratio_pred = p_data['LOW'] / n_p
            ratio_ref = r_data['LOW'] / n_r
            
            acc = 1.0 - abs(ratio_pred - ratio_ref)
            location_accuracies.append(acc)

    if not location_accuracies:
        # If no common locations, accuracy is undefined. 
        # If coverage is 0.0, this value doesn't matter much (final score will be 0).
        # If coverage is 1.0 (both empty), we want 1.0.
        match_score = 1.0 if len(union) == 0 else 0.0
    else:
        match_score = sum(location_accuracies) / len(location_accuracies)

    # 5. FINAL SPACE SCORE
    final_space = match_score * coverage_ratio
    
    return final_space, match_score, coverage_ratio

# ==========================================
# 5. STATISTICS HELPER
# ==========================================
def calculate_stats(values):
    """Returns (mean, stdev) for a list of numbers."""
    if not values: return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) > 1:
        stdev_val = statistics.stdev(values)
    else:
        stdev_val = 0.0
    return mean_val, stdev_val

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def get_event_key(sample_id):
    parts = str(sample_id).split('_')
    if len(parts) >= 3:
        return "_".join(parts[1:4]) 
    return str(sample_id)

def main(input_file, output_file):
    global HIERARCHY_MAP, LOC_PATTERN
    
    print("🌍 Loading Hierarchy...")
    HIERARCHY_MAP = load_hierarchy()
    
    # Compile Regex
    all_locs = sorted(HIERARCHY_MAP.keys(), key=len, reverse=True)
    pattern_str = r'\b(' + '|'.join(map(re.escape, all_locs)) + r')\b'
    print("⚙️  Compiling Regex Pattern...")
    LOC_PATTERN = re.compile(pattern_str, re.IGNORECASE)

    print(f"📂 Loading Data: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Pooling
    print("🏊 Pooling Objects by Event...")
    events_registry = defaultdict(lambda: {'pred': [], 'ref': []})
    
    for sample in tqdm(data, desc="Extracting"):
        sample_id = sample.get('id', 'UNKNOWN')
        event_key = get_event_key(sample_id)
        
        # Note: We removed meta_location arg because we filter local out entirely now
        events_registry[event_key]['pred'].extend(analyze_weather_text(sample.get('prediction', '')))
        events_registry[event_key]['ref'].extend(analyze_weather_text(sample.get('reference', '')))

    # Scoring
    print(f"📊 Scoring {len(events_registry)} Events...")
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

    # Statistics Calculation
    space_scores = [r['score'] for r in results]
    coverage_scores = [r['coverage'] for r in results]
    # Filter out None values for accuracy before calc
    match_scores = [r['accuracy'] for r in results if r['accuracy'] is not None]

    mean_space, std_space = calculate_stats(space_scores)
    mean_cov, std_cov = calculate_stats(coverage_scores)
    mean_match, std_match = calculate_stats(match_scores)

    print("\n" + "="*50)
    print(f"✅ FINAL AGGREGATE RESULTS ({len(results)} valid events)")
    print("="*50)
    print(f"SPACE Score:    {mean_space:.4f} ± {std_space:.4f}")
    print(f"Coverage Ratio: {mean_cov:.4f}   ± {std_cov:.4f}")
    print(f"Match Accuracy: {mean_match:.4f} ± {std_match:.4f}")
    print(f"Ignored Events: {ignored_events}")
    print("="*50)

    # Save Results
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
    print(f"💾 Detailed results and stats saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Input JSON (predictions)")
    parser.add_argument("--output_json", required=True, help="Output JSON (scores)")
    args = parser.parse_args()
    
    main(args.input_json, args.output_json)