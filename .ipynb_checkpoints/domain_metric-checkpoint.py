import re
import json
import math
import os
import time
import statistics
import argparse 
from tqdm import tqdm
from collections import defaultdict
import math
import nltk
from collections import deque

# ==========================================
# PART 1: CONFIGURATION & LISTS
# ==========================================

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

# --- GEOGRAPHY LISTS ---
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
    "LOW": ['trough', 'trof', 'the low', 'this low', 'upper level low', 'low pressure', 'low-pressure', 'upper low', 'cyclone', 'closed low', 'cut-off low', 'troughing']
}

PROBABILITY_MODIFIERS = ['light', 'slight', 'some', 'chance', 'chances', 'possible', 'likely', 'potential', 'isolated', 'scattered']
EXCLUDED_MODELS = ['ECMWF', 'EURO', 'HRRR', 'ECCC', 'CMC', 'GEM', 'NAM', 'UKMET', 'ICON', 'RAP', 'SREF', 'HREF']
EXCLUDED_SURFACE_TERMS = ['surface', 'prefrontal', 'low level', 'sfc', 'ground', '2m', '10m', 'freezing', 'freeze', 'frost', 'dewpoint', 'dew point', 'humid']
EXCLUDED_TIME_TERMS = ['next week', 'extended range', 'long range', 'day 4', 'day 5', 'day 6', 'day 7', '8-14 day', 'outlook', 'climatology']
LOCAL_CONTEXT_TERMS = ["The Region", "Here", "pushes in", "prevail", "moves through", "Offshore", "The Area", "riding along", "provides", "will provide", "may bring", "will bring", "Moves through", "move through", "The state", "The CWA", "Overhead", "Forecast Area", "The Forecast Area", "Our Area", "This Area", "The Coast", "The Coastline", 'The Northeast', 'The Southeast', 'The Northwest', 'The Southwest','The North', 'The South', 'The East', 'The West', 'The Highland', 'eastward', 'southward', 'northward', 'westward']
CARDINAL_DIRECTIONS = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'northern', 'southern', 'eastern', 'western', 'central']
STATES = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
STATE_ABBREVIATIONS = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WI', 'WY']
STATE_ABBREVIATION_MAP = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}
MOUNTAIN_RANGES = ['Sierra Nevada', 'Cascades', 'Appalachians', 'Blue Ridge', 'Smoky Mountains', 'Adirondacks', 'Catskills', 'Cumberland Plateau', 'Green Mountains', 'Ozarks', 'Poconos', 'Sierra Madre', 'White Mountains']
MAJOR_RIVERS = ['Mississippi River', 'Missouri River', 'Colorado River', 'Ohio River', 'Hudson River', 'Atchafalaya River', 'Chattahoochee River', 'James River', 'Red River', 'Snake River', 'St. Lawrence River']
VALLEYS_BASINS_PLAINS = ['Columbia Basin', 'Connecticut River Valley', 'Great Basin', 'Hudson Valley', 'Mississippi River Valley', 'Mohawk Valley', 'Ohio River Valley', 'Sacramento Valley', 'San Joaquin Valley', 'Shenandoah Valley', 'Tennessee Valley', 'Northern Plains', 'Southern Plains', 'Snake River Plain']
GREAT_LAKES = ['Great Lakes', 'Lake Superior', 'Lake Michigan', 'Lake Huron', 'Lake Erie', 'Lake Ontario', 'Lake St. Clair', 'Lake Okeechobee']
OCEANS_BAYS = ['Atlantic', 'Pacific', 'Gulf of Mexico', 'Gulf', 'Chesapeake Bay', 'Buzzards Bay', 'Cape Cod Bay', 'Long Island Sound', 'Monterey Bay', 'Narragansett Bay', 'Puget Sound', 'Saginaw Bay', 'Strait of Juan de Fuca', 'Gulf Waters', 'Atlantic Waters', 'Gulf of Maine']
CAPES_ISLANDS_PENINSULAS = ['Block Island', 'Cape Cod', 'Cape Mendocino', 'Delmarva Peninsula', 'Florida Panhandle', 'Long Island', 'Marthas Vineyard', 'Nantucket', 'Olympic Peninsula', 'Outer Banks', 'Upper Peninsula', 'Vancouver Island', 'Yucatan Peninsula']
CANADIAN_PROVINCES = ['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba', 'Saskatchewan', 'Labrador', 'Nova Scotia']
NEIGHBORING_REGIONS = ['Baja California', 'Bahamas', 'Canada', 'Cuba', 'Mexico']
MISC_REGIONS = ['Tri-State','Twin Cities', 'Tri-Cities', 'West Coast', 'East Coast', 'New England', 'Mid-Atlantic', 'Big Bend', 'Concho Valley', 'Mississippi Valley', 'Ohio Valley', 'Carolinas', 'Downeast', 'Edwards Plateau', 'Finger Lakes', 'Four Corners', 'Hill Country', 'Intermountain West', 'Mid-South', 'Midwest', 'Missouri Bootheel', 'Mohave', 'Pacific Northwest', 'The Panhandle', 'The Thumb', 'Trans Pecos', 'CWA', 'Eastern Seaboard', 'Plains', 'Eastern CONUS', 'Western CONUS', 'Northern CONUS', 'Southern CONUS', 'Eastern U', 'Western U', 'Northern U', 'Southern U']
STATE_ABBREVIATIONS = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IND', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OREGON', 'OK', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
STATE_ABBREVIATION_MAP = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 
    'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 
    'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 
    'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 
    'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

# 2. CREATE A "SAFE" TEXT SEARCH LIST
# We omit abbreviations that are common English words, prepositions, or units.
DANGEROUS_ABBREVS = {'IN', 'OR', 'ME', 'HI', 'MA', 'PA', 'LA', 'AL', 'OK', 'ID', 'AR', 'MI'}
SAFE_STATE_ABBREVIATIONS = [abbr for abbr in STATE_ABBREVIATION_MAP.keys() if abbr not in DANGEROUS_ABBREVS]

NWS_ABBREVIATIONS = {
    r'\bNRN\b': 'Northern', r'\bSRN\b': 'Southern', r'\bERN\b': 'Eastern', r'\bWRN\b': 'Western',
    r'\bCNTRL\b': 'Central', r'\bNE\b': 'Northeast', r'\bNW\b': 'Northwest', r'\bSE\b': 'Southeast', r'\bSW\b': 'Southwest',
    r'\bFM\b': 'from', r'\bAFTN\b': 'afternoon', r'\bMRNG\b': 'morning', r'\bEVNG\b': 'evening',
    r'\bTSTM\b': 'thunderstorm', r'\bFNT\b': 'front', r'\bWND\b': 'wind', r'\bSVR\b': 'severe'
}

HIERARCHY_MAP = {}
if os.path.exists("location_hierarchy.json"):
    try:
        with open("location_hierarchy.json", "r") as f:
            HIERARCHY_MAP = json.load(f)
    except Exception as e:
        print(f"Error loading hierarchy: {e}")


def get_pressure_polarity(term):
    t = term.lower()
    for label, keywords in PRESSURE_POLARITY.items():
        if t in keywords: return label
    return "NEUTRAL"

def split_into_sentences(text):
    text = text.replace('\n', ' ').replace('...', '.')
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_time(sentence):
    time_patterns = [
        r'\b(today|tonight|tomorrow|weekend|overnight|sun|mon|tues|wed|thurs|fri|sat|00Z|02Z|04Z|06Z|08Z|10Z|12Z|14Z|16Z|18Z|20Z|22Z)\b',
        r'\b((mon|tues|wednes|thurs|fri|satur|sun)day)\b',
    ]
    found_times = []
    for pattern in time_patterns:
        matches = re.findall(pattern, sentence, re.IGNORECASE)
        for match in matches:
            found_times.append(match[0].strip() if isinstance(match, tuple) else match.strip())
    return list(set(found_times))

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def resolve_coords_to_place(lat, lon):
    lon_360 = lon % 360
    min_dist = float('inf')
    best_name = None
    
    for code, coords in NWS_STATIONS_COORDS.items():
        dist = haversine_distance(lat, lon_360, coords['lat'], coords['lon'])
        if dist < min_dist:
            min_dist = dist
            best_name = STATION_NAME_MAP.get(code, code)
            
    for key, data in OCEAN_ANCHORS.items():
        dist = haversine_distance(lat, lon_360, data['lat'], data['lon'])
        if dist < min_dist:
            min_dist = dist
            best_name = data['name']
            
    return best_name

def extract_coordinates_with_indices(sentence):
    coord_pattern = r'\b(\d{1,2}(?:\.\d+)?)\s*([NS])\W*(\d{1,3}(?:\.\d+)?)\s*([EW])\b'
    found = []
    for m in re.finditer(coord_pattern, sentence, re.IGNORECASE):
        lat_val, lat_dir, lon_val, lon_dir = m.groups()
        lat = float(lat_val)
        if lat_dir.upper() == 'S': lat = -lat
        lon = float(lon_val)
        if lon_dir.upper() == 'W': lon = -lon
        
        resolved_name = resolve_coords_to_place(lat, lon)
        if resolved_name:
            found.append({
                'name': resolved_name,
                'start': m.start(),
                'end': m.end(),
                'center': (m.start() + m.end()) / 2
            })
    return found

def normalize_location_name(loc, hierarchy_map):
    """
    Strips common prefixes like 'Upper', 'Central', 'Northern' 
    to see if the base location exists in the hierarchy.
    Ex: "Upper Ohio Valley" -> "Ohio Valley"
    """
    loc = loc.strip()
    # Add this 2-letter fix at the very top!
    if len(loc) == 2:
        loc = loc.upper()
    if not loc: return None
    
    # If the exact location is already known, return it
    if loc in hierarchy_map:
        return loc

    # List of prefixes to strip
    prefixes = [
        "Upper", "Lower", "Central", "Middle", "Greater", 
        "Northern", "Southern", "Eastern", "Western", 
        "Northeast", "Southeast", "Northwest", "Southwest"
    ]
    
    parts = loc.split()
    if len(parts) > 1 and parts[0] in prefixes:
        # Try stripping the first word (e.g. "Upper Ohio Valley" -> "Ohio Valley")
        stripped = " ".join(parts[1:])
        if stripped in hierarchy_map:
            return stripped
            
    return loc

# ==========================================
# UPDATED: Cluster ID (Handles Siblings)
# ==========================================
def get_cluster_id(loc, hierarchy_map):
    """
    1. Normalizes the name (Upper Ohio Valley -> Ohio Valley).
    2. Finds the Root (Parent traversal).
    3. CHECKS SIBLINGS to ensure they share the same ID.
    """
    # 1. Normalize
    clean_loc = normalize_location_name(loc, hierarchy_map)
    
    if not clean_loc: return "UNKNOWN"
    
    # 2. Find Root (Walk up parents)
    current = clean_loc
    visited = set()
    while current in hierarchy_map:
        if current in visited: break
        visited.add(current)
        parents = hierarchy_map[current].get('parents', [])
        if not parents:
            break
        current = parents[0]
    
    root = current
    
    # 3. Sibling Check (The "Sideways" Fix)
    # If this root has siblings, we need a consistent ID for all of them.
    # We sort all siblings + the root alphabetically and pick the first one.
    # This ensures "Great Lakes" and "Ohio Valley" both get the same ID (e.g. "Great Lakes")
    if root in hierarchy_map:
        siblings = hierarchy_map[root].get('siblings', [])
        if siblings:
            # Create a group of [Root, Sibling1, Sibling2...]
            cluster_group = sorted([root] + siblings)
            return cluster_group[0] # Return the alphabetical leader as the ID

    return root

def extract_locations_with_indices(sentence):
    all_patterns = []
    for abbr, full_name in STATE_ABBREVIATION_MAP.items():
        for direction in CARDINAL_DIRECTIONS:
            all_patterns.append((f"{direction} {abbr}", f"{direction.title()} {full_name}"))
    
    for state in STATES:
        for direction in CARDINAL_DIRECTIONS:
             all_patterns.append((f"{direction} {state}", f"{direction.title()} {state.title()}"))
    json_locations = list(HIERARCHY_MAP.keys())
    all_keywords = (
        MOUNTAIN_RANGES + MAJOR_RIVERS + VALLEYS_BASINS_PLAINS + GREAT_LAKES + 
        OCEANS_BAYS + CAPES_ISLANDS_PENINSULAS + CANADIAN_PROVINCES + 
        NEIGHBORING_REGIONS + MISC_REGIONS + STATES + SAFE_STATE_ABBREVIATIONS + 
        LOCAL_CONTEXT_TERMS + json_locations
    )
    for kw in all_keywords:
        all_patterns.append((kw, kw.title()))
        for direction in CARDINAL_DIRECTIONS:
            if not any(d in kw.lower().split() for d in CARDINAL_DIRECTIONS):
                all_patterns.append((f"{direction} {kw}", f"{direction.title()} {kw.title()}"))

    hwy_matches = re.findall(r'\b(I-\s*|Interstate\s+)(\d{1,3})\b', sentence, re.IGNORECASE)
    for match in hwy_matches:
        all_patterns.append((match[0] + match[1], f"I-{match[1]}"))

    all_patterns.sort(key=lambda x: len(x[0]), reverse=True)
    found_locations = []
    claimed = [False] * len(sentence)
    sentence_lower = sentence.lower()

    for search_term, normalized_name in all_patterns:
        if not isinstance(search_term, str): continue
        search_term_lower = search_term.lower()
        start_index = 0
        while True:
            pos = sentence_lower.find(search_term_lower, start_index)
            if pos == -1: break 
            is_start_boundary = (pos == 0) or (not sentence[pos - 1].isalnum())
            end_pos = pos + len(search_term)
            is_end_boundary = (end_pos == len(sentence)) or (not sentence[end_pos].isalnum())

            if is_start_boundary and is_end_boundary:
                if not any(claimed[pos:end_pos]):
                    found_locations.append({
                        'name': normalized_name,
                        'start': pos,
                        'end': end_pos,
                        'center': (pos + end_pos) / 2 
                    })
                    for i in range(pos, end_pos):
                        claimed[i] = True
            start_index = pos + 1
            
    coords = extract_coordinates_with_indices(sentence)
    found_locations.extend(coords)
    found_locations.sort(key=lambda x: x['start'])
    
    return found_locations

def is_sentence_valid(sentence):
    return True

def analyze_weather_text(text):
    if not text: return []
    
    extracted_data = []
    sentences = split_into_sentences(text)

    # 1. Flatten your PRESSURE_POLARITY dict for efficient searching
    # We map every keyword back to its Category (HIGH/LOW) and create a sorting list
    pressure_map = {}
    for polarity, keywords in PRESSURE_POLARITY.items():
        for kw in keywords:
            pressure_map[kw.lower()] = polarity

    # Sort by length (descending) to match "Upper Level Low" before "Low"
    sorted_pressure_terms = sorted(pressure_map.keys(), key=len, reverse=True)

    for sentence in sentences:
        if not is_sentence_valid(sentence): 
            continue

        sentence_lower = sentence.lower()
        
        # --- A. Extract Locations First ---
        locations_in_sentence = extract_locations_with_indices(sentence)
        times = extract_time(sentence) # Grab time once per sentence
        
        # --- B. Extract Pressure Systems ---
        pressure_matches = []
        claimed_indices = set()

        for term in sorted_pressure_terms:
            # Regex \b ensures we don't match "low" inside "slow"
            # escape(term) allows terms like "high-pressure" to work safely
            pattern = r'\b' + re.escape(term) + r'\b'
            
            for match in re.finditer(pattern, sentence_lower):
                # Overlap Check
                indices = set(range(match.start(), match.end()))
                if not indices.intersection(claimed_indices):
                    pressure_matches.append({
                        'term': term, # The raw text found
                        'polarity': pressure_map[term], # HIGH or LOW
                        'start': match.start(),
                        'end': match.end(),
                        'center': (match.start() + match.end()) / 2
                    })
                    claimed_indices.update(indices)

        # Sort pressures by start index (reading order)
        pressure_matches.sort(key=lambda x: x['start'])

        # --- C. The Pairing Logic (Forward-Only, Multi-Match) ---
        for press in pressure_matches:
            
            # Find ALL locations that appear AFTER this pressure system
            # Logic: Location Start Index > Pressure End Index
            valid_locs = [
                loc for loc in locations_in_sentence 
                if loc['start'] > press['end']
            ]

            # If NO locations match, we still want to record the pressure system
            # (Context: You might want to know a system exists even if it has no location)
            if not valid_locs:
                # Optional: Uncomment if you want to track systems with "UNKNOWN" location
                # extracted_data.append({
                #     'phenomenon': press['term'],
                #     'category': 'pressure_systems',
                #     'polarity': press['polarity'],
                #     'location': None, 
                #     'sentence': sentence
                # })
                continue 

            # Create an entry for EVERY valid location
            for loc in valid_locs:
                extracted_data.append({
                    'phenomenon': press['term'], # e.g. "trough"
                    'category': 'pressure_systems',
                    'polarity': press['polarity'], # e.g. "LOW"
                    'time': times,
                    'location': loc['name'], # e.g. "Great Lakes"
                    'sentence': sentence,
                    'start_index': press['start']
                })

    return extracted_data


def extract_state_from_meta(meta_location, sample_id=None):
    """Prioritizes the station ID from the JSON 'id' field, uses meta_location as fallback."""
    
    # 1. PRIMARY: Extract the 3-letter code from the ID string (e.g., "FGF_feb...")
    if sample_id:
        station_code = str(sample_id).split('_')[0].upper()
        if station_code in STATION_NAME_MAP:
            return STATION_NAME_MAP[station_code]

    # 2. FALLBACK: Parse the location string (e.g., "Grand Forks, North Dakota")
    if not meta_location: return None
    
    parts = meta_location.split(',')
    state_candidate = parts[-1].strip().title() # .title() converts "NORTH DAKOTA" to "North Dakota"
    
    # Check if it's a known full state name
    if state_candidate in STATES:
        return state_candidate
        
    # Check if it's a 2-letter abbreviation
    if state_candidate.upper() in STATE_ABBREVIATION_MAP:
        return STATE_ABBREVIATION_MAP[state_candidate.upper()]
            
    return state_candidate

def resolve_local_context(loc, meta_location, sample_id=None):
    if not loc: return "UNKNOWN"
    
    l_clean = loc.strip().lower()
    
    # Pass the sample_id in!
    context_state = extract_state_from_meta(meta_location, sample_id) 
    
    # 1. Coastal logic
    if l_clean in ["the coast", "the coastline", "coast", "coastal"]:
        if context_state in COAST_STATE_GROUPS.get("West Coast", []): return "West Coast"
        if context_state in COAST_STATE_GROUPS.get("East Coast", []): return "East Coast"
        if context_state in COAST_STATE_GROUPS.get("Gulf Coast", []): return "Gulf Coast"

    # 2. CHECK SYNONYMS ("The State", "Here")
    is_synonym = l_clean in [t.lower() for t in LOCAL_CONTEXT_TERMS]
    
    if is_synonym and context_state:
        return context_state

    return loc.strip()

# ==========================================
# 3. HIERARCHY LOGIC (Siblings/Cousins)
# ==========================================
def get_canonical_root(loc, hierarchy_map):
    """
    Walks up the hierarchy tree until it hits a root node.
    This creates a 'Common Ancestor' ID for matching.
    """
    if not loc: return None
    current = loc
    visited = set()
    
    # If not in map, return itself (it is its own root)
    if current not in hierarchy_map:
        return current

    # Traverse up
    while current in hierarchy_map:
        if current in visited: break # Safety break
        visited.add(current)
        
        parents = hierarchy_map[current].get('parents', [])
        if not parents:
            # No parents = this is the root
            return current
        
        # Move up to the first parent
        current = parents[0]
        
    return current

# ==========================================
# 4. SCORING FUNCTION (With Debugging)
# ==========================================
def get_lineage(loc, hierarchy_map):
    """
    Returns a list representing the path from the location to the top root.
    Example: ['Columbus', 'Ohio', 'Ohio Valley', 'Central US', 'US']
    """
    path = [loc]
    current = loc
    visited = set()
    
    while current in hierarchy_map:
        if current in visited: break
        visited.add(current)
        
        parents = hierarchy_map[current].get('parents', [])
        if not parents:
            break
        
        # Determine the primary parent (first one)
        # You might need the case-insensitive helper here if keys are messy
        parent = parents[0] 
        path.append(parent)
        current = parent
        
    return path

def calculate_graph_distance(loc1, loc2, hierarchy_map, max_hops=4, stop_nodes=None):
    """
    Calculates the shortest path between two locations.
    Prevents traversing *through* massive hub nodes (like 'Canada' or 'US') 
    to stop distant locations from improperly clustering.
    """
    # Define your massive hub nodes here
    if stop_nodes is None:
        stop_nodes = {"Canada", "CONUS", "Eastern Canada", "Central Canada", "Western Canada", "Eastern CONUS", "Western CONUS", "Central CONUS", "Central Plains", "Ohio Valley", "Great Lakes", "Central U.S.", "Eastern U.S.", "Western U.S.", "Central United States", "Eastern United States", "Western United States", "Central U", "Eastern U", "Western U", "Eastern US", "Western US", "Central US", "Midwest", "The Plains"}

    if loc1 == loc2: return 0
    if loc1 not in hierarchy_map and loc2 not in hierarchy_map: 
        return float('inf')

    # Queue stores: (current_location, current_distance_in_hops)
    queue = deque([(loc1, 0)])
    visited = {loc1}

    while queue:
        current_node, current_dist = queue.popleft()

        # If we reached the target, return the distance!
        if current_node == loc2:
            return current_dist

        # Stop exploring if we hit the maximum allowed hops
        if current_dist >= max_hops:
            continue

        # THE FIX: If this node is a massive hub (and not our starting location),
        # do not let the search use it as a bridge to discover other places.
        if current_node in stop_nodes and current_node != loc1:
            continue

        # Gather ALL valid next steps (Parents, Siblings, and Children)
        neighbors = set()
        
        # 1. Hop up to Parents
        parents = hierarchy_map.get(current_node, {}).get('parents', [])
        neighbors.update(parents)
        
        # 2. Hop sideways to Siblings
        siblings = hierarchy_map.get(current_node, {}).get('siblings', [])
        neighbors.update(siblings)
        
        # 3. Hop down to Children 
        for potential_child, data in hierarchy_map.items():
            if current_node in data.get('parents', []):
                neighbors.add(potential_child)

        # Explore the gathered neighbors
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_dist + 1))

    # If the queue empties and we never found loc2
    return float('inf')

# ==========================================
# REPLACED: SCORING FUNCTION (Pairwise)
# ==========================================

def calculate_space_pressure_score(pred_objs_raw, ref_objs_raw, meta_location, sample_id=None):
    # --- 1. PRE-PROCESSING & CLEANING ---
    def clean_objects(raw_objs):
        cleaned = []
        for o in raw_objs:
            if o['category'] != 'pressure_systems': continue
            pol = get_pressure_polarity(o['phenomenon'])
            if pol not in ['HIGH', 'LOW']: continue
            if not o['location']: continue
            
            # --- START DEBUG BLOCK ---
            raw_loc = o['location']
            if raw_loc.lower() in ["here", "the state"]:
                print(f"\n[DEBUG] -----------------")
                print(f"[DEBUG] Found target: '{raw_loc}'")
                print(f"[DEBUG] sample_id passed in: '{sample_id}'")
                print(f"[DEBUG] meta_location passed in: '{meta_location}'")
                
                c_state = extract_state_from_meta(meta_location, sample_id)
                print(f"[DEBUG] extract_state_from_meta returned: '{c_state}'")
                
                r_loc = resolve_local_context(raw_loc, meta_location, sample_id)
                print(f"[DEBUG] resolve_local_context returned: '{r_loc}'")
            # --- END DEBUG BLOCK ---

            r_loc = resolve_local_context(o['location'], meta_location, sample_id)
            c_loc = normalize_location_name(r_loc, HIERARCHY_MAP)
            
            cleaned.append({**o, 'polarity': pol, 'clean_loc': c_loc})
        return cleaned

    pred_objs = clean_objects(pred_objs_raw)
    ref_objs = clean_objects(ref_objs_raw)

    if not pred_objs and not ref_objs:
        return None  # Skip empty samples

    # --- 2. GRAPH CLUSTERING (The Fix!) ---
    # Gather ALL unique locations present in this specific sample
    
    all_unique_locs = list(set([p['clean_loc'] for p in pred_objs] + [r['clean_loc'] for r in ref_objs]))
    if sample_id is not None: 
        print(f"\n--- TREE DISTANCE DEBUG FOR SAMPLE {sample_id} ---")
        for loc in all_unique_locs:
            print(f"Lineage of '{loc}': {get_lineage(loc, HIERARCHY_MAP)}")
        
        print("\nDistance Matrix:")
        for i in range(len(all_unique_locs)):
            for j in range(i + 1, len(all_unique_locs)):
                loc1 = all_unique_locs[i]
                loc2 = all_unique_locs[j]
                dist = calculate_graph_distance(loc1, loc2, HIERARCHY_MAP)
                print(f"  {loc1} <-> {loc2} = {dist} hops")
        print("---------------------------------------")
    
    # We will use a Disjoint Set Union (DSU) to group locations that are close to each other
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

    # Compare every location to every other location. If they are close, merge them!
    STOP_LOCATIONS = {"Canada", "CONUS", "Eastern Canada", "Central Canada", "Western Canada", "Eastern CONUS", "Western CONUS", "Central CONUS", "Central Plains", "Ohio Valley", "Great Lakes", "Central U.S.", "Eastern U.S.", "Western U.S.", "Central United States", "Eastern United States", "Western United States", "Central U", "Eastern U", "Western U", "Eastern US", "Western US", "Central US", "Midwest", "The Plains"}
    for i in range(len(all_unique_locs)):
        for j in range(i + 1, len(all_unique_locs)):
            loc1 = all_unique_locs[i]
            loc2 = all_unique_locs[j]
            
            # --- Condition A: They are close via the Parent Tree ---
            dist = calculate_graph_distance(loc1, loc2, HIERARCHY_MAP)
            
            # Dynamically adjust the threshold if one of the endpoints is a stop location
            current_threshold = DISTANCE_THRESHOLD # Normally 2
            if loc1 in STOP_LOCATIONS or loc2 in STOP_LOCATIONS:
                current_threshold = 1
                
            if dist <= current_threshold:
                union(loc1, loc2)
                continue
                
            # --- Condition B: The JSON explicitly lists them as siblings ---
            loc1_siblings = HIERARCHY_MAP.get(loc1, {}).get('siblings', [])
            loc2_siblings = HIERARCHY_MAP.get(loc2, {}).get('siblings', [])
            
            if loc2 in loc1_siblings or loc1 in loc2_siblings:
                union(loc1, loc2)

    # Build the clusters based on their unified graph roots
    loc_clusters = defaultdict(lambda: {'p_idxs': set(), 'r_idxs': set()})

    for i, p in enumerate(pred_objs):
        cluster_root = find(p['clean_loc'])
        loc_clusters[cluster_root]['p_idxs'].add(i)

    for j, r in enumerate(ref_objs):
        cluster_root = find(r['clean_loc'])
        loc_clusters[cluster_root]['r_idxs'].add(j)

    # --- 3. SCORING (Local Average) ---
    def get_low_ratio(obj_list):
        if not obj_list: return 0.5 
        low_count = sum(1 for o in obj_list if o['polarity'] == 'LOW')
        return low_count / len(obj_list)

    cluster_scores = []
    global_matched_pred_indices = set()
    global_matched_ref_indices = set()

    for root_key, indices in loc_clusters.items():
        p_idx_set = indices['p_idxs']
        r_idx_set = indices['r_idxs']
        
        # COVERAGE LOGIC: A match is only made if BOTH texts contributed to this cluster
        if len(p_idx_set) > 0 and len(r_idx_set) > 0:
            global_matched_pred_indices.update(p_idx_set)
            global_matched_ref_indices.update(r_idx_set)

        local_preds = [pred_objs[i] for i in p_idx_set]
        local_refs = [ref_objs[i] for i in r_idx_set]
        
        p_ratio = get_low_ratio(local_preds)
        r_ratio = get_low_ratio(local_refs)
        
        local_score = 1.0 - abs(p_ratio - r_ratio)
        cluster_scores.append(local_score)

    match_score = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0

    # --- 4. COVERAGE CALCULATION ---
    total_objects = len(pred_objs) + len(ref_objs)
    matched_objects = len(global_matched_pred_indices) + len(global_matched_ref_indices)

    coverage_ratio = (matched_objects / total_objects) if total_objects > 0 else 1.0
    final_space = match_score * coverage_ratio

    # --- DEBUG PRINT ---
    if sample_id is not None:
        print(f"\n--- DEBUG SAMPLE {sample_id} ---")
        print(f"Preds: {[p['polarity'] + '@' + p['clean_loc'] for p in pred_objs]}")
        print(f"Refs : {[r['polarity'] + '@' + r['clean_loc'] for r in ref_objs]}")
        print(f"Clusters Evaluated (Unified Roots): {list(loc_clusters.keys())}")
        for k, v in loc_clusters.items():
             p_debug = [pred_objs[i]['polarity'] + '@' + pred_objs[i]['clean_loc'] for i in v['p_idxs']]
             r_debug = [ref_objs[i]['polarity'] + '@' + ref_objs[i]['clean_loc'] for i in v['r_idxs']]
             print(f"  > Super-Cluster '{k}': Preds={p_debug} Refs={r_debug}")
        print(f"Avg Match Score: {match_score:.2f} | Coverage: {coverage_ratio:.2f}")

    return final_space, match_score, coverage_ratio

# ==========================================
# UPDATED EXECUTION LOOP
# ==========================================
def process_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Invalid JSON file.")
        return
    
    # --- LIMIT TO TEST SAMPLES (Remove [:10] when ready for full run) ---
    original_len = len(data)
    data = data[:10] 
    print(f"Data limited to {len(data)} samples (Original: {original_len})")
    
    results = []
    
    # We need separate lists now
    space_list = []          # All samples
    coverage_list = []       # All samples
    conditional_match_list = [] # ONLY samples with coverage > 0
    
    ignored_count = 0
    start_time = time.time()

    for i, sample in enumerate(tqdm(data, desc="Scoring")):
        
        pred_text = sample.get('prediction', '')
        ref_text = sample.get('reference', '')
        
        pred_objs = analyze_weather_text(pred_text)
        ref_objs = analyze_weather_text(ref_text)
        
        # --- THE FIX IS HERE ---
        # 1. Grab the actual ID string from your JSON (Fallback to 'None' if missing)
        actual_record_id = sample.get('id', None)
        
        # 2. Pass actual_record_id into the updated record_id parameter!
        metrics = calculate_space_pressure_score(
            pred_objs, 
            ref_objs, 
            sample.get('location'), 
            sample_id=actual_record_id
        )
        # -----------------------
        
        if metrics is None:
            # Both Prediction and Reference were empty (no weather to score)
            ignored_count += 1
            continue 

        space, match, cov = metrics
        
        # 1. Always track SPACE and Coverage (Global metrics)
        space_list.append(space)
        coverage_list.append(cov)
        
        # 2. CONDITIONAL MATCH SCORE
        # Only include the match score if we actually found a matching location.
        # This answers: "When we DO find the location, how often is the Pressure correct?"
        if cov > 0:
            conditional_match_list.append(match)

        results.append({
            "id": sample.get('id', i), 
            "score": space, 
            "match_score": match, 
            "coverage_ratio": cov,
            "included_in_match_avg": (cov > 0) # Helpful flag for CSV analysis
        })
    
    # Helper for stats
    def get_stats_sem(data_list):
        n = len(data_list)
        if n == 0: return 0.0, 0.0
        if n == 1: return data_list[0], 0.0
        mean_val = statistics.mean(data_list)
        stdev_val = statistics.stdev(data_list)
        sem_val = stdev_val / math.sqrt(n)
        return mean_val, sem_val

    avg_space, sem_space = get_stats_sem(space_list)
    avg_cov, sem_cov = get_stats_sem(coverage_list)
    
    # Calculate Conditional Match Stats
    avg_match, sem_match = get_stats_sem(conditional_match_list)
    
    elapsed = time.time() - start_time

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nDone! Processed {len(data)} samples in {elapsed:.2f}s.")
    print(f"Excluded {ignored_count} samples (No pressure systems in Ref or Pred).")
    
    print("\n--- FINAL SCORES ---")
    print("Scores reported as: Mean ± Standard Error (SEM)")
    print(f"SPACE Score (Overall Quality):  {avg_space:.4f} (± {sem_space:.4f})")
    print(f"Coverage Ratio (Recall):        {avg_cov:.4f} (± {sem_cov:.4f})")
    print("-" * 30)
    print(f"Conditional Match Score:        {avg_match:.4f} (± {sem_match:.4f})")
    print(f"   (Calculated from {len(conditional_match_list)} samples where location was found)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPACE Pressure Evaluation")
    
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    
    args = parser.parse_args()
    process_dataset(args.input_json, args.output_json)