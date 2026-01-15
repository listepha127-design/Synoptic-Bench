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

WEATHER_PHENOMENA = {
    'precipitation': ['rain', 'rains', 'drizzle', 'showers', 'precipitation', 'precip', 'downpours', 'sprinkles', 'sleet', 'snow', 'flurries', 'hail', 'ice pellets','flooding', 'flood', 'inundation', 'flash flood'],
    'wind': ['wind', 'gusts', 'breezy', 'gale', 'squalls'],
    'temperature': ['heat', 'warmth', 'cold', 'chill'], 
    'pressure_systems': ['ridge', 'ridging', 'upper high', 'upper low' 'trough', 'troughing', 'high pressure', 'low pressure', 'high-pressure', 'low-pressure', 'cyclone', 'anticyclone', 'closed low', 'blocking', 'cut-off low'], 
    'clouds_visibility': ['fog', 'mist', 'haze', 'clouds', 'cloudy', 'overcast', 'visibility'],
    'severe': ['tornado', 'watersout', 'derecho', 'blizzard']
}

PRESSURE_POLARITY = {
    "HIGH": ['ridge', 'high pressure', 'high-pressure', 'upper high', 'anticyclone', 'blocking', 'ridging'],
    "LOW": ['trough', 'low pressure', 'low-pressure', 'upper low', 'cyclone', 'closed low', 'cut-off low', 'troughing']
}

PROBABILITY_MODIFIERS = ['light', 'slight', 'some', 'chance', 'chances', 'possible', 'likely', 'potential', 'isolated', 'scattered']
EXCLUDED_MODELS = ['ECMWF', 'EURO', 'HRRR', 'ECCC', 'CMC', 'GEM', 'NAM', 'UKMET', 'ICON', 'RAP', 'SREF', 'HREF']
EXCLUDED_SURFACE_TERMS = ['surface', 'sfc', 'ground', '2m', '10m', 'freezing', 'freeze', 'frost', 'dewpoint', 'dew point', 'humid']
EXCLUDED_TIME_TERMS = ['next week', 'extended range', 'long range', 'day 4', 'day 5', 'day 6', 'day 7', '8-14 day', 'outlook', 'climatology']
LOCAL_CONTEXT_TERMS = ["The Region", "The Area", "The CWA", "Overhead", "Forecast Area", "The Forecast Area", "Our Area", "This Area", "The Coast", "The Coastline", 'The Northeast', 'The Southeast', 'The Northwest', 'The Southwest','The North', 'The South', 'The East', 'The West',]
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
CANADIAN_PROVINCES = ['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba', 'Saskatchewan', 'Labrador', 'Nova Scotia']
NEIGHBORING_REGIONS = ['Baja California', 'Bahamas', 'Canada', 'Cuba', 'Mexico']
MISC_REGIONS = ['Tri-State','Twin Cities', 'Tri-Cities', 'West Coast', 'East Coast', 'New England', 'Mid-Atlantic', 'Big Bend', 'Concho Valley', 'Mississippi Valley', 'Ohio Valley', 'Carolinas', 'Downeast', 'Edwards Plateau', 'Finger Lakes', 'Four Corners', 'Hill Country', 'Intermountain West', 'Mid-South', 'Midwest', 'Missouri Bootheel', 'Mohave', 'Pacific Northwest', 'The Panhandle', 'The Thumb', 'Trans Pecos', 'CWA', 'Eastern Seaboard', 'Plains', 'Eastern CONUS', 'Western CONUS', 'Northern CONUS', 'Southern CONUS', 'Eastern U', 'Western U', 'Northern U', 'Southern U']
HIERARCHY_MAP = {}
if os.path.exists("location_hierarchy.json"):
    try:
        with open("location_hierarchy.json", "r") as f:
            HIERARCHY_MAP = json.load(f)
    except Exception as e:
        print(f"Error loading hierarchy: {e}")


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

def extract_locations_with_indices(sentence):
    all_patterns = []
    for abbr, full_name in STATE_ABBREVIATION_MAP.items():
        for direction in CARDINAL_DIRECTIONS:
            all_patterns.append((f"{direction} {abbr}", f"{direction.title()} {full_name}"))
    
    for state in STATES:
        for direction in CARDINAL_DIRECTIONS:
             all_patterns.append((f"{direction} {state}", f"{direction.title()} {state.title()}"))
             
    all_keywords = (
        MOUNTAIN_RANGES + MAJOR_RIVERS + VALLEYS_BASINS_PLAINS + GREAT_LAKES + 
        OCEANS_BAYS + CAPES_ISLANDS_PENINSULAS + CANADIAN_PROVINCES + 
        NEIGHBORING_REGIONS + MISC_REGIONS + STATES + STATE_ABBREVIATIONS + 
        LOCAL_CONTEXT_TERMS
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
    all_phenomena_keywords = {}
    for cat, keywords in WEATHER_PHENOMENA.items():
        for kw in keywords:
            all_phenomena_keywords[kw] = cat
            
    sentences = split_into_sentences(text)
    extracted_data = []

    for sentence in sentences:
        if not is_sentence_valid(sentence): 
            continue

        sentence_lower = sentence.lower()
        phenomena_matches = []
        sorted_kws = sorted(all_phenomena_keywords.items(), key=lambda x: len(x[0]), reverse=True)
        claimed_indices = set()
        
        for kw, category in sorted_kws:
            for match in re.finditer(r'\b' + re.escape(kw) + r'\b', sentence_lower):
                if kw == 'trough':
                    preceding = sentence_lower[:match.start()].strip()
                    if preceding.endswith("shortwave") or preceding.endswith("short-wave"):
                        continue

                indices = set(range(match.start(), match.end()))
                if not indices.intersection(claimed_indices):
                    phenomena_matches.append({
                        'phenomenon': kw,
                        'category': category,
                        'start': match.start(),
                        'end': match.end(),
                        'center': (match.start() + match.end()) / 2
                    })
                    claimed_indices.update(indices)
        
        phenomena_matches.sort(key=lambda x: x['start'])

        if phenomena_matches:
            locations_with_indices = extract_locations_with_indices(sentence)
            times = extract_time(sentence)
            
            for phenom_obj in phenomena_matches:
                best_location = None
                
                if locations_with_indices:
                    min_dist = float('inf')
                    
                    for loc_obj in locations_with_indices:
                        dist = abs(loc_obj['center'] - phenom_obj['center'])

                        if dist < min_dist:
                            min_dist = dist
                            best_location = loc_obj['name']

                
                extracted_data.append({
                    'phenomenon': phenom_obj['phenomenon'],
                    'category': phenom_obj['category'],
                    'time': times,
                    'location': best_location,  
                    'sentence': sentence,
                    'start_index': phenom_obj['start'] 
                })
    return extracted_data


def extract_state_from_meta(meta_location):
    if not meta_location: return None
    parts = meta_location.split(',')
    if len(parts) > 1: return parts[-1].strip()
    return meta_location.strip()

def resolve_local_context(loc, meta_location):
    context_state = extract_state_from_meta(meta_location)
    if not loc: return "UNKNOWN" 
    l = loc.strip()
    
    if "," in l:
        parts = l.split(',')
        l = parts[-1].strip()

    if l.lower() in ["the coast", "the coastline", "coast", "coastal", "along the coast"]:
        if context_state in COAST_STATE_GROUPS["West Coast"]: return "West Coast"
        if context_state in COAST_STATE_GROUPS["East Coast"]: return "East Coast"
        if context_state in COAST_STATE_GROUPS["Gulf Coast"]: return "Gulf Coast"

    is_context_synonym = (l == context_state) or \
                         (l in LOCAL_CONTEXT_TERMS) or \
                         (l in HIERARCHY_MAP and "LOCAL_CONTEXT" in HIERARCHY_MAP[l].get("siblings", []))
    if is_context_synonym and context_state: return context_state
    return l


def get_pressure_polarity(term):
    t = term.lower()
    for label, keywords in PRESSURE_POLARITY.items():
        if t in keywords: return label
    return "NEUTRAL"

def calculate_space_pressure_score(pred_objs_raw, ref_objs_raw, meta_location):
    pred_objs = [o for o in pred_objs_raw if o['category'] == 'pressure_systems']
    ref_objs = [o for o in ref_objs_raw if o['category'] == 'pressure_systems']
    
    pred_objs = [o for o in pred_objs if get_pressure_polarity(o['phenomenon']) in ['HIGH', 'LOW']]
    ref_objs = [o for o in ref_objs if get_pressure_polarity(o['phenomenon']) in ['HIGH', 'LOW']]

    total_objects = len(pred_objs) + len(ref_objs)
    
    if total_objects == 0:
        return None 


    location_clusters = defaultdict(lambda: {'pred': [], 'ref': []})

    def group_objects(obj_list, key_type):
        for obj in obj_list:
            polarity = get_pressure_polarity(obj['phenomenon'])
            raw_loc = obj['location']
            resolved_loc = resolve_local_context(raw_loc, meta_location)
            if resolved_loc == "UNKNOWN": continue 
                
            parent_loc = resolved_loc
            if resolved_loc in HIERARCHY_MAP:
                parents = HIERARCHY_MAP[resolved_loc].get('parents', [])
                if parents: parent_loc = parents[0]
            
            location_clusters[parent_loc][key_type].append(polarity)

    group_objects(pred_objs, 'pred')
    group_objects(ref_objs, 'ref')

    location_accuracies = []
    
    pred_set = set()
    ref_set = set()

    for loc, data in location_clusters.items():
        p_list = data['pred']
        r_list = data['ref']
        n_p = len(p_list)
        n_r = len(r_list)

        if 'HIGH' in p_list: pred_set.add((loc, 'HIGH'))
        if 'LOW' in p_list:  pred_set.add((loc, 'LOW'))
        
        if 'HIGH' in r_list: ref_set.add((loc, 'HIGH'))
        if 'LOW' in r_list:  ref_set.add((loc, 'LOW'))


        if n_p > 0 and n_r > 0:
            ratio_pred = p_list.count('LOW') / n_p
            ratio_ref = r_list.count('LOW') / n_r
            location_accuracies.append(1.0 - abs(ratio_pred - ratio_ref))


    intersection = pred_set.intersection(ref_set)
    union = pred_set.union(ref_set)
    
    if len(union) == 0:
        return None

    coverage_ratio = len(intersection) / len(union)

    if not location_accuracies:
        match_score = None
    else:
        match_score = sum(location_accuracies) / len(location_accuracies)

    final_space = (match_score * coverage_ratio) if match_score is not None else 0.0
    
    return final_space, match_score, coverage_ratio

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
    
    print(f"Processing {len(data)} samples...")
    
    results = []
    match_scores_list = []
    coverage_list = []
    space_list = []
    ignored_count = 0
    start_time = time.time()

    for i, sample in enumerate(tqdm(data, desc="Scoring")):
        
        pred_text = sample.get('prediction', '')
        ref_text = sample.get('reference', '')
        
        pred_objs = analyze_weather_text(pred_text)
        ref_objs = analyze_weather_text(ref_text)
        
        metrics = calculate_space_pressure_score(pred_objs, ref_objs, sample.get('location'))
        
        if metrics is None:
            ignored_count += 1
            continue 

        space, match, cov = metrics
        
        results.append({
            "id": sample.get('id', i), 
            "score": space, 
            "match_score": match, 
            "coverage_ratio": cov 
        })
        
        space_list.append(space)
        coverage_list.append(cov)
        if match is not None:
            match_scores_list.append(match)

    def get_stats_sem(data_list):
        n = len(data_list)
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return data_list[0], 0.0
            
        mean_val = statistics.mean(data_list)
        stdev_val = statistics.stdev(data_list)
        
        sem_val = stdev_val / math.sqrt(n)
        
        return mean_val, sem_val

    avg_space, sem_space = get_stats_sem(space_list)
    avg_cov, sem_cov = get_stats_sem(coverage_list)
    avg_match, sem_match = get_stats_sem(match_scores_list)
    
    elapsed = time.time() - start_time

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Done! Processed {len(data)} total samples in {elapsed:.2f}s.")
    print(f"Excluded {ignored_count} samples.")
    print("Scores reported as: Mean ± Standard Error (SEM)")
    print(f"SPACE Score:     {avg_space:.4f} (± {sem_space:.4f})")
    print(f"Coverage Ratio:  {avg_cov:.4f} (± {sem_cov:.4f})")
    print(f"Match Score:     {avg_match:.4f} (± {sem_match:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPACE Pressure Evaluation")
    
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    
    args = parser.parse_args()
    process_dataset(args.input_json, args.output_json)