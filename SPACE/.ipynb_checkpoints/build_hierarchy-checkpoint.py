import json

# ==========================================
# 1. CONFIGURATION: OVERRIDES & BLACKLIST
# ==========================================

# A. DELETE THESE NODES ENTIRELY
BLACKLIST_NODES = [
    "West West CONUS",
    "Western Western US",
    "Western The West",
    "Northern The North",
    "Southern The South",
    "Eastern The East",
    "Central The Plains",
    "North Northern US",
    "West Western US", 
    "South Southern US",
    "East Eastern US",
    "North Northern US",
    "West Midwest",
    "Southeast Midwest",
    "Southern Midwest",
    "Soutwest Midwest",
    "North Midwest",
    "Northwest Midwest",
    "West The East",
    "East The East",
    "West The West",
    "East The West",
    "North The West",
    "South The West",
    "Northeast The West",
    "Northwest The West",
    "Southeast The West",
    "Southwest The West",
    "Northeast The East",
    "Northwest The East",
    "Southeast The East",
    "Southwest The East",
    "North The West",
    "South The West",
    "Southern The West",
    "Northern The West",
    "Central The West",
    "West Intermountain West",
    "East Intermountain West",
    "North Intermountain West",
    "South Intermountain West",
    "Northern Intermountain West",
    "Southern Intermountain West",
    "Northeast Intermountain West",
    "Northwest Intermountain West",
    "Southeast Intermountain West",
    "Southwest Intermountain West",
    "West Pacific Northwest",
    "East Pacific Northwest",
    "North Pacific Northwest",
    "South Pacific Northwest",
    "Northern Pacific Northwest",
    "Southern Pacific Northwest",
    "Northeast Pacific Northwest",
    "Northwest Pacific Northwest",
    "Southeast Pacific Northwest",
    "Southwest Pacific Northwest",
    "West Great Basin",
    "East Great Basin",
    "North Great Basin",
    "South Great Basin",
    "Northern Great Basin",
    "Southern Great Basin",
    "Northeast Great Basin",
    "Northwest Great Basin",
    "Southeast Great Basin",
    "Southwest Great Basin",
    "Eastern Great Basin",
    "Western Great Basin",
    "Central California Region",
    "West California Region",
    "East California Region",
    "North California Region",
    "South California Region",
    "Northern California Region",
    "Southern California Region",
    "Northeast California Region",
    "Northwest California Region",
    "Southeast California Region",
    "Southwest California Region",
    "Eastern California Region",
    "Western California Region",
    "Central California Region",
    "West The Plains",
    "East The Plains",
    "North The Plains",
    "South The Plains",
    "Northern The Plains",
    "Southern The Plains",
    "Northeast The Plains",
    "Northwest The Plains",
    "Southeast The Plains",
    "Southwest The Plains",
    "Eastern The Plains",
    "Western The Plains",
    "Central The Plains",
    "West Heartland",
    "East Heartland",
    "North Heartland",
    "South Heartland",
    "Northern Heartland",
    "Southern Heartland",
    "Northeast Heartland",
    "Northwest Heartland",
    "Southeast Heartland",
    "Southwest Heartland",
    "Eastern Heartland",
    "Western Heartland",
    "Central Heartland",
    "West Heartland",
    "East Heartland",
    "North Heartland",
    "South Heartland",
    "Northern Heartland",
    "Southern Heartland",
    "Northeast Heartland",
    "Northwest Heartland",
    "Southeast Heartland",
    "Southwest Heartland",
    "Eastern Heartland",
    "Western Heartland",
    "Central Heartland",
    "West Midwest",
    "East Midwest",
    "North Midwest",
    "South Midwest",
    "Northern Midwest",
    "Southern Midwest",
    "Northeast Midwest",
    "Northwest Midwest",
    "Southeast Midwest",
    "Southwest Midwest",
    "Eastern Midwest",
    "Western Midwest",
    "Central Midwest",
    "West Mississippi Valley",
    "East Mississippi Valley",
    "North Mississippi Valley",
    "South Mississippi Valley",
    "Northern Mississippi Valley",
    "Southern Mississippi Valley",
    "Northeast Mississippi Valley",
    "Northwest Mississippi Valley",
    "Southeast Mississippi Valley",
    "Southwest Mississippi Valley",
    "Eastern Mississippi Valley",
    "Western Mississippi Valley",
    "Central Mississippi Valley",
    "North The East",
    "South The East",
    "Northern The East",
    "Southern The East",
    "Western The East",
    "Central The East",
    "West Mid-Atlantic",
    "East Mid-Atlantic",
    "North Mid-Atlantic",
    "South Mid-Atlantic",
    "Northern Mid-Atlantic",
    "Southern Mid-Atlantic",
    "Northeast Mid-Atlantic",
    "Northwest Mid-Atlantic",
    "Southeast Mid-Atlantic",
    "Southwest Mid-Atlantic",
    "Eastern Mid-Atlantic",
    "Western Mid-Atlantic",
    "Central Mid-Atlantic",
    "West Ohio Valley",
    "East Ohio Valley",
    "North Ohio Valley",
    "South Ohio Valley",
    "Northern Ohio Valley",
    "Southern Ohio Valley",
    "Northeast Ohio Valley",
    "Northwest Ohio Valley",
    "Southeast Ohio Valley",
    "Southwest Ohio Valley",
    "Eastern Ohio Valley",
    "Western Ohio Valley",
    "Central Ohio Valley",
    "West New England",
    "East New England",
    "North New England",
    "South New England",
    "Northern New England",
    "Southern New England",
    "Northeast New England",
    "Northwest New England",
    "Southeast New England",
    "Southwest New England",
    "Eastern New England",
    "Western New England",
    "Central New England",
    "West The North",
    "East The North",
    "North The North",
    "South The North",
    "Northern The North",
    "Southern The North",
    "Northeast The North",
    "Northwest The North",
    "Southeast The North",
    "Southwest The North",
    "Eastern The North",
    "Western The North",
    "Central The North",
    "West The Northwest",
    "East The Northwest",
    "North The Northwest",
    "South The Northwest",
    "Northern The Northwest",
    "Southern The Northwest",
    "Northeast The Northwest",
    "Northwest The Northwest",
    "Southeast The Northwest",
    "Southwest The Northwest",
    "Eastern The Northwest",
    "Western The Northwest",
    "Central The Northwest",
    "West The Southwest",
    "East The Southwest",
    "North The Southwest",
    "South The Southwest",
    "Northern The Southwest",
    "Southern The Southwest",
    "Northeast The Southwest",
    "Northwest The Southwest",
    "Southeast The Southwest",
    "Southwest The Southwest",
    "Eastern The Southwest",
    "Western The Southwest",
    "Central The Southwest",
    "West Grand Canyon",
    "East Grand Canyon",
    "North Grand Canyon",
    "South Grand Canyon",
    "Northern Grand Canyon",
    "Southern Grand Canyon",
    "Northeast Grand Canyon",
    "Northwest Grand Canyon",
    "Southeast Grand Canyon",
    "Southwest Grand Canyon",
    "Eastern Grand Canyon",
    "Western Grand Canyon",
    "Central Grand Canyon",
    "West Four Corners",
    "East Four Corners",
    "North Four Corners",
    "South Four Corners",
    "Northern Four Corners",
    "Southern Four Corners",
    "Northeast Four Corners",
    "Northwest Four Corners",
    "Southeast Four Corners",
    "Southwest Four Corners",
    "Eastern Four Corners",
    "Western Four Corners",
    "Central Four Corners",
    "West Sierra Madre",
    "East Sierra Madre",
    "North Sierra Madre",
    "South Sierra Madre",
    "Northern Sierra Madre",
    "Southern Sierra Madre",
    "Northeast Sierra Madre",
    "Northwest Sierra Madre",
    "Southeast Sierra Madre",
    "Southwest Sierra Madre",
    "Eastern Sierra Madre",
    "Western Sierra Madre",
    "Central Sierra Madre",
    "West Cape Mendocino",
    "East Cape Mendocino",
    "North Cape Mendocino",
    "South Cape Mendocino",
    "Northern Cape Mendocino",
    "Southern Cape Mendocino",
    "Northeast Cape Mendocino",
    "Northwest Cape Mendocino",
    "Southeast Cape Mendocino",
    "Southwest Cape Mendocino",
    "Eastern Cape Mendocino",
    "Western Cape Mendocino",
    "Central Cape Mendocino",
    "West Bay Area",
    "East Bay Area",
    "North Bay Area",
    "South Bay Area",
    "Northern Bay Area",
    "Southern Bay Area",
    "Northeast Bay Area",
    "Northwest Bay Area",
    "Southeast Bay Area",
    "Southwest Bay Area",
    "Eastern Bay Area",
    "Western Bay Area",
    "Central Bay Area",
    "West Red River",
    "East Red River",
    "North Red River",
    "South Red River",
    "Northern Red River",
    "Southern Red River",
    "Northeast Red River",
    "Northwest Red River",
    "Southeast Red River",
    "Southwest Red River",
    "Eastern Red River",
    "Western Red River",
    "Central Red River",
    "West Black Hills",
    "East Black Hills",
    "North Black Hills",
    "South Black Hills",
    "Northern Black Hills",
    "Southern Black Hills",
    "Northeast Black Hills",
    "Northwest Black Hills",
    "Southeast Black Hills",
    "Southwest Black Hills",
    "Eastern Black Hills",
    "Western Black Hills",
    "Central Black Hills",
    "West The Panhandle",
    "East The Panhandle",
    "North The Panhandle",
    "South The Panhandle",
    "Northern The Panhandle",
    "Southern The Panhandle",
    "Northeast The Panhandle",
    "Northwest The Panhandle",
    "Southeast The Panhandle",
    "Southwest The Panhandle",
    "Eastern The Panhandle",
    "Western The Panhandle",
    "Central The Panhandle",
    "West Edwards Plateau",
    "East Edwards Plateau",
    "North Edwards Plateau",
    "South Edwards Plateau",
    "Northern Edwards Plateau",
    "Southern Edwards Plateau",
    "Northeast Edwards Plateau",
    "Northwest Edwards Plateau",
    "Southeast Edwards Plateau",
    "Southwest Edwards Plateau",
    "Eastern Edwards Plateau",
    "Western Edwards Plateau",
    "Central Edwards Plateau",
    "West Hill Country",
    "East Hill Country",
    "North Hill Country",
    "South Hill Country",
    "Northern Hill Country",
    "Southern Hill Country",
    "Northeast Hill Country",
    "Northwest Hill Country",
    "Southeast Hill Country",
    "Southwest Hill Country",
    "Eastern Hill Country",
    "Western Hill Country",
    "Central Hill Country",
    "West Strait of Juan de Fuca",
    "East Strait of Juan de Fuca",
    "North Strait of Juan de Fuca",
    "South Strait of Juan de Fuca",
    "Northern Strait of Juan de Fuca",
    "Southern Strait of Juan de Fuca",
    "Northeast Strait of Juan de Fuca",
    "Northwest Strait of Juan de Fuca",
    "Southeast Strait of Juan de Fuca",
    "Southwest Strait of Juan de Fuca",
    "Eastern Strait of Juan de Fuca",
    "Western Strait of Juan de Fuca",
    "Central Strait of Juan de Fuca",
    "West The Thumb",
    "East The Thumb",
    "North The Thumb",
    "South The Thumb",
    "Northern The Thumb",
    "Southern The Thumb",
    "Northeast The Thumb",
    "Northwest The Thumb",
    "Southeast The Thumb",
    "Southwest The Thumb",
    "Eastern The Thumb",
    "Western The Thumb",
    "Central The Thumb",
    "West Twin Cities",
    "East Twin Cities",
    "North Twin Cities",
    "South Twin Cities",
    "Northern Twin Cities",
    "Southern Twin Cities",
    "Northeast Twin Cities",
    "Northwest Twin Cities",
    "Southeast Twin Cities",
    "Southwest Twin Cities",
    "Eastern Twin Cities",
    "Western Twin Cities",
    "Central Twin Cities",
    "West Lake St. Clair",
    "East Lake St. Clair",
    "North Lake St. Clair",
    "South Lake St. Clair",
    "Northern Lake St. Clair",
    "Southern Lake St. Clair",
    "Northeast Lake St. Clair",
    "Northwest Lake St. Clair",
    "Southeast Lake St. Clair",
    "Southwest Lake St. Clair",
    "Eastern Lake St. Clair",
    "Western Lake St. Clair",
    "Central Lake St. Clair",
    "West Saginaw Bay",
    "East Saginaw Bay",
    "North Saginaw Bay",
    "South Saginaw Bay",
    "Northern Saginaw Bay",
    "Southern Saginaw Bay",
    "Northeast Saginaw Bay",
    "Northwest Saginaw Bay",
    "Southeast Saginaw Bay",
    "Southwest Saginaw Bay",
    "Eastern Saginaw Bay",
    "Western Saginaw Bay",
    "Central Saginaw Bay",
    "West The South",
    "East The South",
    "North The South",
    "South The South",
    "Northern The South",
    "Southern The South",
    "Northeast The South",
    "Northwest The South",
    "Southeast The South",
    "Southwest The South",
    "Eastern The South",
    "Western The South",
    "Central The South",
    "West Deep South",
    "East Deep South",
    "North Deep South",
    "South Deep South",
    "Northern Deep South",
    "Southern Deep South",
    "Northeast Deep South",
    "Northwest Deep South",
    "Southeast Deep South",
    "Southwest Deep South",
    "Eastern Deep South",
    "Western Deep South",
    "Central Deep South",
    "West The Southeast",
    "East The Southeast",
    "North The Southeast",
    "South The Southeast",
    "Northern The Southeast",
    "Southern The Southeast",
    "Northeast The Southeast",
    "Northwest The Southeast",
    "Southeast The Southeast",
    "Southwest The Southeast",
    "Eastern The Southeast",
    "Western The Southeast",
    "Central The Southeast",
    "West Cumberland Plateau",
    "East Cumberland Plateau",
    "North Cumberland Plateau",
    "South Cumberland Plateau",
    "Northern Cumberland Plateau",
    "Southern Cumberland Plateau",
    "Northeast Cumberland Plateau",
    "Northwest Cumberland Plateau",
    "Southeast Cumberland Plateau",
    "Southwest Cumberland Plateau",
    "Eastern Cumberland Plateau",
    "Western Cumberland Plateau",
    "Central Cumberland Plateau",
    "West Mid-South",
    "East Mid-South",
    "North Mid-South",
    "South Mid-South",
    "Northern Mid-South",
    "Southern Mid-South",
    "Northeast Mid-South",
    "Northwest Mid-South",
    "Southeast Mid-South",
    "Southwest Mid-South",
    "Eastern Mid-South",
    "Western Mid-South",
    "Central Mid-South",
    "West Atchafalaya River",
    "East Atchafalaya River",
    "North Atchafalaya River",
    "South Atchafalaya River",
    "Northern Atchafalaya River",
    "Southern Atchafalaya River",
    "Northeast Atchafalaya River",
    "Northwest Atchafalaya River",
    "Southeast Atchafalaya River",
    "Southwest Atchafalaya River",
    "Eastern Atchafalaya River",
    "Western Atchafalaya River",
    "Central Atchafalaya River",
    "West Delmarva Peninsula",
    "East Delmarva Peninsula",
    "North Delmarva Peninsula",
    "South Delmarva Peninsula",
    "Northern Delmarva Peninsula",
    "Southern Delmarva Peninsula",
    "Northeast Delmarva Peninsula",
    "Northwest Delmarva Peninsula",
    "Southeast Delmarva Peninsula",
    "Southwest Delmarva Peninsula",
    "Eastern Delmarva Peninsula",
    "Western Delmarva Peninsula",
    "Central Delmarva Peninsula",
    "West The Northeast",
    "East The Northeast",
    "North The Northeast",
    "South The Northeast",
    "Northern The Northeast",
    "Southern The Northeast",
    "Northeast The Northeast",
    "Northwest The Northeast",
    "Southeast The Northeast",
    "Southwest The Northeast",
    "Eastern The Northeast",
    "Western The Northeast",
    "Central The Northeast",
    "West Northeastern",
    "East Northeastern",
    "North Northeastern",
    "South Northeastern",
    "Northern Northeastern",
    "Southern Northeastern",
    "Northeast Northeastern",
    "Northwest Northeastern",
    "Southeast Northeastern",
    "Southwest Northeastern",
    "Eastern Northeastern",
    "Western Northeastern",
    "Central Northeastern",
    "West Tri-State",
    "East Tri-State",
    "North Tri-State",
    "South Tri-State",
    "Northern Tri-State",
    "Southern Tri-State",
    "Northeast Tri-State",
    "Northwest Tri-State",
    "Southeast Tri-State",
    "Southwest Tri-State",
    "Eastern Tri-State",
    "Western Tri-State",
    "Central Tri-State",
    "West Downeast",
    "East Downeast",
    "North Downeast",
    "South Downeast",
    "Northern Downeast",
    "Southern Downeast",
    "Northeast Downeast",
    "Northwest Downeast",
    "Southeast Downeast",
    "Southwest Downeast",
    "Eastern Downeast",
    "Western Downeast",
    "Central Downeast",
    "West The CWA",
    "East The CWA",
    "North The CWA",
    "South The CWA",
    "Northern The CWA",
    "Southern The CWA",
    "Northeast The CWA",
    "Northwest The CWA",
    "Southeast The CWA",
    "Southwest The CWA",
    "Eastern The CWA",
    "Western The CWA",
    "Central The CWA",
    "West Overhead",
    "East Overhead",
    "North Overhead",
    "South Overhead",
    "Northern Overhead",
    "Southern Overhead",
    "Northeast Overhead",
    "Northwest Overhead",
    "Southeast Overhead",
    "Southwest Overhead",
    "Eastern Overhead",
    "Western Overhead",
    "Central Overhead",
    "West The Area",
    "East The Area",
    "North The Area",
    "South The Area",
    "Northern The Area",
    "Southern The Area",
    "Northeast The Area",
    "Northwest The Area",
    "Southeast The Area",
    "Southwest The Area",
    "Eastern The Area",
    "Western The Area",
    "Central The Area",
    "West This Area",
    "East This Area",
    "North This Area",
    "South This Area",
    "Northern This Area",
    "Southern This Area",
    "Northeast This Area",
    "Northwest This Area",
    "Southeast This Area",
    "Southwest This Area",
    "Eastern This Area",
    "Western This Area",
    "Central This Area",
    "West The Region",
    "East The Region",
    "North The Region",
    "South The Region",
    "Northern The Region",
    "Southern The Region",
    "Northeast The Region",
    "Northwest The Region",
    "Southeast The Region",
    "Southwest The Region",
    "Eastern The Region",
    "Western The Region",
    "Central The Region",
    "West Here",
    "East Here",
    "North Here",
    "South Here",
    "Northern Here",
    "Southern Here",
    "Northeast Here",
    "Northwest Here",
    "Southeast Here",
    "Southwest Here",
    "Eastern Here",
    "Western Here",
    "Central Here",
    "West Local Area",
    "East Local Area",
    "North Local Area",
    "South Local Area",
    "Northern Local Area",
    "Southern Local Area",
    "Northeast Local Area",
    "Northwest Local Area",
    "Southeast Local Area",
    "Southwest Local Area",
    "Eastern Local Area",
    "Western Local Area",
    "Central Local Area",
    "IN",
    
]

MANUAL_OVERRIDES = {
    "Twin Cities": {
        "remove_siblings": ["Ohio", "Indiana", "Illinois", "Missouri", "Michigan", "The Thumb"],
        "parents": ["Minnesota", "Wisconsin"], 
    },
    "Great Lakes": {
        "siblings": ["Great Lakes Region", "Southern Canada", "Canada"],
        "parents": ["Midwest", "Northern US", "The North"], 
    },
    "Eastern Canada": {
        "siblings": ["Maine", "New Hampshire", "Vermont", "New England", "New York", "Pennsylvania"]
    },
    "Southeastern Canada": {
        "siblings": ["Maine", "New Hampshire", "Vermont", "New England", "New York", "Pennsylvania"]
    },
    "Ohio": {
        "remove_siblings": ["Wisconsin", "WI", "MO", "IL", "IA", "Missouri", "Illinois", "Iowa"],
        "siblings": ["West Indiana", "West IN", "North Kentucky", "North KY", "Northern KY", "Northern Kentucky", "Southern Great Lakes", "South Great Lakes", "Southern MI", "South MI", "Southern Michigan", "South Michigan", "Northern WV", "North WV", "Northern West Virginia", "North West Virginia","Lake Erie", "Western PA", "Western Pennsylvania", "West Pennsylvania"]
    },
    "OH": {
        "remove_siblings": ["Wisconsin", "WI", "MO", "IL", "IA", "Missouri", "Illinois", "Iowa"],
        "siblings": ["West Indiana", "West IN", "North Kentucky", "North KY", "Northern KY", "Northern Kentucky", "Southern Great Lakes", "South Great Lakes", "Southern MI", "South MI", "Southern Michigan", "South Michigan", "Northern WV", "North WV", "Northern West Virginia", "North West Virginia","Lake Erie", "Western PA", "Western Pennsylvania", "West Pennsylvania"]
    },
    "Nevada": {
        "parents": ["Great Basin", "Southwest", "Western US"],
        "siblings": ["Utah", "UT", "AZ", "CA", "Oregon", "OR", "Arizona", "California", "West UT", "Western Utah", "West Utah", "Western UT", "ID", "Idaho", "South Idaho", "South ID", "Southern Idaho", "Southern ID", "North AZ", "North Arizona", "Northern AZ", "Northern Arizona", "West AZ", "Western Arizona", "West Arizona", "Western AZ", "Southern CA", "Southern California", "South California", "South CA", "Northestern Arizona", "Northwest Arizona", "Northwestern AZ", "Northwest AZ", "Southeastern CA", "Southeast CA", "Southeastern California", "Southeast California", "Eastern CA", "Eastern California", "Southern OR", "Southern Oregon", "South OR", "Southern OR", "Southeastern OR", "Southeastern Oregon", "Northern CA", "Northern California", "Northeastern CA", "Northeastern California"],
        "remove_parents": ["Southern US", "The South"], 
    },
    "Snake River": {
        "parents": ["Pacific Northwest", "Northern Rockies"],
        "siblings": ["Snake River Plain", "Columbia Basin"]
    },
    "The South": {
        "twins": ["Southern US", "Southeast", "Deep South"]
    }

}

# ==========================================
# 2. THE BACKBONE HIERARCHY
# ==========================================
hierarchy = {
    # --- LEVEL 1: MACRO REGIONS ---
    "Western US": {
        "aliases": ["Western CONUS", "The West", "West Coast", "Western U", "Intermountain West"],
        "subregions": ["Pacific Northwest", "Southwest", "Great Basin", "The Rockies", "Northern Rockies", "Southern Rockies", "California Region"],
        "loose_grouping": True 
    },
    "Central US": {
        "aliases": ["Central CONUS", "The Plains", "Heartland", "Plains"],
        "subregions": ["Northern Plains", "Southern Plains", "Midwest", "Mississippi Valley"],
        "loose_grouping": True
    },
    "Eastern US": {
        "aliases": ["Eastern CONUS", "The East", "East Coast", "Eastern U", "Eastern Seaboard"],
        "subregions": ["Northeast", "Southeast", "Mid-Atlantic", "Ohio Valley", "New England"],
        "loose_grouping": True
    },
    "Northern US": { 
        "aliases": ["Northern CONUS", "The North", "Northern U"],
        "subregions": ["Pacific Northwest", "Northern Plains", "Great Lakes", "Northeast", "New England"],
        "loose_grouping": True
    },
    "Southern US": { 
        "aliases": ["Southern CONUS", "Southern U"], 
        "subregions": ["Southern Plains", "Southeast", "Gulf Coast"],
        "loose_grouping": True
    },

    # --- LEVEL 2: REGIONAL DEFINITIONS ---
    "Pacific Northwest": {
        "parents": ["Western US", "Northern US", "The Northwest"],
        "states": ["Washington", "Oregon", "Idaho"],
        "features": ["Cascades", "Columbia Basin", "Olympic Peninsula", "Puget Sound", "Snake River", "Snake River Plain", "Strait of Juan de Fuca"]
    },
    "Southwest": {
        "parents": ["Western US", "The Southwest"], 
        "states": ["Arizona", "New Mexico", "Utah", "Colorado", "Nevada"], 
        "features": ["Four Corners", "Grand Canyon", "Mohave", "Sierra Madre", "Colorado River"]
    },
    "California Region": {
        "parents": ["Western US", "West Coast"],
        "states": ["California"],
        "features": ["Sierra Nevada", "Sacramento Valley", "San Joaquin Valley", "Bay Area", "Cape Mendocino", "Mohave"]
    },
    "Northern Plains": {
        "parents": ["Central US", "Northern US", "The Plains"],
        "states": ["North Dakota", "South Dakota", "Nebraska", "Minnesota", "Montana", "Wyoming"],
        "features": ["Missouri River", "Red River", "Black Hills"]
    },
    "Southern Plains": {
        "parents": ["Central US", "Southern US", "The Plains"],
        "states": ["Kansas", "Oklahoma", "Texas"],
        "features": ["Panhandle", "The Panhandle", "Red River", "Edwards Plateau", "Hill Country", "Trans Pecos", "Big Bend", "Concho Valley"]
    },
    "Midwest": {
        "parents": ["Central US", "Eastern US"],
        "states": ["Illinois", "Indiana", "Ohio", "Michigan", "Wisconsin", "Iowa", "Missouri"],
        "features": ["Great Lakes", "Corn Belt", "The Thumb", "Twin Cities", "Mississippi River", "Ohio River"]
    },
    "Great Lakes Region": { 
        "parents": ["Midwest", "Northern US", "Great Lakes"],
        "states": ["Michigan", "Wisconsin"],
        "features": ["Lake Superior", "Lake Michigan", "Lake Huron", "Lake Erie", "Lake Ontario", "Lake St. Clair", "Saginaw Bay"]
    },
    "Southeast": {
        "parents": ["Eastern US", "Southern US", "The Southeast"],
        "aliases": ["The South", "Deep South"], 
        "states": ["Florida", "Georgia", "Alabama", "Mississippi", "South Carolina", "North Carolina", "Tennessee", "Kentucky", "Virginia", "West Virginia", "Arkansas", "Louisiana"],
        "features": ["Gulf Coast", "Appalachians", "Smoky Mountains", "Blue Ridge", "Cumberland Plateau", "Chattahoochee River", "Florida Panhandle", "Mississippi River", "Mid-South"]
    },
    "Gulf Coast": {
        "parents": ["Southern US", "Southeast"],
        "states": ["Texas", "Louisiana", "Mississippi", "Alabama", "Florida"],
        "features": ["Gulf of Mexico", "Gulf", "Gulf Waters", "Atchafalaya River", "Lake Okeechobee"]
    },
    "Mid-Atlantic": {
        "parents": ["Eastern US"],
        "states": ["New York", "Pennsylvania", "New Jersey", "Delaware", "Maryland", "Virginia", "West Virginia"],
        "features": ["Chesapeake Bay", "Delmarva Peninsula", "Poconos", "Catskills", "Susquehanna", "Outer Banks"] 
    },
    "Northeast": {
        "parents": ["Eastern US", "The Northeast", "Northeast", "Northeastern"],
        "states": ["New York", "Pennsylvania", "New Jersey", "Connecticut", "Rhode Island", "Massachusetts", "Vermont", "New Hampshire", "Maine"],
        "features": ["Adirondacks", "Catskills", "Hudson River", "Hudson Valley", "Mohawk Valley", "Finger Lakes", "Tri-State"]
    },
    "New England": {
        "parents": ["Eastern US", "Northeast"],
        "states": ["Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut"],
        "features": ["White Mountains", "Green Mountains", "Connecticut River Valley", "Cape Cod", "Cape Cod Bay", "Buzzards Bay", "Block Island", "Marthas Vineyard", "Nantucket", "Narragansett Bay", "Downeast"]
    },
    "Ohio Valley": {
        "parents": ["Eastern US", "Midwest"],
        "states": ["Ohio", "Indiana", "Kentucky", "West Virginia", "Pennsylvania"],
        "features": ["Ohio River", "Ohio River Valley"]
    },
    "Canada Region": {
        "parents": ["Canada"],
        "states": ["Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba", "Saskatchewan"],
        "features": ["St. Lawrence River"]
    },
    "LOCAL_CONTEXT": {
        "aliases": ["CWA", "The CWA", "Overhead", "The Area", "The Region", "Here", "Our Area", "Local Area", "This Area"],
        "parents": [], 
        "subregions": []
    },
    "The Rockies": {
        "aliases": [],
        "parents": ["Western US", "Western CONUS", "The West", "Intermountain West"], 
        "subregions": ["Colorado", "Wyoming", "Montana", "Colorado River", "Utah", "New Mexico"]
    }
}

# ==========================================
# 2. HELPER LISTS
# ==========================================
CARDINAL_DIRECTIONS = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'northern', 'southern', 'eastern', 'western', 'central']
STATE_ABBREVIATION_MAP = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}

# ==========================================
# 3. BUILD THE GRAPH
# ==========================================
lookup_table = {}

def ensure_node(name):
    if name not in lookup_table:
        lookup_table[name] = {"parents": set(), "grandparents": set(), "siblings": set(), "children": set(), "synonyms": {name}, "loose_grouping": False}

def link_parent_child(parent, child):
    ensure_node(parent)
    ensure_node(child)
    lookup_table[child]["parents"].add(parent)
    lookup_table[parent]["children"].add(child)

def link_synonyms(s1, s2):
    ensure_node(s1)
    ensure_node(s2)
    lookup_table[s1]["synonyms"].add(s2)
    lookup_table[s2]["synonyms"].add(s1)
    lookup_table[s1]["siblings"].add(s2)
    lookup_table[s2]["siblings"].add(s1)

def link_neighbors(n1, n2):
    ensure_node(n1)
    ensure_node(n2)
    lookup_table[n1]["siblings"].add(n2)
    lookup_table[n2]["siblings"].add(n1)

# A. Ingest Backbone Hierarchy
for region_name, data in hierarchy.items():
    ensure_node(region_name)
    if data.get("loose_grouping", False):
        lookup_table[region_name]["loose_grouping"] = True
    
    for alias in data.get("aliases", []):
        link_synonyms(region_name, alias)

    for p in data.get("parents", []): link_parent_child(p, region_name)
    children = data.get("states", []) + data.get("features", []) + data.get("subregions", [])
    for c in children:
        link_parent_child(region_name, c)

# B. Expand Cardinal Directions
DIRECTION_PREFIXES = set([d.title() for d in CARDINAL_DIRECTIONS])
existing_places = list(lookup_table.keys())
for place in existing_places:
    if place == "LOCAL_CONTEXT": continue 
    
    place_words = set(place.lower().split())
    if any(d in place_words for d in CARDINAL_DIRECTIONS): continue

    for direction in CARDINAL_DIRECTIONS:
        directional_name = f"{direction.title()} {place}"
        link_parent_child(place, directional_name)

# C. Handle State Abbreviations
for abbr, full_name in STATE_ABBREVIATION_MAP.items():
    if full_name in lookup_table:
        link_synonyms(abbr, full_name)
        for direction in CARDINAL_DIRECTIONS:
            dir_abbr = f"{direction.title()} {abbr}" 
            dir_full = f"{direction.title()} {full_name}" 
            link_parent_child(abbr, dir_abbr)
            link_synonyms(dir_abbr, dir_full) 

# ==========================================
# 4. NEIGHBOR LINKING (TIGHT GROUPS)
# ==========================================
for name, data in lookup_table.items():
    if data["children"] and not data["loose_grouping"]:
        children = list(data["children"])
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                link_neighbors(children[i], children[j])

# ==========================================
# 5. APPLY MANUAL OVERRIDES
# ==========================================
print("Applying Manual Overrides...")
for name, override in MANUAL_OVERRIDES.items():
    ensure_node(name)
    
    if "parents" in override:
        for p in override["parents"]: link_parent_child(p, name)
    if "siblings" in override:
        for s in override["siblings"]: link_neighbors(name, s)
    if "twins" in override:
        for t in override["twins"]: link_synonyms(name, t)

    # REMOVALS
    if "remove_siblings" in override:
        for target in override["remove_siblings"]:
            if target in lookup_table and target in lookup_table[name]["siblings"]:
                lookup_table[name]["siblings"].remove(target)
                lookup_table[target]["siblings"].remove(name)
                print(f"   ✂️ Severed sibling link: {name} <X> {target}")

    if "remove_parents" in override:
        for target in override["remove_parents"]:
            if target in lookup_table and target in lookup_table[name]["parents"]:
                lookup_table[name]["parents"].remove(target)
                lookup_table[target]["children"].remove(name)
                print(f"   ✂️ Severed parent link: {target} <X> {name}")

# ==========================================
# 6. BLACKLIST CLEANUP
# ==========================================
print("Running Garbage Collection (Blacklist)...")
for target in BLACKLIST_NODES:
    if target in lookup_table:
        print(f"   🗑️ Deleting node: {target}")
        
        for p in list(lookup_table[target]["parents"]):
            if p in lookup_table: lookup_table[p]["children"].discard(target)
        for c in list(lookup_table[target]["children"]):
            if c in lookup_table: lookup_table[c]["parents"].discard(target)
        for s in list(lookup_table[target]["siblings"]):
            if s in lookup_table: lookup_table[s]["siblings"].discard(target)
        for s in list(lookup_table[target]["synonyms"]):
            if s in lookup_table: lookup_table[s]["synonyms"].discard(target)

        del lookup_table[target]

# ==========================================
# 7. TWIN MERGE
# ==========================================
print("Unifying Synonym Twins...")
visited_synonyms = set()
all_nodes = list(lookup_table.keys())

for node in all_nodes:
    if node not in lookup_table: continue 
    if node in visited_synonyms: continue
    
    family = {node}
    queue = [node]
    while queue:
        curr = queue.pop(0)
        for syn in lookup_table[curr]["synonyms"]:
            if syn not in family and syn in lookup_table:
                family.add(syn)
                queue.append(syn)
    
    visited_synonyms.update(family)
    
    all_parents = set()
    all_children = set()
    all_siblings = set()
    all_loose = False
    
    for member in family:
        all_parents.update(lookup_table[member]["parents"])
        all_children.update(lookup_table[member]["children"])
        all_siblings.update(lookup_table[member]["siblings"])
        if lookup_table[member]["loose_grouping"]: all_loose = True
            
    for member in family:
        lookup_table[member]["parents"] = all_parents
        lookup_table[member]["children"] = all_children
        lookup_table[member]["siblings"] = all_siblings
        lookup_table[member]["loose_grouping"] = all_loose
        for other_member in family:
            if member != other_member:
                lookup_table[member]["siblings"].add(other_member)

# ==========================================
# 8. PROPAGATE GRANDPARENTS
# ==========================================
print("Propagating ancestry...")
for _ in range(2): 
    for name, data in lookup_table.items():
        for p in list(data["parents"]):
            if p in lookup_table:
                for gp in lookup_table[p]["parents"]:
                    data["grandparents"].add(gp)
                for ggp in lookup_table[p]["grandparents"]:
                    data["grandparents"].add(ggp)

# ==========================================
# 9. EXPORT
# ==========================================
final_json = {}
for k, v in lookup_table.items():
    final_json[k] = {
        "parents": list(v["parents"]),
        "grandparents": list(v["grandparents"]),
        "siblings": list(v["siblings"])
    }

print(f"Generated hierarchy for {len(final_json)} locations.")
with open("location_hierarchy.json", "w") as f:
    json.dump(final_json, f, indent=2)
print("Saved to location_hierarchy.json")