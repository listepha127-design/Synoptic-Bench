import os
import requests
from datetime import datetime, timedelta, timezone
import re
from dateutil import parser
from dateutil.tz import gettz
import json 


NWS_TZINFOS = {
    'CDT': gettz("America/Chicago"),
    'CST': gettz("America/Chicago"),
    'MDT': gettz("America/Denver"),
    'MST': gettz("America/Denver"),
    'EDT': gettz("America/New_York"),
    'EST': gettz("America/New_York"),
    'PDT': gettz("America/Los_Angeles"),
    'PST': gettz("America/Los_Angeles"),
    'Z': timezone.utc
}

def extract_afd_timestamp(afd_text, nominal_date):
    """
    Extracts the primary issue timestamp from an AFD text block.
    Prioritizes human-readable full date timestamps by reconstructing a standard HH:MM format.
    Only falls back to PIL (DDHHMM) if no human-readable timestamp is found.
    Returns a datetime object if successfully extracted, otherwise None.
    NOTE: The nominal_date is primarily for parsing PIL, and for debugging context.
          The returned timestamp is the product's *actual* internal timestamp.
    """
    lines = afd_text.splitlines()

    print(f"\n--- extract_afd_timestamp called for nominal_date: {nominal_date} ---")
    print(f"Product Text Snippet (first 200 chars): \n'''{afd_text[:200]}'''")

    human_readable_pattern_found = False

    human_ts_pattern = re.compile(
        r'(\d{1,2})'                       # Hour (1 or 2 digits) - Group 1
        r'(?:(\d{2}))?'                    # Optional minutes (2 digits) - Group 2
        r'\s*(AM|PM|Z)?\s+'                # Optional AM/PM/Z and spaces - Group 3
        r'([A-Z]{2,5})\s+'                 # Timezone (e.g., CDT, MST, Z) - Group 4
        r'([A-Z][a-z]{2,8})\s+'            # Day of week (e.g., Fri) - Group 5
        r'([A-Z][a-z]{2,8})\s+'            # Month (e.g., Jul) - Group 6
        r'(\d{1,2})\s+'                    # Day of month - Group 7
        r'(\d{4})'                         # Year - Group 8
    )
    
    for line_num, line in enumerate(lines[:15]):
        match = human_ts_pattern.search(line)
        if match:
            human_readable_pattern_found = True
            
            hour_str, min_str, ampm_z, tz_str, _, month_str, day_of_month_str, year_str = match.groups()

            reconstructed_time_str = hour_str
            if min_str:
                reconstructed_time_str += f":{min_str}"
            if ampm_z:
                reconstructed_time_str += f" {ampm_z}"
            
            timestamp_string_for_parser = (
                f"{reconstructed_time_str} {tz_str} "
                f"{month_str} {day_of_month_str} {year_str}"
            )
            
            timestamp_string_for_parser = timestamp_string_for_parser.replace('Z ', 'UTC ').strip()
            
            try:
                dt_object = parser.parse(timestamp_string_for_parser, tzinfos=NWS_TZINFOS)
                
                print(f"  Human TS Pattern Found on line {line_num+1}: '{line.strip()}'")
                print(f"  Attempting to parse reconstructed human timestamp: '{timestamp_string_for_parser}'")

                if dt_object.tzinfo is not None and dt_object.tzinfo.utcoffset(dt_object) is not None:
                    extracted_dt_utc = dt_object.astimezone(timezone.utc)
                else:
                    print(f"  Human TS is Naive AFTER tzinfos. Skipping this timestamp.")
                    continue 

                print(f"  Human TS Parsed (UTC): {extracted_dt_utc}")
                print(f"  Successfully extracted human timestamp: {extracted_dt_utc}. Returning this.")
                return extracted_dt_utc 

            except Exception as e:
                print(f"  Human TS Parsing Error for '{timestamp_string_for_parser}': {e}. Skipping this timestamp.")
                continue
    
    if not human_readable_pattern_found:
        pil_pattern = re.compile(r'FXUS\d{2}\s+[A-Z]{4}\s+(\d{6})')
        
        for line_num, line in enumerate(lines[:3]):
            match = pil_pattern.search(line)
            if match:
                ddhhmm = match.group(1)
                try:
                    day_of_month = int(ddhhmm[:2])
                    hour = int(ddhhmm[2:4])
                    minute = int(ddhhmm[4:6])

                    extracted_dt_z = datetime(
                        nominal_date.year,
                        nominal_date.month,
                        day_of_month,
                        hour,
                        minute,
                        tzinfo=timezone.utc
                    )
                    
                    print(f"  PIL Found on line {line_num+1}: '{line.strip()}' -> DDHHMM: {ddhhmm}")
                    print(f"  PIL Parsed (assuming nominal month/year): {extracted_dt_z}")
                    
                    print(f"  Successfully extracted PIL timestamp: {extracted_dt_z}. Returning this.")
                    return extracted_dt_z
                except (ValueError, IndexError) as e:
                    print(f"  PIL Parsing Error: {e}. Skipping this PIL timestamp.")
                    continue

    print(f"--- No valid timestamp could be extracted from product. Returning None. ---")
    return None

def download_afd(pil, start_date, end_date, output_dir="/scratch/hay3fm/afd_data"):
    """
    Downloads AFD data for a given PIL and date range using the Mesonet API.
    Iterates through dates, lists product IDs, and downloads individual products.
    """
    os.makedirs(output_dir, exist_ok=True)

    seen_product_identifiers = set() 

    current_date = start_date
    while current_date <= end_date:
        api_list_date_str = current_date.strftime("%Y-%m-%d") 
        list_api_url = (
            f"https://mesonet.agron.iastate.edu/api/1/nws/afos/list.json?"
            f"pil={pil}&date={api_list_date_str}"
        )
        
        print(f"\n--- Processing Date: {api_list_date_str} for PIL: {pil} ---")
        print(f"Attempting to list products from: {list_api_url}")
        
        try:
            list_response = requests.get(list_api_url)
            list_response.raise_for_status() 
            
            product_metadata = list_response.json()
            
            # --- CRITICAL CHANGE HERE: Check for 'data' key instead of 'products' ---
            if not product_metadata or 'data' not in product_metadata or not product_metadata['data']:
                print(f"  No products found for {pil} on {api_list_date_str} from listing API. Skipping.")
                current_date += timedelta(days=1)
                continue

            products_downloaded_for_this_day = 0
            
            # --- CRITICAL CHANGE HERE: Iterate through 'data' list ---
            for product_info in product_metadata['data']: 
                product_id = product_info.get('product_id')
                if not product_id:
                    print(f"  Skipping product with missing 'product_id' in metadata: {product_info}")
                    continue

                if product_id in seen_product_identifiers:
                    print(f"  Skipping duplicate product_id: {product_id}")
                    continue
                
                product_text_url = f"https://mesonet.agron.iastate.edu/api/1/nwstext/{product_id}"
                print(f"  Attempting to download product text from: {product_text_url}")
                
                try:
                    text_response = requests.get(product_text_url)
                    text_response.raise_for_status()
                    
                    product_text = text_response.content.decode('latin-1', errors='ignore').strip()

                    if not product_text:
                        print(f"  Downloaded empty text for {product_id}. Skipping.")
                        continue

                    issued_dt_utc = extract_afd_timestamp(product_text, current_date.date())

                    if issued_dt_utc:
                        timestamp_for_filename = issued_dt_utc.strftime("%Y%m%d_%H%M%S")
                        # Using product_id parts for even more unique filename
                        # Format: PIL_YYYYMMDD_HHMMSS_PIL_YYYYMMDDHHMM.txt
                        unique_filename = f"{pil}_{timestamp_for_filename}_{product_id.split('-')[-1]}_{product_id.split('-')[0]}.txt" 
                        file_path = os.path.join(output_dir, unique_filename)
                        
                        seen_product_identifiers.add(product_id)

                        print(f"  Saving to: {unique_filename}")
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(product_text)
                        products_downloaded_for_this_day += 1
                    else:
                        print(f"  Could not extract valid timestamp for product_id: {product_id}. Skipping file save.")

                except requests.exceptions.RequestException as e:
                    print(f"  Error downloading product text for {product_id}: {e}. Skipping this product.")
                except json.JSONDecodeError as e:
                    print(f"  Error decoding JSON from product text URL (unexpected JSON for raw text): {e}. Skipping this product.")

            if products_downloaded_for_this_day > 0:
                print(f"  Successfully downloaded and saved {products_downloaded_for_this_day} new files for {pil} on {api_list_date_str}.")
            else:
                print(f"  No new files downloaded for {pil} on {api_list_date_str} (either no products, parse failed, or all were duplicates).")

        except requests.exceptions.RequestException as e:
            print(f"  Error accessing listing API {list_api_url}: {e}. Skipping this date.")
        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON from listing API response for {api_list_date_str}: {e}. Skipping this date.")
        
        current_date += timedelta(days=1)
        
# Example usage
#afd_pils = ['AFDABR', 'AFDALY', 'AFDABQ', 'AFDAMA', 'AFDEWX', 'AFDLWX', 'AFDBYZ','AFDBGM', 'AFDBMX', 'AFDBIS', 'AFDRNK','AFDBOI', 'AFDBOX', 'AFDBRO', 'AFDBUF','AFDBTV', 'AFDCAR', 'AFDCHS', 'AFDRLX','AFDCYS', 'AFDLOT', 'AFDCLE', 'AFDCAE','AFDCRP', 'AFDFWD', 'AFDBOU', 'AFDDMX','AFDDTX', 'AFDDDC', 'AFDDLH', 'AFDLKN','AFDEPZ', 'AFDEKA', 'AFDPAFG', 'AFDFGZ','AFDAPX', 'AFDGGW', 'AFDGLD', 'AFDFGF','AFDGJT', 'AFDGRR', 'AFDGYX', 'AFDTFX','AFDGRB', 'AFDGSP', 'AFDGID', 'AFDPHFO','AFDHGX', 'AFDHUN', 'AFDIND', 'AFDJAN','AFDJKL', 'AFDJAX', 'AFDPAJK', 'AFDEAX','AFDKEY', 'AFDARX', 'AFDLCH', 'AFDVEF','AFDILX', 'AFDLZK', 'AFDLOX', 'AFDLMK','AFDLUB', 'AFDMQT', 'AFDMFR', 'AFDMLB','AFDMEG', 'AFDMFL', 'AFDMAF', 'AFDMKX','AFDMSO', 'AFDMOB', 'AFDMRX', 'AFDPHI','AFDOHX', 'AFDLIX', 'AFDMHX', 'AFDOKX','AFDOUN', 'AFDIWX', 'AFDLBF', 'AFDOAX','AFDPAH', 'AFDFFC', 'AFDPDT', 'AFDPSR','AFDPBZ', 'AFDPIH', 'AFDPQR', 'AFDPUB','AFDDVN', 'AFDRAH', 'AFDUNR', 'AFDREV','AFDRIW', 'AFDSTO', 'AFDSLC', 'AFDSJT','AFDSGX', 'AFDMTR', 'AFDHNX', 'AFDTJSJ','AFDSEW', 'AFDSHV', 'AFDFSD', 'AFDOTX','AFDSGF', 'AFDCTP', 'AFDLSX', 'AFDTAE','AFDTBW', 'AFDTOP', 'AFDTWC', 'AFDTSA','AFDMPX', 'AFDAKQ', 'AFDICT', 'AFDILM','AFDILN']
afd_pils = ['AFDBUF','AFDBTV', 'AFDCAR']

start_date = datetime(2019, 6, 12) # Example: Start a few days earlier
end_date = datetime(2025, 11, 30)   # Example: End today

for pil in afd_pils:
     download_afd(pil, start_date, end_date)