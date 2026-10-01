import pandas as pd
import requests
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'opa1_variants.csv'))

BASE_URL = "https://spliceai-38-xwkwwwxdwq-uc.a.run.app/spliceai/"

def get_spliceai_scores(chrom, pos, ref, alt, distance=500, retries=3):
    url = f"{BASE_URL}?hg=38&distance={distance}&variant={chrom}-{pos}-{ref}-{alt}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=(10, 60))
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if not response.ok:
                logging.warning(f"Bad response {response.status_code} for {chrom}-{pos}-{ref}-{alt}")
                return {}
            data = response.json()
            scores = data.get('scores', [])
            if not scores:
                return {}
            # Prefer the "MS" (most significant) transcript, fall back to first
            ms = next((s for s in scores if s.get('t_priority') == 'MS'), scores[0])
            return ms
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {chrom}-{pos}-{ref}-{alt}: {e}")
            time.sleep(2 ** attempt)
    return {}

def fetch_row(row):
    scores = get_spliceai_scores(row['CHROMO'], row['TARGET_POS_1BASED'], row['REF_BASE'], row['ALT_BASE'])
    return {
        'DBID': row['DBID'],
        'NAME': row['NAME'],
        'cDNA': row['cDNA'],
        'CHROMO': row['CHROMO'],
        'TARGET_POS_1BASED': row['TARGET_POS_1BASED'],
        'REF_BASE': row['REF_BASE'],
        'ALT_BASE': row['ALT_BASE'],
        'Gene': scores.get('g_name'),
        'Transcript': scores.get('t_id'),
        'RefSeq': ', '.join(scores.get('t_refseq_ids') or []),
        'DS_AG': scores.get('DS_AG'),
        'DS_AL': scores.get('DS_AL'),
        'DS_DG': scores.get('DS_DG'),
        'DS_DL': scores.get('DS_DL'),
        'DP_AG': scores.get('DP_AG'),
        'DP_AL': scores.get('DP_AL'),
        'DP_DG': scores.get('DP_DG'),
        'DP_DL': scores.get('DP_DL'),
    }

df = pd.read_csv(file_path)
rows = [row for _, row in df.iterrows()]
final_results = []

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(fetch_row, row): row for row in rows}
    for future in tqdm(as_completed(futures), total=len(rows), desc="Fetching SpliceAI scores"):
        try:
            final_results.append(future.result())
        except Exception as e:
            logging.warning(f"Row failed: {e}")

output_df = pd.DataFrame(final_results)
for col in ['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL']:
    output_df[col] = pd.to_numeric(output_df[col], errors='coerce')

output_df['Max_DS'] = output_df[['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL']].max(axis=1)
output_df.to_csv('opa1_spliceai.csv', index=False)
logging.info(f"Done! Saved {len(output_df)} rows.")