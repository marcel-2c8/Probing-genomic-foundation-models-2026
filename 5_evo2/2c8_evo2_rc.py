### Evo2 (40B) focused on human sequences ###
    ### Causal (Autoregressive) Inference — Reverse Complement Pass Only
    ### TEST SET ###
    # NO HUMAN IDENTIFIER
    # 06262026
import requests
import numpy as np
import pandas as pd
import json
import time
import os
from tqdm import tqdm
from google.colab import userdata
from google.colab import files

# ── 1. Configuration ──────────────────────────────────────────────────────────
API_KEY    = userdata.get('evo2_api_key')
DATA_DIR   = './data'
CSV_PATH   = os.path.join(DATA_DIR, 'opa1_variants.csv')
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVO2_ENDPOINT = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"
UCSC_ENDPOINT = "https://api.genome.ucsc.edu/getData/sequence"
GENOME        = "hg38"

#TAX_PREFIX    = "|D__EUKARYOTA;P__CHORDATA;C__MAMMALIA;O__PRIMATES;F__HOMINIDAE;G__HOMO;S__HOMO_SAPIENS|"
#CONTEXT       = "512"     ## TAX_PREFIX is 87, so adjust accordingly. Adjust also lines 94 and 137.

CONTEXT       = "425"

evo2_headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

BASE_TO_IDX = {'A': 65, 'C': 67, 'T': 84, 'G': 71}
COMPLEMENT  = str.maketrans('ACGT', 'TGCA')

# ── 2. Helpers ────────────────────────────────────────────────────────────────
def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def complement_base(base: str) -> str:
    return base.translate(COMPLEMENT)

def fetch_ucsc_sequence(chrom: str, start_0based: int, end_1based: int,
                        genome: str = GENOME, retries: int = 3) -> str | None:
    """
    Fetch reference sequence from UCSC REST API.
    Coordinates: start 0-based, end 1-based (BED convention, same as pysam.fetch).
    """
    params = {
        "genome": genome,
        "chrom":  chrom,
        "start":  start_0based,
        "end":    end_1based
    }
    for attempt in range(retries):
        try:
            resp = requests.get(UCSC_ENDPOINT, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("dna", "").upper()
            print(f"  [UCSC] HTTP {resp.status_code} for {chrom}:{start_0based}-{end_1based}")
        except Exception as e:
            print(f"  [UCSC] Attempt {attempt+1} failed: {e}")
        time.sleep(1.5 * (attempt + 1))
    return None

def fetch_chrom_length(chrom: str, genome: str = GENOME) -> int | None:
    """
    Fetch chromosome length from UCSC REST API (replaces
    fasta.get_reference_length).
    """
    try:
        resp = requests.get(
            "https://api.genome.ucsc.edu/getData/track",
            params={"genome": genome, "track": "chromInfo", "chrom": chrom},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            # Response is a list of dicts with 'chrom' and 'size' keys
            for item in data.get("chromInfo", []):
                if item.get("chrom") == chrom:
                    return int(item["size"])
        print(f"  [UCSC] Could not fetch length for {chrom}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [UCSC] chrom length fetch failed: {e}")
    return None

def get_evo2_causal_scores(prefix_seq: str, ref_base: str, alt_base: str,
                            label: str = ""):
    #full_seq = TAX_PREFIX + prefix_seq    ## Add taxonomic prefix if needed
    full_seq = prefix_seq
    payload  = {
        "sequence":      full_seq,
        "num_tokens":    1,
        "top_k":         1,
        "enable_logits": True
    }
    try:
        time.sleep(1.0)
        response = requests.post(EVO2_ENDPOINT, headers=evo2_headers,
                                 json=payload, timeout=60)
        if response.status_code != 200:
            print(f"  [{label}] API Error {response.status_code}: {response.text}")
            return None, None

        data   = response.json()
        logits = data.get("logits")
        if logits is None:
            print(f"  [{label}] No logits in response. Keys: {list(data.keys())}")
            return None, None

        last_logits = logits[-1]
        ref_idx     = BASE_TO_IDX.get(ref_base)
        alt_idx     = BASE_TO_IDX.get(alt_base)

        if ref_idx is None or alt_idx is None:
            print(f"  [{label}] Unknown base: {ref_base} or {alt_base}")
            return None, None

        log_probs = (last_logits
                     - np.log(np.sum(np.exp(last_logits - np.max(last_logits))))
                     - np.max(last_logits))

        return float(log_probs[ref_idx]), float(log_probs[alt_idx])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None

# ── 3. Main loop ──────────────────────────────────────────────────────────────
df             = pd.read_csv(CSV_PATH)
context_window = int(CONTEXT) #- len(TAX_PREFIX)    ## Adjust accordingly.

# Cache chrom lengths to avoid a redundant API call per row
chrom_length_cache: dict[str, int] = {}

results = []

for index, row in tqdm(df.iterrows(), total=len(df)):
    current_pos = row.get('TARGET_POS_1BASED', 'UNKNOWN')

    try:
        pos   = int(current_pos)
        ref   = row['REF_BASE'].strip().upper()
        alt   = row['ALT_BASE'].strip().upper()
        label = row['DBID']
        chrom = row['CHROMO'].strip()          # ← read chromosome from CSV

        # Validate ref base
        actual_ref = fetch_ucsc_sequence(chrom, pos - 1, pos)
        if actual_ref is None:
            print(f"  [SKIP] {label}: UCSC fetch failed for ref base")
            continue
        if actual_ref != ref:
            print(f"  [SKIP] {label}: CSV says {ref}, genome says {actual_ref} "
                  f"at {chrom}:{pos}")
            continue

        # Get chrom length (cached)
        if chrom not in chrom_length_cache:
            length = fetch_chrom_length(chrom)
            if length is None:
                print(f"  [SKIP] {label}: could not fetch length for {chrom}")
                continue
            chrom_length_cache[chrom] = length
        chrom_len = chrom_length_cache[chrom]

        # Build RC prefix: downstream sequence, reverse-complemented
        rc_end        = min(chrom_len, pos + context_window)
        downstream    = fetch_ucsc_sequence(chrom, pos, rc_end)   # 0-based start = pos (the base AFTER the variant)
        if downstream is None:
            print(f"  [SKIP] {label}: UCSC fetch failed for downstream window")
            continue

        rc_prefix = reverse_complement(downstream)
        rc_ref    = complement_base(ref)
        rc_alt    = complement_base(alt)

        ref_ll, alt_ll = get_evo2_causal_scores(rc_prefix, rc_ref, rc_alt, label=label)

        if ref_ll is None or alt_ll is None:
            print(f"  [MISSING DATA] {label}: logprobs not found for "
                  f"{rc_ref}/{rc_alt}. Skipping.")
            continue

        results.append({
            'DBID': label,
            'CHROM': chrom,
            'TARGET_POS_1BASED': pos,
            'REF_BASE': ref,
            'ALT_BASE': alt,
            'RC_Ref_LL': ref_ll,
            'RC_Alt_LL': alt_ll,
            'RC_LLR_Score': alt_ll - ref_ll
        })

    except Exception as e:
        print(f"  [ERROR] index {index} (pos {current_pos}): {e}")

# ── 4. Final report ───────────────────────────────────────────────────────────
if results:
    final_df = pd.merge(df, pd.DataFrame(results), on='DBID')
    final_df.to_csv(os.path.join(OUTPUT_DIR, "opa1_evo2_rc.csv"), index=False)
    print("\n--- OPA1 RC Analysis Table ---")
    print(final_df[['DBID', 'CHROM', 'RC_Ref_LL', 'RC_Alt_LL', 'RC_LLR_Score']])
else:
    print("No results to save.")