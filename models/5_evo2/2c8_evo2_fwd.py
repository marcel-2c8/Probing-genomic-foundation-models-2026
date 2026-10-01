### Evo2 (40B) Causal (Autoregressive) Inference — Forward Pass Only ###
    
import requests
import numpy as np
import pandas as pd
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
#CONTEXT       = "512"     ## TAX_PREFIX is 87, so adjust accordingly. Adjust also lines 66 and 109.

CONTEXT       = "425"

evo2_headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

BASE_TO_IDX = {'A': 65, 'C': 67, 'T': 84, 'G': 71}

# ── 2. Helpers ────────────────────────────────────────────────────────────────
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

def get_evo2_causal_scores(prefix_seq: str, ref_base: str, alt_base: str,
                            label: str = ""):
    """
    Sends prefix to Evo2 with enable_logits=True, generates 1 token,
    and reads the logit at the final position for ref and alt bases.
    """

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
context_window = int(CONTEXT) #- len(TAX_PREFIX)         ## Adjust accordingly.

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

        # Fetch prefix: the context_window bases immediately BEFORE the variant
        start      = max(0, (pos - 1) - context_window)
        prefix_seq = fetch_ucsc_sequence(chrom, start, pos - 1)
        if prefix_seq is None:
            print(f"  [SKIP] {label}: UCSC fetch failed for prefix window")
            continue

        # Score
        ref_ll, alt_ll = get_evo2_causal_scores(prefix_seq, ref, alt, label=label)

        if ref_ll is None or alt_ll is None:
            print(f"  [MISSING DATA] {label}: logprobs not found for "
                  f"{ref}/{alt}. Skipping.")
            continue

        results.append({
            'DBID':          label,
            'CHROM':         chrom,
            'TARGET_POS':    pos,
            'REF_BASE':      ref,
            'ALT_BASE':      alt,
            'Causal_Ref_LL': ref_ll,
            'Causal_Alt_LL': alt_ll,
            'LLR_Score':     alt_ll - ref_ll
        })

    except Exception as e:
        print(f"  [ERROR] index {index} (pos {current_pos}): {e}")

# ── 4. Final report ───────────────────────────────────────────────────────────
if results:
    final_df = pd.merge(df, pd.DataFrame(results), on='DBID')
    final_df.to_csv(os.path.join(OUTPUT_DIR, "opa1_evo2_fwd.csv"), index=False)
    print("\n--- OPA1 Analysis Table ---")
    print(final_df[['DBID', 'CHROM', 'Causal_Ref_LL', 'Causal_Alt_LL', 'LLR_Score']])
else:
    print("No results to save.")