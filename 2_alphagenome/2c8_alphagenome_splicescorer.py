### ALPHAGENOME - SPLICE SCORER ###
import os
import requests
import pandas as pd
import numpy as np
import logging
import time
from tqdm import tqdm
from google.colab import userdata
from os.path import join

from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models.variant_scorers import GeneMaskSplicingScorer

# ── 1. Setup Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)

# ── 2. UCSC sequence fetcher ──────────────────────────────────────────────────
UCSC_ENDPOINT = "https://api.genome.ucsc.edu/getData/sequence"
GENOME        = "hg38"

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
            logging.warning(f"[UCSC] HTTP {resp.status_code} for "
                            f"{chrom}:{start_0based}-{end_1based}")
        except Exception as e:
            logging.warning(f"[UCSC] Attempt {attempt+1} failed: {e}")
        time.sleep(1.5 * (attempt + 1))
    return None

def main():
    # ── 3. Configuration & Paths ──────────────────────────────────────────────
    API_KEY = userdata.get('api_key')
    if not API_KEY:
        logging.error("API_KEY environment variable not set.")
        return

    DATA_DIR    = os.path.expanduser('./data')
    CSV_PATH    = os.path.join(DATA_DIR, 'opa1_variants.csv')
    OUTPUT_PATH = os.path.expanduser('./outputs')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    file_name   = "opa1_alphagenome_splicescorer"

    # ── 4. Initialize Model and Scorer ────────────────────────────────────────
    model = dna_client.create(API_KEY)

    splicing_scorer = GeneMaskSplicingScorer(
        requested_output=dna_client.OutputType.SPLICE_SITES,
        width=1001
    )

    RADIUS          = 500           # Lookup window
    INTERVAL_LENGTH = 1048576       # Genomic context
    HALF_INTERVAL   = INTERVAL_LENGTH // 2
    TRACK_NAMES     = ["Acceptor", "Donor", "ExonBody", "Intron"]

    # ── 5. Load CSV ───────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        logging.error(f"Input file not found: {CSV_PATH}")
        return

    df_input = pd.read_csv(CSV_PATH)
    logging.info(f"Processing {len(df_input)} variants from OPA1 gold standard...")

    all_results = []

    # ── 6. Batch Loop ─────────────────────────────────────────────────────────
    for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
        dbid       = row['DBID']
        pos_1based = row['TARGET_POS_1BASED']
        ref_csv    = str(row['REF_BASE']).upper()
        alt_csv    = str(row['ALT_BASE']).upper()
        chrom      = row['CHROMO'].strip()         # ← read chromosome from CSV

        target_pos = pos_1based - 1                # convert to 0-based

        # --- Reference Validation ---
        actual_ref = fetch_ucsc_sequence(chrom, target_pos, target_pos + 1)
        if actual_ref is None:
            logging.error(f"{dbid}: UCSC fetch failed for ref base at "
                          f"{chrom}:{pos_1based}. Skipping.")
            continue
        if actual_ref != ref_csv:
            logging.error(f"{dbid}: REF mismatch at {chrom}:{pos_1based}. "
                          f"CSV:{ref_csv} vs UCSC:{actual_ref}. Skipping.")
            continue

        # --- Prediction ---
        opa1_interval = genome.Interval(
            chromosome=chrom,                      # ← dynamic chromosome
            start=target_pos - HALF_INTERVAL,
            end=target_pos + HALF_INTERVAL
        )

        variant = genome.Variant(
            chromosome=chrom,                      # ← dynamic chromosome
            position=target_pos,
            reference_bases=ref_csv,
            alternate_bases=alt_csv
        )

        try:
            response = model.predict_variant(
                interval=opa1_interval,
                variant=variant,
                requested_outputs=[dna_client.OutputType.SPLICE_SITES],
                ontology_terms=[]
            )

            ref_vals     = response.reference.splice_sites.values
            alt_vals     = response.alternate.splice_sites.values
            delta_matrix = alt_vals - ref_vals

            center_idx  = len(ref_vals) // 2
            local_slice = slice(center_idx - RADIUS, center_idx + RADIUS)

            for t_idx, name in enumerate(TRACK_NAMES):
                l_deltas = delta_matrix[local_slice, t_idx]
                l_ref    = ref_vals[local_slice, t_idx]
                l_alt    = alt_vals[local_slice, t_idx]

                loss_idx = np.argmin(l_deltas)
                gain_idx = np.argmax(l_deltas)

                all_results.append({
                    "DBID":                   dbid,
                    "CHROM":                  chrom,
                    "Position_1Based":        pos_1based,
                    "Ref_Alt":                f"{ref_csv}>{alt_csv}",
                    "ClinicalClassification": row.get('ClinicalClassification', 'Unknown'),
                    "Track":                  name,
                    "Max_Loss_Delta":         float(l_deltas[loss_idx]),
                    "Loss_Ref_Prob":          float(l_ref[loss_idx]),
                    "Loss_Alt_Prob":          float(l_alt[loss_idx]),
                    "Loss_Dist":              int(loss_idx - RADIUS),
                    "Max_Gain_Delta":         float(l_deltas[gain_idx]),
                    "Gain_Ref_Prob":          float(l_ref[gain_idx]),
                    "Gain_Alt_Prob":          float(l_alt[gain_idx]),
                    "Gain_Dist":              int(gain_idx - RADIUS)
                })

        except Exception as e:
            logging.warning(f"API Failure for {dbid}: {e}")

    # ── 7. Export ─────────────────────────────────────────────────────────────
    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_csv(    join(OUTPUT_PATH, f"{file_name}.csv"),     index=False)
        df_final.to_parquet(join(OUTPUT_PATH, f"{file_name}.parquet"), index=False)
        logging.info(f"Successfully exported to {csv_path}")
        print(df_final.to_string(index=False))
    else:
        logging.error("No data processed successfully.")

if __name__ == "__main__":
    main()