### ALPHAGENOME - JUNCTION SCORER ###
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

# ── 1. Setup Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("opa1_junction_batch.log"),
        logging.StreamHandler()
    ]
)

# ── 2. UCSC sequence fetcher ──────────────────────────────────────────────────
UCSC_ENDPOINT = "https://api.genome.ucsc.edu/getData/sequence"
GENOME        = "hg38"

def fetch_ucsc_sequence(chrom: str, start_0based: int, end_1based: int,
                        genome_build: str = GENOME, retries: int = 3) -> str | None:
    """
    Fetch reference sequence from UCSC REST API.
    Coordinates: start 0-based, end 1-based (BED convention, same as pysam.fetch).
    """
    params = {
        "genome": genome_build,
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

    model = dna_client.create(API_KEY)

    DATA_DIR    = os.path.expanduser('./data')
    CSV_PATH    = os.path.join(DATA_DIR, 'opa1_variants.csv')
    OUTPUT_PATH = os.path.expanduser('./outputs')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    file_name   = "opa1_alphagenome_junctionscorer"

    INTERVAL_LENGTH = 131072
    HALF_INTERVAL   = INTERVAL_LENGTH // 2
    SIG_THRESHOLD   = 0.1

    # ── 4. Load CSV ───────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        logging.error(f"Input file not found: {CSV_PATH}")
        return

    df_input = pd.read_csv(CSV_PATH)
    logging.info(f"Processing {len(df_input)} variants for Junction analysis...")

    all_results = []

    # ── 5. Batch Loop ─────────────────────────────────────────────────────────
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
            logging.warning(f"{dbid}: REF mismatch at {chrom}:{pos_1based}. "
                            f"CSV:{ref_csv} vs UCSC:{actual_ref}. Skipping.")
            continue

        # --- Prediction ---
        target_interval = genome.Interval(
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

        variant_junctions = []
        try:
            response = model.predict_variant(
                interval=target_interval,
                variant=variant,
                requested_outputs=[dna_client.OutputType.SPLICE_JUNCTIONS],
                ontology_terms=[]
            )

            ref_obj       = response.reference.splice_junctions
            alt_obj       = response.alternate.splice_junctions
            junction_defs = ref_obj.junctions
            deltas        = alt_obj.values - ref_obj.values

            for i in range(len(junction_defs)):
                j        = junction_defs[i]
                peak_idx = np.argmax(np.abs(deltas[i]))
                max_delta = deltas[i][peak_idx]

                if abs(max_delta) >= SIG_THRESHOLD:
                    dist = min(abs(j.start - target_pos), abs(j.end - target_pos))
                    variant_junctions.append({
                        "Delta": float(max_delta),
                        "Dist":  int(dist),
                        "Coord": f"{j.start}-{j.end}"
                    })

            variant_junctions.sort(key=lambda x: x['Dist'])

            row_data = {
                "DBID":                   dbid,
                "CHROM":                  chrom,
                "Position_1Based":        pos_1based,
                "Ref_Alt":                f"{ref_csv}>{alt_csv}",
                "ClinicalClassification": row.get('ClinicalClassification', 'Unknown'),
                "Total_Sig_Junctions":    len(variant_junctions)
            }

            for rank in range(1, 4):
                if len(variant_junctions) >= rank:
                    item = variant_junctions[rank - 1]
                    row_data[f"J{rank}_Dist"]  = item['Dist']
                    row_data[f"J{rank}_Delta"] = item['Delta']
                    row_data[f"J{rank}_Coord"] = item['Coord']
                else:
                    row_data[f"J{rank}_Dist"]  = np.nan
                    row_data[f"J{rank}_Delta"] = np.nan
                    row_data[f"J{rank}_Coord"] = None

            all_results.append(row_data)

        except Exception as e:
            logging.warning(f"API Failure for {dbid}: {e}")
            all_results.append({
                "DBID":                dbid,
                "CHROM":               chrom,
                "Position_1Based":     pos_1based,
                "Ref_Alt":             f"{ref_csv}>{alt_csv}",
                "Total_Sig_Junctions": np.nan
            })

    # ── 6. Export ─────────────────────────────────────────────────────────────
    if all_results:
        df_final = pd.DataFrame(all_results)
        cols = [
            "DBID", "CHROM", "Position_1Based", "Ref_Alt",
            "ClinicalClassification", "Total_Sig_Junctions",
            "J1_Dist", "J1_Delta", "J1_Coord",
            "J2_Dist", "J2_Delta", "J2_Coord",
            "J3_Dist", "J3_Delta", "J3_Coord"
        ]
        df_final = df_final.reindex(columns=cols)
        df_final.to_csv(    join(OUTPUT_PATH, f"{file_name}.csv"),     index=False)
        df_final.to_parquet(join(OUTPUT_PATH, f"{file_name}.parquet"), index=False)
        logging.info(f"Successfully exported {len(df_final)} variants.")

if __name__ == "__main__":
    main()