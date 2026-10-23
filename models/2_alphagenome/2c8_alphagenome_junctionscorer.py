### JUNCTION SCORER ###
"""
AlphaGenome splice-junction scoring for a list of variants (Colab).

Coordinate conventions (verified against alphagenome genome.py):
  - genome.Interval.start / .end : 0-based, half-open (BED-style)
  - genome.Variant.position      : 1-based (VCF-style)
  - UCSC getData/sequence        : start 0-based, end 1-based (BED-style)

The interval is centred on the true variant base (0-based `target_pos`). The
Variant itself gets the 1-based position. Junction coordinates (j.start, j.end)
are 0-based, so distances are computed against `target_pos`.

Junction effect size is the natural-log fold change between alternate and
reference predicted counts, ln((alt + EPS) / (ref + EPS)), taken at the track
(tissue) with the largest absolute value. No significance threshold is applied
here: every junction is scored, and the N_TOP_JUNCTIONS with the largest
|ln fold change| are reported per variant (J1 = largest), together with their
distance to the variant, coordinates and ref/alt counts at the peak track.
Thresholds are chosen downstream.

Input CSV columns: DBID, CHROMO, TARGET_POS_1BASED, REF_BASE, ALT_BASE
                   (optional: ClinicalClassification)
"""
import logging
import os
import time
from os.path import join

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from google.colab import userdata

from alphagenome.data import genome
from alphagenome.models import dna_client

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("process.log"), logging.StreamHandler()],
    force=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
UCSC_ENDPOINT = "https://api.genome.ucsc.edu/getData/sequence"
ASSEMBLY = "hg38"

DATA_DIR = "./data"
CSV_PATH = join(DATA_DIR, "variants.csv")
OUTPUT_PATH = "./outputs"
FILE_NAME = "alphagenome_junctionscorer_results_ln"

INTERVAL_LENGTH = 131072          # model context length (must stay fixed)
HALF_INTERVAL = INTERVAL_LENGTH // 2

# Pseudocount added to ref and alt counts before the ratio, to avoid log(0).
# At 1e-8 it barely damps near-zero junctions, so use the reported ref/alt
# counts to filter low-count junctions downstream.
EPS = 1e-8

N_TOP_JUNCTIONS = 3                # junctions with largest |ln fold change| reported

# API rate limiting (quota is per minute, in megabytes of prediction output)
REQUEST_DELAY = 1.0                # seconds to pause after every prediction
MAX_RETRIES = 5                    # retries when the quota is exceeded
BACKOFF_BASE = 30                  # seconds; wait = BACKOFF_BASE * attempt number


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_ucsc_sequence(chrom, start_0based, end_1based,
                        assembly=ASSEMBLY, retries=3):
    """Fetch reference sequence from the UCSC REST API (BED-style coordinates)."""
    params = {"genome": assembly, "chrom": chrom,
              "start": start_0based, "end": end_1based}
    for attempt in range(retries):
        try:
            resp = requests.get(UCSC_ENDPOINT, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("dna", "").upper()
            logging.warning(f"[UCSC] HTTP {resp.status_code} for "
                            f"{chrom}:{start_0based}-{end_1based}")
        except Exception as e:
            logging.warning(f"[UCSC] Attempt {attempt + 1} failed: {e}")
        time.sleep(1.5 * (attempt + 1))
    return None


def get_api_key():
    try:
        key = userdata.get("api_key")
    except Exception as e:  # Colab raises if the secret is missing/not shared
        logging.error(f"Could not read Colab secret 'api_key': {e}")
        return None
    return key or None


def predict_with_retry(model, **kwargs):
    """predict_variant with a fixed pause afterwards and backoff on quota errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.predict_variant(**kwargs)
            time.sleep(REQUEST_DELAY)
            return response
        except Exception as e:
            msg = str(e)
            is_quota = "RESOURCE_EXHAUSTED" in msg or "Quota exceeded" in msg
            if not is_quota or attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * attempt
            logging.warning(f"Quota exceeded (attempt {attempt}/{MAX_RETRIES}). "
                            f"Waiting {wait}s before retrying...")
            time.sleep(wait)


def ln_fold_change(ref_vals, alt_vals, eps=EPS):
    """Element-wise ln((alt + eps) / (ref + eps)); counts clipped at 0."""
    ref = np.clip(np.asarray(ref_vals, dtype=float), 0, None)
    alt = np.clip(np.asarray(alt_vals, dtype=float), 0, None)
    return np.log((alt + eps) / (ref + eps))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = get_api_key()
    if not api_key:
        logging.error("API key not available. Aborting.")
        return

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        logging.error(f"Input file not found: {CSV_PATH}")
        return

    model = dna_client.create(api_key)

    df_input = pd.read_csv(CSV_PATH)
    logging.info(f"Processing {len(df_input)} variants from {CSV_PATH}...")

    all_results = []
    skipped = []                # (DBID, reason)
    n_scored = 0

    for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
        dbid = row["DBID"]
        chrom = str(row["CHROMO"]).strip()
        pos_1based = int(row["TARGET_POS_1BASED"])
        ref_csv = str(row["REF_BASE"]).strip().upper()
        alt_csv = str(row["ALT_BASE"]).strip().upper()
        target_pos = pos_1based - 1          # 0-based coordinate of the variant

        # Skip variants too close to the chromosome start (interval would be
        # clipped or negative, which breaks the fixed context length/centring).
        if target_pos - HALF_INTERVAL < 0:
            logging.warning(f"{dbid}: too close to chromosome start. Skipping.")
            skipped.append((dbid, "too close to chromosome start"))
            continue

        # --- Reference validation (0-based start, end = start + len(ref)) ---
        actual_ref = fetch_ucsc_sequence(chrom, target_pos,
                                         target_pos + len(ref_csv))
        if actual_ref is None:
            logging.error(f"{dbid}: UCSC fetch failed at {chrom}:{pos_1based}.")
            skipped.append((dbid, "UCSC fetch failed"))
            continue
        if actual_ref != ref_csv:
            logging.error(f"{dbid}: REF mismatch at {chrom}:{pos_1based}. "
                          f"CSV:{ref_csv} vs UCSC:{actual_ref}.")
            skipped.append((dbid, f"REF mismatch (CSV {ref_csv}, UCSC {actual_ref})"))
            continue

        # --- Prediction ---
        interval = genome.Interval(
            chromosome=chrom,
            start=target_pos - HALF_INTERVAL,     # 0-based
            end=target_pos + HALF_INTERVAL,       # 0-based, half-open
        )
        variant = genome.Variant(
            chromosome=chrom,
            position=pos_1based,                  # 1-based
            reference_bases=ref_csv,
            alternate_bases=alt_csv,
        )

        try:
            response = predict_with_retry(
                model,
                interval=interval,
                variant=variant,
                requested_outputs=[dna_client.OutputType.SPLICE_JUNCTIONS],
                ontology_terms=[],
            )

            ref_obj = response.reference.splice_junctions
            alt_obj = response.alternate.splice_junctions
            junction_defs = ref_obj.junctions
            ref_vals = ref_obj.values                 # (n_junctions, n_tracks)
            alt_vals = alt_obj.values

            if ref_vals.shape != alt_vals.shape or ref_vals.shape[0] != len(junction_defs):
                logging.error(f"{dbid}: unexpected junction output shape "
                              f"{ref_vals.shape} / {alt_vals.shape}. Skipping.")
                skipped.append((dbid, f"unexpected junction output shape {ref_vals.shape}"))
                continue

            variant_junctions = []
            if ref_vals.shape[0] > 0:
                lnfc = ln_fold_change(ref_vals, alt_vals)          # (n_junctions, n_tracks)
                junction_idx = np.arange(lnfc.shape[0])
                peak_idx = np.argmax(np.abs(lnfc), axis=1)         # strongest track per junction
                max_lnfc = lnfc[junction_idx, peak_idx]
                peak_ref = ref_vals[junction_idx, peak_idx]
                peak_alt = alt_vals[junction_idx, peak_idx]

                for i, j in enumerate(junction_defs):
                    dist = min(abs(j.start - target_pos), abs(j.end - target_pos))
                    variant_junctions.append({
                        "LnFC": float(max_lnfc[i]),
                        "Dist": int(dist),
                        "Coord": f"{j.start}-{j.end}",
                        "Ref": float(peak_ref[i]),
                        "Alt": float(peak_alt[i]),
                    })

            # Largest |ln fold change| first
            variant_junctions.sort(key=lambda x: abs(x["LnFC"]), reverse=True)

            row_data = {
                "DBID": dbid,
                "CHROM": chrom,
                "Position_1Based": pos_1based,
                "Ref_Alt": f"{ref_csv}>{alt_csv}",
                "ClinicalClassification": row.get("ClinicalClassification", "Unknown"),
                "Total_Junctions": len(variant_junctions),
            }

            for rank in range(1, N_TOP_JUNCTIONS + 1):
                if len(variant_junctions) >= rank:
                    item = variant_junctions[rank - 1]
                    row_data[f"J{rank}_Dist"] = item["Dist"]
                    row_data[f"J{rank}_LnFC"] = item["LnFC"]
                    row_data[f"J{rank}_Ref"] = item["Ref"]
                    row_data[f"J{rank}_Alt"] = item["Alt"]
                    row_data[f"J{rank}_Coord"] = item["Coord"]
                else:
                    row_data[f"J{rank}_Dist"] = np.nan
                    row_data[f"J{rank}_LnFC"] = np.nan
                    row_data[f"J{rank}_Ref"] = np.nan
                    row_data[f"J{rank}_Alt"] = np.nan
                    row_data[f"J{rank}_Coord"] = None

            all_results.append(row_data)
            n_scored += 1

        except Exception as e:
            logging.warning(f"API/processing failure for {dbid}: {e}")
            skipped.append((dbid, f"API/processing failure: {e}"))

    # ── Export & summary ──────────────────────────────────────────────────────
    logging.info(f"Scored {n_scored}/{len(df_input)} variants; "
                 f"{len(skipped)} skipped.")

    if skipped:
        skipped_path = join(OUTPUT_PATH, f"{FILE_NAME}_skipped.csv")
        pd.DataFrame(skipped, columns=["DBID", "Reason"]).to_csv(skipped_path, index=False)
        logging.warning(f"Skipped variants written to {skipped_path}")

    if all_results:
        cols = ["DBID", "CHROM", "Position_1Based", "Ref_Alt",
                "ClinicalClassification", "Total_Junctions"]
        for rank in range(1, N_TOP_JUNCTIONS + 1):
            cols += [f"J{rank}_Dist", f"J{rank}_LnFC",
                     f"J{rank}_Ref", f"J{rank}_Alt", f"J{rank}_Coord"]

        df_final = pd.DataFrame(all_results).reindex(columns=cols)
        csv_path = join(OUTPUT_PATH, f"{FILE_NAME}.csv")
        df_final.to_csv(csv_path, index=False)
        logging.info(f"Successfully exported to {csv_path}")
        print(df_final.to_string(index=False))
    else:
        logging.error("No data processed successfully.")


if __name__ == "__main__":
    main()
