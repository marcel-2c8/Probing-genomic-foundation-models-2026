### ALPHAGENOME - SPLICE SCORER ###
"""
AlphaGenome splice-site scoring for a list of variants (Colab).

Coordinate conventions (verified against alphagenome genome.py):
  - genome.Interval.start / .end : 0-based, half-open (BED-style)
  - genome.Variant.position      : 1-based (VCF-style)
  - UCSC getData/sequence        : start 0-based, end 1-based (BED-style)

The interval is centred on the true variant base (0-based `target_pos`), so
output index `len // 2` is the variant base and a distance of 0 means "at the
variant". The Variant itself gets the 1-based position.

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
FILE_NAME = "alphagenome_splicescorer_results"

RADIUS = 500                       # scoring window: variant +/- RADIUS (1001 bp)
INTERVAL_LENGTH = 1048576          # model context length (must stay fixed)
HALF_INTERVAL = INTERVAL_LENGTH // 2

# Used only if track metadata cannot be read from the response. Order is NOT
# verified; check the metadata line logged for the first variant.
FALLBACK_TRACK_NAMES = ["Donor (+)", "Acceptor (+)", "Donor (-)", "Acceptor (-)"]


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


def get_track_names(track_data, n_tracks):
    """Read track labels from the response metadata; fall back to a fixed list."""
    try:
        md = track_data.metadata
        names = [f"{n} ({s})" for n, s in zip(md["name"], md["strand"])]
        if len(names) == n_tracks:
            return names
    except Exception as e:
        logging.warning(f"Could not read track metadata ({e}).")
    logging.warning("Using FALLBACK_TRACK_NAMES; track order is unverified.")
    return FALLBACK_TRACK_NAMES[:n_tracks]


def get_api_key():
    try:
        key = userdata.get("api_key")
    except Exception as e:  # Colab raises if the secret is missing/not shared
        logging.error(f"Could not read Colab secret 'api_key': {e}")
        return None
    return key or None


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
    track_names = None

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
            response = model.predict_variant(
                interval=interval,
                variant=variant,
                requested_outputs=[dna_client.OutputType.SPLICE_SITES],
                ontology_terms=[],
            )

            ref_vals = response.reference.splice_sites.values
            alt_vals = response.alternate.splice_sites.values

            if ref_vals.shape[0] != INTERVAL_LENGTH:
                logging.error(f"{dbid}: unexpected output length "
                              f"{ref_vals.shape[0]}. Skipping.")
                skipped.append((dbid, f"unexpected output length {ref_vals.shape[0]}"))
                continue

            if track_names is None:
                track_names = get_track_names(response.reference.splice_sites,
                                              ref_vals.shape[1])
                logging.info(f"Track labels: {track_names}")

            delta_matrix = alt_vals - ref_vals

            center_idx = ref_vals.shape[0] // 2          # the variant base
            local_slice = slice(center_idx - RADIUS, center_idx + RADIUS + 1)

            for t_idx, name in enumerate(track_names):
                l_deltas = delta_matrix[local_slice, t_idx]
                l_ref = ref_vals[local_slice, t_idx]
                l_alt = alt_vals[local_slice, t_idx]

                loss_idx = int(np.argmin(l_deltas))
                gain_idx = int(np.argmax(l_deltas))

                all_results.append({
                    "DBID": dbid,
                    "CHROM": chrom,
                    "Position_1Based": pos_1based,
                    "Ref_Alt": f"{ref_csv}>{alt_csv}",
                    "ClinicalClassification": row.get("ClinicalClassification", "Unknown"),
                    "Track": name,
                    "Max_Loss_Delta": float(l_deltas[loss_idx]),
                    "Loss_Ref_Prob": float(l_ref[loss_idx]),
                    "Loss_Alt_Prob": float(l_alt[loss_idx]),
                    "Loss_Dist": loss_idx - RADIUS,
                    "Max_Gain_Delta": float(l_deltas[gain_idx]),
                    "Gain_Ref_Prob": float(l_ref[gain_idx]),
                    "Gain_Alt_Prob": float(l_alt[gain_idx]),
                    "Gain_Dist": gain_idx - RADIUS,
                })
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
        df_final = pd.DataFrame(all_results)
        csv_path = join(OUTPUT_PATH, f"{FILE_NAME}.csv")
        df_final.to_csv(csv_path, index=False)
        logging.info(f"Successfully exported to {csv_path}")
        print(df_final.to_string(index=False))
    else:
        logging.error("No data processed successfully.")


if __name__ == "__main__":
    main()
