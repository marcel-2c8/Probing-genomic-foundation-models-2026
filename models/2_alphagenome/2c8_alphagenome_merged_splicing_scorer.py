### ALPHAGENOME - MERGED SPLICING SCORER ###
"""
AlphaGenome "merged splicing" score for a list of variants (Colab). Follows the
AlphaGenome splicing tutorial, the approach used in the paper to score ClinVar
variants for missplicing.

Three splicing scorers are run per variant:
  - SPLICE_SITES       : change in splice site class probabilities (donor/acceptor)
  - SPLICE_SITE_USAGE  : change in relative splice site usage
  - SPLICE_JUNCTIONS   : change in splice junction usage (log fold change)

For each variant the max raw_score is taken per scorer (across genes and
tracks) and combined as:

  alphagenome_splicing = max(splice_sites) + max(splice_site_usage)
                         + max(splice_junctions) / 5

Junctions are down-weighted because their magnitude is larger. Per the
documentation, most variants score in [0, 6], and scores > 1.0 generally
indicate a large effect (no official cutoff yet).

Coordinate conventions (verified against alphagenome genome.py):
  - genome.Variant.position      : 1-based (VCF-style)
  - UCSC getData/sequence        : start 0-based, end 1-based (BED-style)

The prediction interval is centred on the variant with
`variant.reference_interval.resize(...)`. The Variant gets the 1-based
position straight from the CSV.

Input CSV columns: DBID, CHROMO, TARGET_POS_1BASED, REF_BASE, ALT_BASE
                   (optional: ClinicalClassification)
"""
import logging
import os
import time
from os.path import join

import pandas as pd
import requests
from tqdm import tqdm
from google.colab import userdata

from alphagenome.data import genome
from alphagenome.models.v1 import dna_client, variant_scorers

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
FILE_NAME = "merged_splicing_scores"

REQUIRED_COLUMNS = ["DBID", "CHROMO", "TARGET_POS_1BASED", "REF_BASE", "ALT_BASE"]

ORGANISM = dna_client.Organism.HOMO_SAPIENS

# 1MB gives the best predictions for gene-level scoring (per the tutorial).
SEQUENCE_LENGTH_KEY = "1MB"
SEQUENCE_LENGTH = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
    f"SEQUENCE_LENGTH_{SEQUENCE_LENGTH_KEY}"
]
HALF_INTERVAL = SEQUENCE_LENGTH // 2

SPLICING_SCORER_KEYS = ["SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"]
JUNCTION_WEIGHT_DIVISOR = 5.0      # alphagenome_splicing uses junctions / 5

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


def get_splicing_scorers():
    """The three recommended splicing scorers, as in the AlphaGenome tutorial."""
    recommended = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    scorers = [recommended[key] for key in SPLICING_SCORER_KEYS]
    for scorer in scorers:
        logging.info(f"Scorer: {scorer.name} (signed={scorer.is_signed})")
    return scorers


def score_with_retry(model, **kwargs):
    """score_variant with a fixed pause afterwards and backoff on quota errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = model.score_variant(**kwargs)
            time.sleep(REQUEST_DELAY)
            return result
        except Exception as e:
            msg = str(e)
            is_quota = "RESOURCE_EXHAUSTED" in msg or "Quota exceeded" in msg
            if not is_quota or attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * attempt
            logging.warning(f"Quota exceeded (attempt {attempt}/{MAX_RETRIES}). "
                            f"Waiting {wait}s before retrying...")
            time.sleep(wait)


def compute_merged_splicing_score(df):
    """Merged splicing score per variant, from tidy scores (tutorial method).

    Takes the max raw_score per variant and output type, then combines:
      alphagenome_splicing = SPLICE_SITES + SPLICE_SITE_USAGE
                             + SPLICE_JUNCTIONS / 5
    """
    df = df.copy()
    df["variant_id"] = df["variant_id"].map(str)
    max_scores = (
        df.groupby(["variant_id", "output_type"])["raw_score"]
        .max()
        .reset_index()
        .pivot(index="variant_id", columns="output_type", values="raw_score")
        .fillna(0.0)
    )

    for key in SPLICING_SCORER_KEYS:
        if key not in max_scores.columns:
            logging.warning(f"No '{key}' scores found in the output; counted as 0.")

    max_scores["alphagenome_splicing"] = (
        max_scores.get("SPLICE_SITES", 0.0)
        + max_scores.get("SPLICE_SITE_USAGE", 0.0)
        + max_scores.get("SPLICE_JUNCTIONS", 0.0) / JUNCTION_WEIGHT_DIVISOR
    )
    return max_scores.reset_index()


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

    df_input = pd.read_csv(CSV_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df_input.columns]
    if missing:
        logging.error(f"Input file is missing required columns: {missing}")
        return

    model = dna_client.create(api_key)
    splicing_scorers = get_splicing_scorers()

    logging.info(f"Sequence length {SEQUENCE_LENGTH_KEY}. "
                 f"Processing {len(df_input)} variants from {CSV_PATH}...")

    tidy_frames = []            # tidy scores, one frame per successfully scored variant
    skipped = []                # (DBID, reason)

    for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
        dbid = str(row["DBID"])
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

        # --- Scoring ---
        variant = genome.Variant(
            chromosome=chrom,
            position=pos_1based,                  # 1-based
            reference_bases=ref_csv,
            alternate_bases=alt_csv,
            name=dbid,
        )
        interval = variant.reference_interval.resize(SEQUENCE_LENGTH)

        try:
            variant_scores = score_with_retry(
                model,
                interval=interval,
                variant=variant,
                variant_scorers=splicing_scorers,
                organism=ORGANISM,
            )
            # Tidy each variant on its own and label it with the input DBID, so
            # rows can be matched back to the CSV without relying on the format
            # of the variant_id that tidy_scores generates.
            df_one = variant_scorers.tidy_scores([variant_scores])
            df_one["variant_key"] = df_one["variant_id"].map(str)   # library's own id
            df_one["variant_id"] = dbid                             # input DBID
            tidy_frames.append(df_one)
        except Exception as e:
            logging.warning(f"API/processing failure for {dbid}: {e}")
            skipped.append((dbid, f"API/processing failure: {e}"))

    # ── Export & summary ──────────────────────────────────────────────────────
    logging.info(f"Scored {len(tidy_frames)}/{len(df_input)} variants; "
                 f"{len(skipped)} skipped.")

    if skipped:
        skipped_path = join(OUTPUT_PATH, f"{FILE_NAME}_skipped.csv")
        pd.DataFrame(skipped, columns=["DBID", "Reason"]).to_csv(skipped_path, index=False)
        logging.warning(f"Skipped variants written to {skipped_path}")

    if not tidy_frames:
        logging.error("No data processed successfully.")
        return

    df_tidy = pd.concat(tidy_frames, ignore_index=True)

    # Interval objects are not serializable by PyArrow (parquet); store as text.
    if "scored_interval" in df_tidy.columns:
        df_tidy["scored_interval"] = df_tidy["scored_interval"].apply(
            lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
        )

    # Save the per-track scores first, so the raw scores survive any later error.
    tidy_path = join(OUTPUT_PATH, f"{FILE_NAME}_tidy_tracks.csv")
    df_tidy.to_csv(tidy_path, index=False)

    merged = compute_merged_splicing_score(df_tidy)
    # Add all columns of the input CSV (original order), followed by the scores.
    input_cols = list(df_input.columns)
    df_original = df_input.assign(variant_id=df_input["DBID"].astype(str))
    merged = merged.merge(df_original, on="variant_id", how="left")
    if merged["DBID"].isna().any():
        logging.warning("Some scored variants could not be matched back to the "
                        "input CSV; their input columns are empty.")
    score_cols = [*SPLICING_SCORER_KEYS, "alphagenome_splicing"]
    merged = merged[[*input_cols, *[c for c in score_cols if c in merged.columns]]]

    csv_path = join(OUTPUT_PATH, f"{FILE_NAME}.csv")
    merged.to_csv(csv_path, index=False)
    logging.info(f"Successfully exported {len(merged)} variants to {csv_path}")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()