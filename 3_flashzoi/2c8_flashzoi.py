### Borzoi variant effect prediction pipeline ###
# Flashzoi implementation - 22 ENCODE Tracks - Forward-Reverse Complement Averaging - Ensemble of 4 replicates
import os
import gc
import time
import requests
import torch
import numpy as np
import pandas as pd
from borzoi_pytorch import Borzoi
from tqdm import tqdm
import logging

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR      = "./data"
CSV_PATH      = os.path.join(BASE_DIR, "opa1_variants.csv")
TARGETS_PATH  = os.path.join(BASE_DIR, "./targets_human.txt")
OUTPUT_FOLDER = os.path.join( "./outputs/opa1_flashzoi")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

WINDOW_SIZE = 524288
BIN_SIZE    = 32 #Adjust accordingly

BORZOI_TRACK_ID = [
    6068, 6070, 6072, 6074, 6076, 6078, 6080, 6082, 6084, 6086,
    6088, 6090, 6092, 6094, 6096, 6098, 6100, 6102, 6104, 6106,
    6108, 6110,
] 

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)

# ── UCSC sequence fetcher (replaces pysam) ────────────────────────────────────
UCSC_ENDPOINT = "https://api.genome.ucsc.edu/getData/sequence"
GENOME_BUILD  = "hg38"

def fetch_ucsc_sequence(chrom: str, start_0based: int, end_1based: int,
                        retries: int = 3) -> str | None:
    """
    Fetch reference sequence from UCSC REST API.
    Coordinates: start 0-based, end 1-based (BED convention, same as pysam.fetch).
    """
    params = {
        "genome": GENOME_BUILD,
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

# ── Encoding helpers ──────────────────────────────────────────────────────────
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode(seq: str) -> torch.Tensor:
    mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1]}
    return torch.tensor(
        [mapping.get(b, [0,0,0,0]) for b in seq], dtype=torch.float32
    ).t()

def reverse_complement_tensor(dna: torch.Tensor) -> torch.Tensor:
    return torch.flip(dna, dims=[1, 2])

# ── Load targets & variants ───────────────────────────────────────────────────
targets_df = pd.read_csv(TARGETS_PATH, sep="\t")
targets_df = targets_df.iloc[BORZOI_TRACK_ID].reset_index(drop=True)

variant_df = pd.read_csv(CSV_PATH)
variants   = variant_df[variant_df["type"] == "subst"].to_dict("records")

# ── In-memory cache ───────────────────────────────────────────────────────────
variant_cache: dict[str, pd.DataFrame] = {}

def get_variant_df(vid: str, file_path: str) -> pd.DataFrame:
    if vid not in variant_cache:
        if os.path.exists(file_path):
            variant_cache[vid] = pd.read_parquet(file_path)
        else:
            variant_cache[vid] = targets_df.copy()
    return variant_cache[vid]

# ── Inference ─────────────────────────────────────────────────────────────────
for i in range(4):
    logging.info(f"Loading Replicate {i}...")
    model = Borzoi.from_pretrained(f"johahi/borzoi-replicate-{i}").to("cuda").eval()

    with torch.inference_mode():
        for v in tqdm(variants, desc=f"Rep {i}"):
            vid        = v["DBID"]
            chrom      = str(v["CHROMO"]).strip()          # ← read chromosome from CSV
            target_pos = v["TARGET_POS_1BASED"] - 1        # convert to 0-based

            standard_start = target_pos - (WINDOW_SIZE // 2)
            adjustment     = ((target_pos - standard_start) % BIN_SIZE) - (BIN_SIZE // 2)
            window_start   = max(0, int(standard_start + adjustment))
            window_end     = window_start + WINDOW_SIZE

            file_path = os.path.join(
                OUTPUT_FOLDER, f"variant_{vid}_{v['TARGET_POS_1BASED']}.parquet"
            )

            try:
                # Fetch sequence from UCSC instead of local FASTA
                raw_seq = fetch_ucsc_sequence(chrom, window_start, window_end)

                if raw_seq is None:
                    logging.error(f"{vid}: UCSC fetch failed for "
                                  f"{chrom}:{window_start}-{window_end}, skipping")
                    continue

                if len(raw_seq) != WINDOW_SIZE:
                    logging.error(
                        f"{vid}: short sequence ({len(raw_seq)} bp) — "
                        f"near chromosome boundary, skipping"
                    )
                    continue

                rel_pos = target_pos - window_start

                if raw_seq[rel_pos] != v["REF_BASE"]:
                    logging.error(
                        f"{vid}: REF mismatch at {chrom}:{target_pos+1} "
                        f"(expected {v['REF_BASE']}, got {raw_seq[rel_pos]}), skipping"
                    )
                    continue

                # ── Build forward tensors ─────────────────────────────────────
                wt_fwd  = one_hot_encode(raw_seq).to("cuda").unsqueeze(0)
                mut_fwd = wt_fwd.clone()
                mut_idx = BASE_TO_IDX[v["ALT_BASE"]]
                mut_fwd[0, :, rel_pos]       = 0
                mut_fwd[0, mut_idx, rel_pos] = 1

                # ── Build RC tensors ──────────────────────────────────────────
                wt_rc  = reverse_complement_tensor(wt_fwd)
                mut_rc = reverse_complement_tensor(mut_fwd)

                # ── Single forward pass for all 4 sequences ───────────────────
                batch = torch.cat([wt_fwd, mut_fwd, wt_rc, mut_rc], dim=0)
                del wt_fwd, mut_fwd, wt_rc, mut_rc

                out = model(batch).detach().cpu().float().numpy()[:, BORZOI_TRACK_ID, :]
                del batch

                # ── Unpack; flip RC bin axis ──────────────────────────────────
                fwd_wt  = out[0]
                fwd_mut = out[1]
                rc_wt   = out[2, :, ::-1].copy()
                rc_mut  = out[3, :, ::-1].copy()
                del out

                wt_out  = (fwd_wt + rc_wt)  * 0.5
                mut_out = (fwd_mut + rc_mut) * 0.5

                # ── Locate target bin ─────────────────────────────────────────
                num_bins         = wt_out.shape[1]
                output_start_rel = (WINDOW_SIZE - num_bins * BIN_SIZE) // 2
                target_bin       = (rel_pos - output_start_rel) // BIN_SIZE

                if not (0 <= target_bin < num_bins):
                    logging.error(
                        f"{vid}: target_bin={target_bin} out of range "
                        f"[0, {num_bins}), skipping"
                    )
                    continue

                wt_vals  = wt_out[:, target_bin].copy()
                mut_vals = mut_out[:, target_bin].copy()

                # ── Log transform ─────────────────────────────────────────────
                eps = 1e-8
                log_wt_vals  = np.log(np.maximum(wt_vals,  0) + eps)
                log_mut_vals = np.log(np.maximum(mut_vals, 0) + eps)

                # ── Update cache ──────────────────────────────────────────────
                df_var = get_variant_df(vid, file_path)
                df_var[f"rep_{i}_wt"]      = wt_vals
                df_var[f"rep_{i}_mut"]     = mut_vals
                df_var[f"rep_{i}_wt_log"]  = log_wt_vals
                df_var[f"rep_{i}_mut_log"] = log_mut_vals

                existing = df_var["reps_present"].iloc[0] if "reps_present" in df_var.columns else ""
                df_var["reps_present"] = existing + f"{i},"

            except torch.cuda.OutOfMemoryError:
                logging.error(f"{vid} rep {i}: CUDA OOM — skipping and clearing cache")
                for name in ("batch", "out", "wt_out", "mut_out"):
                    if name in dir():
                        try:
                            exec(f"del {name}")
                        except Exception:
                            pass
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logging.error(f"{vid} rep {i}: {e}", exc_info=True)
                continue

    del model
    gc.collect()
    torch.cuda.empty_cache()

# ── Flush cache to parquet with ensemble averages ─────────────────────────────
logging.info("Writing parquet files with ensemble averages...")
for vid, df_var in variant_cache.items():
    v_row = next(v for v in variants if v["DBID"] == vid)
    file_path = os.path.join(
        OUTPUT_FOLDER, f"variant_{vid}_{v_row['TARGET_POS_1BASED']}.parquet"
    )

    wt_cols      = [c for c in df_var.columns if c.startswith("rep_") and c.endswith("_wt")]
    mut_cols     = [c for c in df_var.columns if c.startswith("rep_") and c.endswith("_mut")]
    wt_log_cols  = [c for c in df_var.columns if c.startswith("rep_") and c.endswith("_wt_log")]
    mut_log_cols = [c for c in df_var.columns if c.startswith("rep_") and c.endswith("_mut_log")]

    if wt_cols:
        df_var["ensemble_mean_wt"]  = df_var[wt_cols].mean(axis=1)
        df_var["ensemble_mean_mut"] = df_var[mut_cols].mean(axis=1)

    if wt_log_cols:
        df_var["ensemble_mean_wt_log"]  = df_var[wt_log_cols].mean(axis=1)
        df_var["ensemble_mean_mut_log"] = df_var[mut_log_cols].mean(axis=1)
        df_var["ensemble_lfc"] = (df_var["ensemble_mean_mut_log"]
                                  - df_var["ensemble_mean_wt_log"])

    df_var.to_parquet(file_path)

logging.info(f"Done. Output in {OUTPUT_FOLDER}")