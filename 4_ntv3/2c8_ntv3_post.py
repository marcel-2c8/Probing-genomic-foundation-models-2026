import torch
import os
import time
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging
import gc
from huggingface_hub import login

# ── AUTH & CONFIG ─────────────────────────────────────────────────────────────
HF_TOKEN = os.environ["HF_TOKEN"]          # fixed: () → [] (environ is a mapping)
login(token=HF_TOKEN, add_to_git_credential=False)

MODEL_ID       = "InstaDeepAI/NTv3_650M_post"
WINDOW_SIZE    = 16384
BATCH_SIZE     = 12
CONTEXT_WINDOW = 16
CSV_PATH       = "./data/opa1_variants.csv"
OUTPUT_PATH    = "./outputs/opa1_ntv3_post.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
logging.basicConfig(level=logging.INFO)

# ── UCSC sequence fetcher ─────────────────────────────────────────────────────
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

# ── Model initialisation ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda").eval()

base_species_ids = model.encode_species(['human']).to("cuda")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_rc(seq: str) -> str:
    rc_map = str.maketrans("ACGT", "TGCA")
    return seq.translate(rc_map)[::-1]

def process_batch(batch_data: list) -> list:
    batch_results = []
    num_vars = len(batch_data)

    fwd_refs, fwd_alts, rc_refs, rc_alts = [], [], [], []
    for v in batch_data:
        s_list = list(v['seq'])
        s_list[v['rel_pos']] = v['alt']
        fwd_alt_seq = "".join(s_list)

        fwd_refs.append(v['seq'])
        fwd_alts.append(fwd_alt_seq)
        rc_refs.append(get_rc(v['seq']))
        rc_alts.append(get_rc(fwd_alt_seq))

    all_seqs = fwd_refs + fwd_alts + rc_refs + rc_alts

    try:
        inputs = tokenizer(
            all_seqs,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=128,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch_species = base_species_ids.repeat(len(all_seqs), 1).view(-1)
                logits = model(**inputs, species_ids=batch_species).logits
                
        for i in range(num_vars):
            v_info  = batch_data[i]
            L       = len(v_info['seq'])
            fwd_pos = v_info['rel_pos']
            rc_pos  = L - 1 - fwd_pos

            def calculate_llr(batch_idx_ref, batch_idx_alt, target_pos):
                sum_ref = sum_alt = 0.0
                for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
                    p = target_pos + offset
                    if 0 <= p < logits.shape[1]:
                        tid_ref  = inputs.input_ids[batch_idx_ref, p]
                        tid_alt  = inputs.input_ids[batch_idx_alt, p]
                        lp_ref   = torch.log_softmax(logits[batch_idx_ref, p, :], dim=-1)
                        lp_alt   = torch.log_softmax(logits[batch_idx_alt, p, :], dim=-1)
                        sum_ref += lp_ref[tid_ref].item()
                        sum_alt += lp_alt[tid_alt].item()
                return sum_alt - sum_ref

            llr_fwd = calculate_llr(i,               i + num_vars,     fwd_pos)
            llr_rc  = calculate_llr(i + 2 * num_vars, i + 3 * num_vars, rc_pos)
            avg_llr = (llr_fwd + llr_rc) / 2

            batch_results.append({
                "DBID":            v_info['dbid'],
                "CHROM":           v_info['chrom'],    # ← carry chromosome through
                "NTv3_RC_AVG_LLR": avg_llr,
                "NTv3_FWD_LLR":    llr_fwd,
                "NTv3_RC_LLR":     llr_rc,
                "status":          "success",
            })

    except Exception as e:
        logging.error(f"Batch failed: {e}")
        for v in batch_data:
            batch_results.append({
                "DBID":  v['dbid'],
                "CHROM": v['chrom'],
                "status": "failed"
            })

    return batch_results

# ── Execution ─────────────────────────────────────────────────────────────────
df       = pd.read_csv(CSV_PATH)
variants = df[df["type"] == "subst"].to_dict("records")
final_results = []

for i in tqdm(range(0, len(variants), BATCH_SIZE), desc="NTv3 RC-Averaging Pipeline"):
    batch_subset = variants[i : i + BATCH_SIZE]
    batch_prep   = []

    for v in batch_subset:
        chrom = str(v["CHROMO"]).strip()               # ← read chromosome from CSV
        t_pos = v["TARGET_POS_1BASED"] - 1             # convert to 0-based
        start = max(0, t_pos - (WINDOW_SIZE // 2))
        end   = start + WINDOW_SIZE

        seq = fetch_ucsc_sequence(chrom, start, end)
        if seq is None:
            logging.warning(f"{v['DBID']}: UCSC fetch failed for "
                            f"{chrom}:{start}-{end}, skipping")
            continue
        if len(seq) < WINDOW_SIZE:
            logging.warning(f"{v['DBID']}: short sequence ({len(seq)} bp), skipping")
            continue

        rel_pos = t_pos - start
        if seq[rel_pos] != v["REF_BASE"]:
            logging.warning(f"{v['DBID']}: REF mismatch at {chrom}:{t_pos+1} "
                            f"(expected {v['REF_BASE']}, got {seq[rel_pos]}), skipping")
            continue

        batch_prep.append({
            'seq':     seq,
            'rel_pos': rel_pos,
            'ref':     v['REF_BASE'],
            'alt':     v['ALT_BASE'],
            'dbid':    v['DBID'],
            'chrom':   chrom,                          # ← pass through to results
        })

    if batch_prep:
        final_results.extend(process_batch(batch_prep))

    torch.cuda.empty_cache()
    gc.collect()

pd.DataFrame(final_results).to_csv(OUTPUT_PATH, index=False)
logging.info("Done. Final RC-Averaged results saved.")