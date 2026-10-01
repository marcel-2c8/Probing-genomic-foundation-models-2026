import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.multiprocessing.set_sharing_strategy('file_system')

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID            = "ZhejiangLab/Genos-10B"
CSV_PATH            = './data/opa1_variants.csv'
OUTPUT_DIR          = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_FOR_SCORING = 4096
CONTEXT_FOR_PREFIX  = 94208
BATCH_SIZE          = 1

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
            print(f"[UCSC] HTTP {resp.status_code} for "
                  f"{chrom}:{start_0based}-{end_1based}")
        except Exception as e:
            print(f"[UCSC] Attempt {attempt+1} failed: {e}")
        time.sleep(1.5 * (attempt + 1))
    return None

def fetch_chrom_length(chrom: str) -> int | None:
    """Fetch chromosome length from UCSC REST API (replaces fasta.get_reference_length)."""
    try:
        resp = requests.get(
            "https://api.genome.ucsc.edu/getData/track",
            params={"genome": GENOME_BUILD, "track": "chromInfo", "chrom": chrom},
            timeout=30
        )
        if resp.status_code == 200:
            for item in resp.json().get("chromInfo", []):
                if item.get("chrom") == chrom:
                    return int(item["size"])
        print(f"[UCSC] Could not fetch length for {chrom}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"[UCSC] chrom length fetch failed: {e}")
    return None

# ── Reverse complement helper ─────────────────────────────────────────────────
COMPLEMENT = str.maketrans('ACGT', 'TGCA')

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

# ── Model Loading ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
model.eval()

# ── PyTorch Dataset ───────────────────────────────────────────────────────────
# Chromosome lengths are cached here to avoid one API call per __getitem__
_chrom_length_cache: dict[str, int] = {}

class VariantDatasetRC(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        pos  = int(row['TARGET_POS_1BASED'])
        ref  = str(row['REF_BASE']).strip().upper()
        alt  = str(row['ALT_BASE']).strip().upper()
        dbid = str(row['DBID'])
        chrom = str(row['CHROMO']).strip()             # ← read chromosome from CSV

        # Chromosome length (cached per chrom)
        if chrom not in _chrom_length_cache:
            length = fetch_chrom_length(chrom)
            if length is None:
                return {"skip": True, "dbid": dbid, "pos": pos}
            _chrom_length_cache[chrom] = length
        chr_len = _chrom_length_cache[chrom]

        left_start = max(0, (pos - 1) - CONTEXT_FOR_SCORING)
        right_end  = min(chr_len, pos + CONTEXT_FOR_PREFIX)

        # Validate ref base
        actual_ref = fetch_ucsc_sequence(chrom, pos - 1, pos)
        if actual_ref is None:
            return {"skip": True, "dbid": dbid, "pos": pos}
        if actual_ref != ref:
            return {"skip": True, "dbid": dbid, "pos": pos}

        # Fetch left and right flanks separately
        left_flank  = fetch_ucsc_sequence(chrom, left_start, pos - 1)
        right_flank = fetch_ucsc_sequence(chrom, pos, right_end)
        if left_flank is None or right_flank is None:
            return {"skip": True, "dbid": dbid, "pos": pos}

        # Build forward windows and reverse-complement
        fwd_ref_window = left_flank + ref + right_flank
        fwd_alt_window = left_flank + alt + right_flank
        rc_ref_window  = reverse_complement(fwd_ref_window)
        rc_alt_window  = reverse_complement(fwd_alt_window)

        fwd_window_len = len(fwd_ref_window)
        rel_pos_rc     = fwd_window_len - 1 - len(left_flank)

        return {
            "skip":       False,
            "dbid":       dbid,
            "pos":        pos,
            "chrom":      chrom,                       # ← carry through to results
            "ref_window": rc_ref_window,
            "mt_window":  rc_alt_window,
            "rel_pos":    rel_pos_rc,
        }

# ── Scoring helper ────────────────────────────────────────────────────────────
def get_expression_track_batched(out, inputs):
    logits       = out.logits.float()
    input_ids    = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask   = attention_mask[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    gathered_log_probs = torch.gather(
        log_probs, 2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    gathered_log_probs = gathered_log_probs * shift_mask
    return gathered_log_probs, shift_mask

# ── Main Execution Loop ───────────────────────────────────────────────────────
df      = pd.read_csv(CSV_PATH)
dataset = VariantDatasetRC(df)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

results = []
device  = next(model.parameters()).device
print(f"Saturating Tensor Cores with Batch Size {BATCH_SIZE}...")

for batch in tqdm(dataloader):
    valid_mask = [not s for s in batch['skip']]
    if not any(valid_mask):
        continue

    dbids       = [b for b, v in zip(batch['dbid'],       valid_mask) if v]
    poses       = [int(p) for p, v in zip(batch['pos'],   valid_mask) if v]
    chroms      = [c for c, v in zip(batch['chrom'],       valid_mask) if v]
    ref_windows = [w for w, v in zip(batch['ref_window'], valid_mask) if v]
    mt_windows  = [w for w, v in zip(batch['mt_window'],  valid_mask) if v]
    rel_poses   = [int(r) for r, v in zip(batch['rel_pos'], valid_mask) if v]

    with torch.inference_mode():
        inputs_wt = tokenizer(ref_windows, padding=True, return_tensors="pt").to(device)
        inputs_mt = tokenizer(mt_windows,  padding=True, return_tensors="pt").to(device)

        out_wt = model(**inputs_wt)
        out_mt = model(**inputs_mt)

        track_wt, mask_wt = get_expression_track_batched(out_wt, inputs_wt)
        track_mt, mask_mt = get_expression_track_batched(out_mt, inputs_mt)

    track_wt_np = track_wt.cpu().numpy()
    track_mt_np = track_mt.cpu().numpy()

    for i in range(len(dbids)):
        variant_token_idx = inputs_wt.char_to_token(i, rel_poses[i])

        if variant_token_idx is None:
            prefix_tokens = tokenizer(
                ref_windows[i][:rel_poses[i]],
                return_tensors="pt",
                add_special_tokens=False
            )
            prefix_len   = prefix_tokens['input_ids'].shape[1]
            total_len    = inputs_wt['input_ids'].shape[1]
            content_len  = int(inputs_wt['attention_mask'][i].sum().item())
            pad_offset   = total_len - content_len
            variant_token_idx = pad_offset + prefix_len

        variant_token_idx = max(0, variant_token_idx - 1)

        sum_wt = float(np.sum(track_wt_np[i, variant_token_idx:]))
        sum_mt = float(np.sum(track_mt_np[i, variant_token_idx:]))
        llr    = sum_mt - sum_wt

        results.append({
            'DBID':     dbids[i],
            'CHROM':    chroms[i],                     # ← chromosome in output
            'POS':      poses[i],
            'RC_REF_LL': sum_wt,
            'RC_ALT_LL': sum_mt,
            'RC_LLR':    llr,
        })

    del inputs_wt, inputs_mt, out_wt, out_mt, track_wt, track_mt
    del track_wt_np, track_mt_np, mask_wt, mask_mt
    torch.cuda.empty_cache()

# ── Output ────────────────────────────────────────────────────────────────────
output_df = pd.DataFrame(results)
output_df.to_csv(os.path.join(OUTPUT_DIR, 'opa1_genos10b_rc.csv'), index=False)
print(f"Done. Processed {len(output_df)} variants using parallel batched processing.")