### TEST SET ###
import pandas as pd
import os
from genos import create_client
from genos.exceptions import APIRequestError, ValidationError
from google.colab import userdata
import time

# Configuration
DATA_DIR = os.path.expanduser('./data/')
CSV_PATH = os.path.join(DATA_DIR, 'opa1_variants.csv')
OUTPUT_DIR = os.path.expanduser('./outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
file_name = "opa1_genos10b_clinvar.csv"
FULL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, file_name)

def run_batch_prediction(csv_file):
    # 1. Setup Authentication
    try:
        api_token = userdata.get('genos_api_key')
    except userdata.SecretNotFoundError:
        print("❌ Error: Please add 'genos_api_key' to Colab Secrets.")
        return

    client = create_client(token=api_token)

    # 2. Load your variant list
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    results_list = []

    print(f"🧬 Processing {len(df)} variants...")
    print("=" * 50)

    # 3. Iterate through the list
    for index, row in df.iterrows():
        dbid = row['DBID']
        chr = row['CHROMO']
        pos = row['TARGET_POS_1BASED']
        ref = row['REF_BASE']
        alt = row['ALT_BASE']

        try:
            response = client.variant_predict(
                assembly="hg38",
                chrom=(chr),
                pos=int(pos),
                ref=ref,
                alt=alt
            )

            res_data = response['result']
            results_list.append({
                "DBID": dbid,
                "Variant": res_data['variant'],
                "Prediction": res_data['prediction'],
                "Pathogenic_Score": res_data['score_Pathogenic'],
                "Benign_Score": res_data['score_Benign']
            })
            print(f"✅ Processed {dbid}: {res_data['prediction']}")

        except (APIRequestError, ValidationError) as e:
            print(f"❌ Error processing {dbid}: {e}")
            results_list.append({"DBID": dbid, "Prediction": "Error", "Pathogenic_Score": None, "Benign_Score": None})

        time.sleep(0.1)

    # 4. Save and Return Results
    final_df = pd.DataFrame(results_list)

    # Save to CSV
    final_df.to_csv(FULL_OUTPUT_PATH, index=False)
    print("\n" + "=" * 50)
    print(f"📂 Results saved to: {FULL_OUTPUT_PATH}")

    return final_df

# Run the process
results = run_batch_prediction(CSV_PATH)
results