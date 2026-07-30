"""
STEP 2: Full Cohort Download: AD + CN
  - Downloads ALL available AD (Group == 'A') and CN (Group == 'C') subjects
  - Excludes FTD (Group == 'F')
  - Uses openneuro-py to download derivatives only
"""
import openneuro as on
import pandas as pd
import os

df = pd.read_csv('data/participants.tsv', sep='\t')
df['Group'] = df['Group'].str.strip()

# ALL AD and CN subjects (not just head(25))
ad_ids = df[df['Group'] == 'A']['participant_id'].tolist()
cn_ids = df[df['Group'] == 'C']['participant_id'].tolist()
targets = ad_ids + cn_ids

print(f"Targeting {len(targets)} subjects: {len(ad_ids)} AD, {len(cn_ids)} CN")
print("Excluding FTD subjects entirely.\n")

for sub_id in targets:
    print(f"Downloading {sub_id}...")
    try:
        on.download(dataset='ds004504',
                    target_dir='./data',
                    include=[f'derivatives/{sub_id}'])
    except Exception as e:
        print(f"  Could not download {sub_id}: {e}")

print("\nDownload complete. Check data/derivatives/")
