"""
STEP 01
Download Participant Metadata
- Downloads the participants.tsv from OpenNeuro ds004504 to inspect group labels before downloading EEG files.
"""


#Import
import openneuro as on
import pandas as pd
import os

if not os.path.exists('data'):
    os.makedirs('data')

print("Downloading participant metadata...")
on.download(dataset='ds004504',
            target_dir='./data',
            include=['participants.tsv'])

df = pd.read_csv('data/participants.tsv', sep='\t')
print("\n--- Dataset Summary ---")
print(df['Group'].value_counts())
print("\nThis dataset has 36 Alzheimer's (A), 23 Frontotemporal Dementia (F), and 29 Healthy (C).")
