"""
STEP 3: EEG Quality Check (Single Subject)
- Loads one subject's EEG and plots the power spectrum to verify 1/f decay (normal) and alpha peak (~10 Hz).
- Diagnostic only
"""

import mne
import matplotlib.pyplot as plt

sub_id = "sub-001"
data_path = f"data/derivatives/{sub_id}/eeg/{sub_id}_task-eyesclosed_eeg.set"

print(f"--- Loading {sub_id} ---")
raw = mne.io.read_raw_eeglab(data_path, preload=True)
print(raw.info)

spectrum = raw.compute_psd(fmin=1, fmax=40)
spectrum.plot(average=True, picks="eeg", exclude="bads")
plt.title(f"Power Spectrum: {sub_id}")
plt.savefig(f"figures/{sub_id}_power_spectrum.png", dpi=300)
plt.show()
print("Saved power spectrum figure.")
