# eeg-alzheimers-biomarker

Alpha-Band Hub Disruptions as an EEG Biomarker for Alzheimer's Disease: A Graph-Theoretical Analysis of Frontal Midline Centrality

<img width="4608" height="4608" alt="2026 Science Fair" src="https://github.com/user-attachments/assets/200b1c36-88ba-4b4b-9c21-e4acfd18b6d7" />

# ABSTRACT:

BACKGROUND: 
Current diagnostics for Alzheimer’s Disease (AD) rely on expensive neuroimaging or invasive lumbar
punctures, which often fail to detect pathology until significant, irreversible neuronal loss has occurred. There is an urgent need
for a low-cost, functional biomarker capable of detecting early-stage "miscommunication" within neural pathways. This study
aims to develop a topometric biomarker for AD by quantifying disruptions in brain network hub architecture using
graph-theoretical analysis of resting-state EEG data.

METHODS: 
Using a cohort of 65 subjects (AD = 36, CN = 29) from the OpenNeuro ds004504 dataset, EEG signals were
preprocessed and filtered into Theta (4-8 Hz), Alpha (8-13 Hz), and Beta (13-30 Hz) frequency bands. Functional connectivity
was estimated using Phase Lag Index (PLI) to construct subject-specific networks. A comprehensive battery of graph metrics was
calculated including betweenness centrality, modularity, and global efficiency.

RESULTS: 
While Theta and Beta bands showed preserved global topology, Alpha-band analysis revealed significant network
fragmentation, with elevated modularity (p = 0.045, d = 0.50). Critically, betweenness centrality of the Fz electrode, a frontal
midline hub, showed a robust effect surviving FDR correction (p < 0.001, d = 0.945). As a single-feature classifier, Fz centrality
achieved the highest discriminative accuracy (AUC = 0.824, 95% CI [0.720-0.916]). Interestingly, multi-feature models
combining Alpha and Beta hubs (Fz + P4, AUC = 0.812) did not outperform the single-electrode Alpha Fz model.

CONCLUSION: 
This finding is consistent with compensatory hub loading, wherein degraded distributed connectivity forces
increased traffic through surviving frontal hubs. These results demonstrate that nodal centrality analysis of the Alpha-band
provides a sensitive, non-invasive framework for AD classification. The high accuracy of a single-electrode feature offers
significant potential for low-cost early-stage clinical screening protocols.
