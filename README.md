# **B**enefit **E**stim**a**tion with **U**ncertain**ty**

This repository contains the code accompanying the paper "[Uncertainty-Aware Benefit Estimation: A Reliable Framework for Index Tuning.](./Beauty.pdf)" which is currently under submission to VLDB 2025.

## Project Overview

The folder `index_selection_evaluation/ML/AImeetsAI` includes various uncertainty-aware models based on the paper "AI Meets AI: Leveraging Query Executions to Improve Index Recommendations." These models include:

- AMA-BNN
- AMA-MCD-all
- AMA-MCD-last
- AMA-MCD-dpp
- AMA-Ensemble
- AMA-AE
- AMA-AE-MCD

## Integration

The script `index_selection_evaluation/selection/ProcessedWorkloadBeauty.py` demonstrates how to integrate the BE-UQ models into the index tuning process.

## References

- **Uncertainty-Aware Benefit Estimation: A Reliable Model for Index Tuning**
- **Refactoring Index Tuning Process with Benefit Estimation**
- **AI Meets AI: Leveraging Query Executions to Improve Index Recommendations**
- **Magic mirror in my hand, which is the best in the land?: an experimental evaluation of index selection algorithms**

For detailed usage and implementation, please refer to the individual scripts and the accompanying documentation within the repository.
