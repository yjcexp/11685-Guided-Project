# Task 1 Report: EEG Classification Baselines

## Overview

Task 1 is a 20-way EEG classification problem. The input is a preprocessed EEG trial with shape `[122, 500]`, and the target is the image category label.  
The current data protocol uses:

- low-speed paradigm only
- all subjects trained jointly
- session-based split
  - 3 sessions for train
  - 1 session for validation
  - 1 session for test

The main goal of this stage was to identify a stable EEG encoder for later cross-modal alignment.


## Models Tried

We compared several model families.

### 1. MLP baseline

The first baseline flattened the EEG trial and applied a small multilayer perceptron.

- input: `[122, 500]`
- flatten to one vector
- 2 hidden layers
- cross-entropy loss

This model was simple but weak. It did not provide a strong representation for later use.


### 2. Temporal CNN baseline

The second baseline used 1D temporal convolutions over the EEG signal.

- input: `[B, C, T]`
- Conv1d over the temporal dimension
- temporal pooling
- linear classifier

This model was more appropriate than the MLP, but its performance was still limited.


### 3. CNN + Transformer variants

We then explored several CNN + Transformer models.

- CNN token encoder
- shared Transformer backbone
- shared classifier or subject-aware classifier

We tested:

- shared-head CNN + Transformer
- subject-specific heads
- subject embedding variants

These models were able to fit training data, especially on small subsets, but they generalized poorly. Validation accuracy stayed close to random chance in most cases. This suggested that the Transformer-based tokenization strategy was not a good inductive bias for this dataset.


### 4. EEGNet baseline

The strongest simple baseline was EEGNet.

- temporal convolution first
- spatial depthwise convolution across channels
- separable convolution
- linear classifier

EEGNet was consistently more stable than the MLP, temporal CNN, and CNN + Transformer variants. This suggests that its inductive bias is better matched to EEG classification.


### 5. EEGNet + residual refinement

Finally, we implemented a refined encoder based on EEGNet:

- EEGNet-style stem
- temporal residual refinement block(s)
- gated temporal pooling
- embedding head
- thin classification head

This model was designed not only for classification, but also to serve as a reusable EEG encoder for later alignment with CLIP-style embeddings.

Under the current reproducible protocol, this residual encoder with the best preprocessing setting gave the strongest result among the currently compared models.


## Normalization Experiments

We compared several normalization schemes.

### A. Per-trial z-score

This was the initial baseline normalization.

- normalize each trial by its own mean and standard deviation

This worked, but it may remove useful amplitude information.


### B. Per-channel train-set statistics

- compute channel-wise mean and standard deviation on the training set
- apply the same statistics to validation and test

This was more principled than per-trial normalization, but it did not become the best setting.


### C. Per-subject per-channel normalization

- compute channel-wise statistics for each subject using training sessions only
- apply them to that subject’s validation and test sessions

This was motivated by cross-session subject variability. It was reasonable, but still not the strongest option.


### D. Demean only

- subtract the channel-wise mean
- do not divide by standard deviation

This gave the best overall behavior among the tested normalization methods.

Interpretation:

- simple baseline drift removal is helpful
- full z-scoring may suppress useful magnitude information


## Time Window Experiments

The raw trial length is 500 time steps, but not all time steps appear equally useful. We therefore tested several windows:

- `0:500`
- `0:300`
- `50:350`
- `100:300`

Among these, `100:300` was the strongest and most stable setting.

Interpretation:

- the full 500-step window likely contains irrelevant or harmful information
- the middle portion of the trial appears to contain the most discriminative signal


## Main Findings

The main findings are:

1. Model choice matters strongly.
   EEGNet-style models are clearly better suited to this dataset than generic MLP, CNN, or CNN + Transformer architectures.

2. More expressive architectures do not automatically help.
   Transformer-based subject-aware models improved training fit, but did not improve validation or test accuracy.

3. Normalization matters.
   Demean-only preprocessing performed better than stronger z-score normalization.

4. Time window selection matters.
   Restricting the input to `100:300` improved results compared with using the full trial.

5. The encoder remains the main bottleneck.
   Classification head changes alone were not enough. The main progress came from improving preprocessing and using a better EEG-specific encoder.


## Result Summary

The table below summarizes the main experimental trends under the current project workflow. Exact values are omitted for runs that were exploratory or not kept as the final reproducible reference. The purpose of the table is to record the comparison direction clearly.

| Model / Setting | Normalization | Time Window | Main Outcome | Summary |
|---|---|---:|---:|---|
| MLP baseline | initial baseline setting | full trial | weak | Too simple for stable EEG representation learning. |
| Temporal CNN baseline | initial baseline setting | full trial | weak | Better than MLP, but still limited. |
| CNN + Transformer (shared head) | several settings | full trial | near random on validation | Could fit training data weakly, but generalized poorly. |
| CNN + Transformer (subject-aware variants) | several settings | full trial | near random on validation | Increased training fit, but mostly increased memorization rather than generalization. |
| EEGNet baseline | `zscore_per_trial` | full trial | strongest early baseline | Best among the simple initial baselines. |
| EEGNet baseline | `demean_only` | `0:500` | improved preprocessing baseline | Better preprocessing than strong z-scoring, but not the best time-windowed setting. |
| EEGNet baseline | `demean_only` | `0:300` | better than full trial | Suggests the full 500-step window includes irrelevant signal. |
| EEGNet baseline | `demean_only` | `50:350` | competitive | Middle portion of the trial is more informative than the full window. |
| EEGNet baseline | `demean_only` | `100:300` | best window among EEGNet baselines | Best temporal window found in the current reproducible protocol. |
| EEGNet + residual encoder | `demean_only` | `100:300` | current best reproducible model | Best result under the current frozen codebase and comparison protocol. |


## Current Best Reproducible Setting

Under the current reproducible codebase and protocol, the strongest setting is:

- model: EEGNet residual encoder
- normalization: demean only
- time window: `100:300`

This is the version we currently keep as the best Task 1 model under the frozen protocol.


## Implications for Later CLIP Alignment

For later retrieval and CLIP alignment, the key takeaway is not only the final classification score, but also the encoder design:

- EEG-specific inductive bias is important
- simple preprocessing can outperform stronger normalization
- focusing on a more informative time window is beneficial
- the encoder should output a reusable embedding rather than only logits

Therefore, the Task 1 study supports using an EEGNet-style encoder with a reusable embedding interface for later multimodal alignment.


## Limitations

- Some earlier exploratory runs achieved better numbers, but those runs were not preserved under a fully frozen and reproducible protocol, so they are not treated as final evidence.
- Generalization across sessions remains difficult.
- Subject variability is still a major challenge.


## Conclusion

The Task 1 experiments show that EEG classification benefits more from a strong EEG-specific inductive bias than from simply increasing architectural complexity. EEGNet-based encoders were the most reliable family. After testing multiple normalization schemes and time windows, the best reproducible setup used demean-only preprocessing and the `100:300` temporal window. This provides a practical encoder foundation for the next stage of cross-modal alignment.
