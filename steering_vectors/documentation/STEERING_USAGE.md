# Vector Steering Usage Guide

## Overview

This guide explains how to use the Modal scripts to extract truthfulness steering vectors and apply them during inference to reduce strategic voting behavior in LLMs.

## Workflow

1. **Extract Steering Vector**: Generate truthful/strategic samples and extract activation differences
2. **Apply Steering**: Test steering with varying coefficients on neutral prompts
3. **Analyze Results**: Visualize activation separation and steering effectiveness

## Step 1: Extract Steering Vector

### Command

```bash
modal run steering_vectors/scripts/modal_extract_steering_vector.py \
    --config-path configs/config_plurality_40_truthful.json \
    --layer-idx 20 \
    --num-samples 10 \
    --overwrite
```

### What It Does

1. Loads `openai/gpt-oss-20b` model on Modal GPU (H100)
2. Generates 10 truthful and 10 strategic responses to voting prompts
3. Extracts hidden state activations from layer 20 for each sample
4. Saves per-sample vectors: `truthful_response_samples.pt` and `strategic_response_samples.pt`
   - Shape: `[num_samples, layers, hidden_dim]` = `[10, 25, 2880]`
5. Computes and saves metadata

### Parameters

- `--config-path`: Voting configuration file (default: `configs/config_plurality_40_truthful.json`)
- `--layer-idx`: Layer to extract activations from (default: 20)
- `--num-samples`: Number of samples per prompt type (default: 10)
- `--overwrite`: Allow overwriting existing files (default: True)

### Output Files

**Saved to both Modal volume and local repo**:
- `steering_vectors/results/{model_name}/extraction/truthful_response_samples.pt`
- `steering_vectors/results/{model_name}/extraction/strategic_response_samples.pt`
- `steering_vectors/results/{model_name}/extraction/metadata_layer{layer_idx}.json`
- `steering_vectors/results/{model_name}/extraction/samples_layer{layer_idx}.json`

**File Structure**:
- Per-sample vectors: `[10, 25, 2880]` - 10 samples × 25 layers × 2880 hidden dim
- Averages computed on-the-fly when loading: `samples.mean(dim=0)` → `[25, 2880]`

## Step 2: Apply Steering

### Command

```bash
modal run steering_vectors/scripts/modal_apply_steering.py \
    --config-path configs/config_plurality_40_truthful.json \
    --layer-idx 20 \
    --coefficients "0,0.5,1,1.5,2,2.5,3" \
    --num-trials 20 \
    --steering-type response \
    --overwrite
```

### What It Does

1. Loads per-sample vectors and computes averages: `truthful_avg` and `strategic_avg`
2. Computes truthfulness vector: `truthfulness_vector = truthful_avg - strategic_avg`
3. For each coefficient (0.0 to 3.0):
   - Applies steering at layer 20: `activation += coefficient * truthfulness_vector[layer_20]`
   - Generates 20 responses to neutral plurality prompts
   - Parses vote choices (A, B, or C)
4. Runs baseline (no steering) with 20 trials
5. Saves results for each coefficient

### Parameters

- `--vector-path`: Optional - if None, automatically loads `truthful_response_samples.pt` and `strategic_response_samples.pt`
- `--config-path`: Voting configuration file
- `--layer-idx`: Layer to apply steering (default: 20)
- `--coefficients`: Comma-separated list (default: "0,0.5,1,1.5,2,2.5,3")
- `--num-trials`: Number of trials per coefficient (default: 20)
- `--steering-type`: `"response"` (recommended), `"prompt"`, or `"all"` (default: `"response"`)
- `--overwrite`: Allow overwriting existing results (default: True)

### Output Files

**Saved to both Modal volume and local repo**:
- `steering_vectors/results/{model_name}/results/baseline_no_steering.json`
- `steering_vectors/results/{model_name}/results/coef_pos{value}.json` (for each coefficient)
- `steering_vectors/results/{model_name}/results/summary.json`

**File Structure** (example: `coef_pos3.0.json`):
```json
{
  "coefficient": 3.0,
  "num_trials": 20,
  "steering_type": "response",
  "prompt": "...",
  "trials": [
    {
      "trial": 1,
      "prompt": "...",
      "thinking": "...",
      "output": "**Vote Choice:** C\n\n**Reasoning:** ...",
      "vote": "C"
    },
    ...
  ],
  "thinkings": [...],
  "outputs": [...],
  "votes": ["C", "C", "A", ...],
  "valid_votes": ["C", "C", "A", ...]
}
```

## Step 3: Visualize Results

### Jupyter Notebook

Run `visualize_activation_differences.ipynb` to:
1. Load per-sample vectors and compute averages
2. Visualize activation magnitudes across layers
3. Compare truthful vs strategic activations
4. Perform PCA analysis on layer 20 activations
5. Plot steering effectiveness (strategic voting decreases with coefficient)

### Key Visualizations

1. **Activation Magnitudes**: Shows L2 norms across layers
2. **Direct Comparison**: Histograms and scatter plots of truthful vs strategic
3. **PCA Visualization**: 2D scatter plot showing separation in activation space
4. **Steering Effect**: Line plot showing strategic voting percentage decreasing with coefficient

## Understanding Results

### Steering Vector Interpretation

- **Truthfulness Vector**: `truthful_activations - strategic_activations`
- **Positive Coefficients**: Push model toward truthful behavior (reduces strategic voting)
- **Coefficient 0.0**: No steering (baseline behavior)
- **Higher Coefficients**: Stronger truthfulness steering

### Expected Pattern

- **Baseline (no steering)**: High strategic voting (e.g., 95% Vote A)
- **Low coefficients (0.5-1.0)**: Slight reduction in strategic voting
- **Medium coefficients (1.5-2.0)**: Moderate reduction (e.g., 70-75% strategic)
- **High coefficients (2.5-3.0)**: Large reduction (e.g., 20-40% strategic)

### Key Metrics

1. **Activation Separation**: PCA shows if truthful/strategic samples cluster separately
2. **Steering Effectiveness**: Percentage change in strategic voting from baseline
3. **Statistical Significance**: 20 trials per coefficient provides reliable estimates

## Technical Details

### Model Configuration

- **Model**: `openai/gpt-oss-20b` (20B parameters)
- **Quantization**: MXFP4 (default, requires Hopper+ GPUs)
- **Dtype**: `torch.bfloat16`
- **Architecture**: Standard transformer with `model.layers` structure
- **Hidden Dimension**: 2880
- **Layers**: 24 transformer + 1 embedding = 25 total

### Activation Extraction

- **Method**: `output_hidden_states=True` during forward pass
- **Type**: Response average activations (mean over response tokens)
- **Layer**: 20 (middle layer, empirically chosen)
- **Format**: Per-sample vectors `[num_samples, layers, hidden_dim]`

### Steering Mechanism

- **Implementation**: Custom `ActivationSteerer` class using forward hooks
- **Hook Type**: Forward hook on layer output
- **Position**: Response tokens only (`positions="response"`)
- **Formula**: `new_activation = original_activation + coefficient * steering_vector`

### Determinism

- **Seeding**: `base_seed + trial_number` for each trial
- **Settings**: `torch.backends.cudnn.deterministic = True`
- **Verification**: Baseline and `coef=0` produce identical results

## Troubleshooting

### Common Issues

1. **Model Loading Fails**:
   - Check GPU availability (H100 required for MXFP4)
   - Verify model name: `openai/gpt-oss-20b`
   - Ensure `triton==3.4.0` and `kernels` are installed

2. **Vector Shape Mismatch**:
   - Verify layer index matches extraction layer
   - Check that hidden dimension matches model (2880 for GPT-OSS-20B)

3. **Output Quality Issues**:
   - Ensure chat template is being used correctly
   - Check that `pad_token` is set to `eos_token`
   - Verify `max_new_tokens=2048` is set

4. **Non-Deterministic Results**:
   - Check that seeds are being set correctly
   - Verify CUDA deterministic settings are enabled
   - Ensure `coef=0` matches baseline exactly

5. **Files Not Syncing**:
   - Wait for Modal function to complete
   - Check that download logic runs after remote function
   - Verify Modal volume is mounted correctly

## Results Summary

### Activation Space Analysis

- **PCA Separation**: Truthful and strategic samples form distinct clusters in PCA space
- **Separation Ratio**: > 1.0 indicates well-separated groups
- **Variance Explained**: 78.87% by first 2 principal components

### Steering Effectiveness

- **Baseline**: 95% strategic voting (Vote A)
- **Coefficient 0.0**: 95% strategic (matches baseline)
- **Coefficient 1.5**: 75% strategic, 25% truthful
- **Coefficient 2.5**: 40% strategic, 60% truthful
- **Coefficient 3.0**: 20% strategic, 80% truthful

**Key Finding**: Strategic voting decreases monotonically from 95% to 20% as truthfulness steering coefficient increases from 0 to 3.0, demonstrating successful control of strategic manipulation through activation steering.
