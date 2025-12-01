# Vector Steering Setup and Methodology

## Overview

This project implements activation steering to study and control strategic manipulation in LLM voting behavior. We extract a "truthfulness" steering vector by comparing activations from truthful vs strategic responses, then apply this vector during inference to reduce strategic voting behavior.

## Methodology

### 1. Vector Extraction

**Goal**: Extract a steering vector that represents the difference between truthful and strategic activation patterns.

**Process**:
1. Generate 10 samples each of truthful and strategic responses to voting prompts
2. Extract hidden state activations from layer 20 for each sample
3. Compute per-sample vectors: `[num_samples, layers, hidden_dim]` shape
4. Compute average activations: `truthful_avg = mean(truthful_samples)` and `strategic_avg = mean(strategic_samples)`
5. Compute truthfulness vector: `truthfulness_vector = truthful_avg - strategic_avg`

**Key Technical Details**:
- **Model**: `openai/gpt-oss-20b` (20B parameters, MXFP4 quantized)
- **Layer**: Layer 20 (middle layer, empirically chosen for best separation)
- **Activation Type**: Response average activations (mean over all response tokens)
- **Vector Shape**: `[layers, hidden_dim]` = `[25, 2880]` (24 transformer layers + 1 embedding layer)
- **Storage**: Per-sample vectors saved as `truthful_response_samples.pt` and `strategic_response_samples.pt` for analysis flexibility

**Implementation**:
- Uses `output_hidden_states=True` during forward pass to extract activations
- Activations extracted from response tokens only (not prompt tokens)
- All samples generated with deterministic seeds for reproducibility
- GPT-OSS Harmony format handled with chat template and response parsing

### 2. Vector Application

**Goal**: Apply the truthfulness vector during inference to steer model behavior.

**Process**:
1. Load per-sample vectors and compute averages on-the-fly
2. Compute difference vector: `truthfulness_vector = truthful_avg - strategic_avg`
3. For each steering coefficient (0.0 to 3.0 in 0.5 steps):
   - Apply vector at layer 20: `activation += coefficient * truthfulness_vector[layer_20]`
   - Generate responses to neutral plurality voting prompts
   - Parse vote choices from outputs
   - Run 20 trials per coefficient for statistical significance

**Key Technical Details**:
- **Steering Position**: Response tokens only (`positions="response"`)
- **Steering Mechanism**: Forward hook that adds `coeff * vector` to layer output
- **Coefficients Tested**: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 (truthfulness steering)
- **Baseline**: 20 trials with no steering (coefficient = 0)
- **Deterministic**: Each trial uses `base_seed + trial_number` for reproducibility

**Implementation**:
- Uses custom `ActivationSteerer` class with forward hooks
- Context manager pattern for hook registration/cleanup
- Even with `coef=0`, hook is registered (multiplies by 0, ensuring identical baseline)
- GPT-OSS chat template with `reasoning_effort="low"` for consistent output format

## Technical Considerations

### Model Compatibility

**GPT-OSS-20B Specifics**:
- Uses Harmony response format with channels (`analysis`, `commentary`, `final`)
- Requires chat template formatting: `tokenizer.apply_chat_template()`
- MXFP4 quantization requires `triton==3.4.0` and `kernels` packages
- Uses `bfloat16` dtype (not `float16`)
- `pad_token` must be set to `eos_token` for proper tokenization

**Architecture Compatibility**:
- Standard transformer architecture with `model.layers` structure
- Layer detection uses automatic path finding across common transformer architectures
- Hidden dimension: 2880
- Number of layers: 24 transformer layers + 1 embedding layer

### Activation Extraction

**Why Response Activations?**:
- Response tokens contain the model's decision-making process
- Prompt activations may be contaminated with input formatting
- Response average (mean over all response tokens) captures overall response pattern

**Why Layer 20?**:
- Empirically chosen based on activation magnitude analysis
- Middle layers often contain semantic representations
- Layer 20 showed good separation between truthful and strategic patterns

### Vector Computation

**Why Per-Sample Storage?**:
- Enables PCA analysis to visualize sample-level clustering
- Allows computing averages with different methods if needed
- Provides flexibility for future analysis

**Why Truthfulness Vector (truthful - strategic)?**:
- Positive coefficients push toward truthful behavior
- Negative coefficients would push toward strategic behavior
- Intuitive interpretation: adding truthfulness reduces strategic voting

## Troubleshooting and Challenges

### Initial Challenges

1. **Model Output Quality Issues**:
   - **Problem**: GPT-OSS outputs were rambling, incomplete, not following format
   - **Solutions**:
     - Implemented proper chat template with `apply_chat_template()`
     - Set `pad_token` and `pad_token_id` correctly
     - Added `max_new_tokens=2048` to allow full generation
     - Used Harmony format parsing to extract final output

2. **Dtype Mismatch**:
   - **Problem**: `RuntimeError: expected scalar type Half but found BFloat16`
   - **Solution**: Changed `torch_dtype` to `dtype` and used `torch.bfloat16` for GPT-OSS

3. **Deterministic Generation**:
   - **Problem**: Outputs varied between baseline and `coef=0` runs
   - **Solution**: Comprehensive seeding (`torch`, `random`, `numpy`, CUDA) + deterministic CUDA settings

4. **File Syncing**:
   - **Problem**: Files not appearing in local repo
   - **Solution**: Explicit download logic using `results_vol.read_file()` after remote function completes

5. **Quantization Support**:
   - **Problem**: MXFP4 quantization not working
   - **Solution**: Added `triton==3.4.0` and `kernels` to Modal image `pip_install`

6. **Reasoning Level Parameter**:
   - **Problem**: `reasoning_effort` parameter not supported in some Transformers versions
   - **Solution**: Used `reasoning_effort` in `apply_chat_template()` (may raise error in older versions)

### Current Configuration

- **Model**: `openai/gpt-oss-20b`
- **Quantization**: MXFP4 (default for GPT-OSS)
- **Dtype**: `torch.bfloat16`
- **Layer**: 20
- **Samples**: 10 per type (truthful/strategic)
- **Trials**: 20 per coefficient
- **Coefficients**: 0.0 to 3.0 in 0.5 steps
- **Reasoning Level**: "low" (via `reasoning_effort` parameter)

## Results Summary

### Key Findings

1. **Activation Space Separation**:
   - PCA analysis on layer 20 activations shows clear separation between truthful and strategic samples
   - Separation ratio > 1.0 indicates well-separated clusters in PCA space
   - 78.87% of variance explained by first 2 principal components

2. **Steering Effectiveness**:
   - Baseline: 95% strategic voting (Vote A), 5% truthful voting (Vote C)
   - At coefficient 0.0: Same as baseline (95% strategic)
   - At coefficient 1.5: 75% strategic, 25% truthful
   - At coefficient 2.5: 40% strategic, 60% truthful
   - At coefficient 3.0: 20% strategic, 80% truthful
   - **Clear trend**: Strategic voting decreases monotonically as truthfulness steering coefficient increases

3. **Statistical Significance**:
   - 20 trials per coefficient provides sufficient sample size
   - Consistent downward trend in strategic voting across all coefficients
   - Maximum reduction: 75 percentage points (from 95% to 20% strategic voting)

### Interpretation

The results demonstrate that:
1. **Truthful and strategic responses are separable in activation space** - PCA visualization shows distinct clusters
2. **Truthfulness steering successfully reduces strategic voting** - As coefficient increases from 0 to 3.0, strategic voting decreases from 95% to 20%
3. **Steering is effective and controllable** - Linear relationship between coefficient strength and behavioral change

This validates that activation steering can be used as a tool to control strategic manipulation in LLM voting behavior, with the truthfulness vector successfully pushing the model toward more truthful (less strategic) voting decisions.
