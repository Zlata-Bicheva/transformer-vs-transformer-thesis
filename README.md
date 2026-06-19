# Transformer-vs-Transformer

This repository contains the code for the bachelor’s thesis:

**_Transformer-vs-Transformer: a comparative robustness study of transformer-based deception classifiers under heterogeneous black-box adversarial paraphrasing strategies_**

The project investigates whether **semantic-preserving black-box adversarial paraphrases** can successfully fool transformer-based deception classifiers, and whether such attacks **transfer across transformer families**.

The experimental setup is based on a **3 × 3 attacker–victim matrix** using three architecture families:

- Encoder-only (RoBERTa)
- Decoder-only (GPT-2)
- Encoder–decoder (FLAN-T5)

All experiments are conducted on the **OpSpam deceptive hotel review corpus**. 

---

## Project Overview

The repository provides a full experimental pipeline for:

1. **Fine-tuning deception detection models (victims)**
2. **Generating adversarial paraphrases (attack pipeline)**
3. **Evaluating attack success and robustness**
   - Attack Success Rate (ASR)
   - Transferability across models
   - Psycholinguistic feature analysis 

---

## Requirements

Install the main dependencies:

```bash
pip install torch transformers sentence-transformers datasets pandas numpy tqdm matplotlib seaborn scipy statsmodels kagglehub
```

---

## Repository Structure

```text
.
├── saved_models/
├── results/
├── analysis_outputs/
├── liwc_analysis/
├── result_analysis_scripts/
│   ├── asr.py
│   ├── psucholinguistin_features.py
│   └── transferability.py
├── fine_tune_models.py
├── attack_pipeline.py
└── README.md
```

> Adjust script names in commands if your local filenames differ.

---

# 1. Fine-Tuning Victim Models

This step trains the deception classifiers used as **victim models**:

- **GPT-2** (decoder-only)
- **RoBERTa** (encoder-only)
- **FLAN-T5** (encoder–decoder)

These models are fine-tuned on the OpSpam dataset for binary classification:

- `truthful = 0`
- `deceptive = 1` 

### Run

```bash
python fine_tune_models.py
```

### Output

Saved models will be stored in:

```text
./saved_models/decoder_only/
./saved_models/encoder_only_roberta/
./saved_models/flan_t5_deception/
```

---

# 2. Attack Pipeline

This script runs **black-box adversarial attack experiments**.

## Key Characteristics

- **Strict black-box setting** (no access to model internals)
- **Query-budgeted iterative attack loop**
- **Semantic similarity constraint**
- Evaluation of **cross-model transferability**

The goal is to rewrite **deceptive reviews** so they are classified as **truthful**, while preserving meaning.

### Run

```bash
python attack_pipeline.py
```

### Output

Each attacker–victim pair produces one CSV:

```text
./results/<attacker>_<victim>.csv
```

### Example Files

```text
results/gpt2_encoder_only.csv
results/gpt2_decoder_only.csv
results/gpt2_encoder_decoder.csv
results/t5_encoder_only.csv
results/t5_decoder_only.csv
results/t5_encoder_decoder.csv
results/bert_encoder_only.csv
results/bert_decoder_only.csv
results/bert_encoder_decoder.csv
```

> **Important:** Update any hardcoded dataset paths before running.

---

# 3. Result Analysis

All evaluation scripts are located in:

```text
result_analysis_scripts/
```

---

## 3.1 Attack Success Rate — `asr.py`

Computes **Attack Success Rate (ASR)** along with:

- Confidence intervals
- Statistical significance tests

### Run

```bash
python result_analysis_scripts/asr.py
```

### Output

```text
analysis_outputs/table_01_asr_summary.csv
analysis_outputs/table_03_significance_vs_chance.csv
analysis_outputs/fig_01_asr_<attacker>.png
```

---

## 3.2 Psycholinguistic Features — `psycholinguistic_features.py`

Performs analysis of **linguistic feature changes** between:

- Original reviews
- Successful adversarial paraphrases
- Failed attacks

Focuses on LIWC-inspired features such as:

- Perceptual language
- Affective intensity
- Self-reference
- Certainty markers 【1-873e9a】

### Run

```bash
python result_analysis_scripts/psycholinguistic_features.py \
    --results_dir ./results \
    --output_dir ./liwc_analysis \
    --opspam_csv path/to/deceptive-opinion.csv
```

### Output

```text
liwc_analysis/truthful_reference_medians.csv
liwc_analysis/comparison_features.csv
liwc_analysis/statistical_results.csv
liwc_analysis/trajectory_categorical_bar.png
liwc_analysis/stacked_feature_comparison.png
liwc_analysis/attacker_victim_heatmap.png
```

---

## 3.3 Transferability — `transferability.py`

Evaluates whether adversarial examples that fool one model also fool others.

### Run

```bash
python result_analysis_scripts/transferability.py
```

### Output

```text
analysis_outputs/table_04_transferability.csv
analysis_outputs/fig_04_transferability_<attacker>.png
```

---

# Recommended Workflow

Run the full pipeline in the following order:

1. **Fine-tune models**
   ```bash
   python fine_tune_models.py
   ```

2. **Run attack experiments**
   ```bash
   python attack_pipeline.py
   ```

3. **Run analysis**
   ```bash
   python result_analysis_scripts/asr.py
   python result_analysis_scripts/transferability.py
   python result_analysis_scripts/psucholinguistin_features.py \
       --opspam_csv path/to/deceptive-opinion.csv
   ```

---

## Notes

- Ensure trained models exist in `./saved_models/` before running attacks.
- The psycholinguistic script requires the original **OpSpam dataset** stored locally. The dataset can be found via this link: https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus.


---

## Thesis Reference

**Zlata Bicheva**  
*Transformer-vs-Transformer: a comparative robustness study of transformer-based deception classifiers under heterogeneous black-box adversarial paraphrasing strategies*  
Eindhoven University of Technology, June 2026. 