# Adaptive Logit Adjustment (ALA) for Vision-Language Models

**ICLR 2026**

This repository provides a unified framework for **debiasing** vision-language models (VLMs) via **Adaptive Logit Adjustment**. It supports **LLaVA**, **PaliGemma**, and **Qwen2.5-VL** on two settings:

- **FACET** (gender/race): reduce misgendering and race-related bias in person descriptions.
- **Counterfactual Social / Toxicity**: steer generations away from toxic language using a target signal (e.g. non-toxicity, s_scale = -1).

The pipeline: (1) **Image signal**: detect the desired attribute from the image (e.g. gender via image classifier) or set it to zero (neutralization) or to non-toxicity (-1). (2) **Logit mitigation**: adjust logits for tokens related to gender or toxicity using a **text classifier** and token importance (β), pushing outputs in the correct direction.

---

## Structure

```
Adaptive-Logit-Adjustment-ICLR2026/
├── README.md
├── requirements.txt
├── LICENSE
├── ala/                          # Core library
│   ├── __init__.py
│   ├── model.py                  # SimpleTransformerClassifier (text classifier)
│   ├── utils.py                  # decide_gender, load_and_normalize_beta, evaluation
│   ├── llava_model.py            # LLaVA + LogitAdjustmentProcessor, DEAR, etc.
│   ├── paligemma_model.py        # PaliGemma + logit adjustment (gender/toxicity)
│   └── qwen_model.py             # Qwen2.5-VL + logit adjustment
├── tasks/
│   ├── facet/                    # FACET dataset (gender / race debiasing)
│   │   └── main.py
│   ├── counterfactual/           # SocialCounterfactuals (toxicity debiasing)
│   │   ├── main.py
│   │   └── evaluate_ci.py
│   └── judge/                     # Accuracy evaluation (gender/race)
│       └── main.py
├── scripts/
│   ├── run_facet.sh
│   ├── run_counterfactual.sh
│   └── run_judge.sh
├── nlp_classification/            # Place classifier checkpoints and token bias here
│   └── README.md
└── config.py                     # Paths: DATA_DIR, NLP_CLASSIFICATION_DIR
```

---

## Models and Datasets

| Model | ID |
|-------|-----|
| LLaVA | `llava-hf/llava-1.5-7b-hf` |
| PaliGemma | `google/paligemma-3b-mix-224` |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` (or 7B) |

| Dataset | Task | Signal |
|---------|------|--------|
| **FACET** | Gender / race debiasing | Image classifier (alignment) or 0 (neutralization) |
| **Counterfactual Social** | Toxicity | s_scale = -1 (non-toxicity) |

---

## Setup

### 1. Environment

```bash
cd Adaptive-Logit-Adjustment-ICLR2026
pip install -r requirements.txt
```

### 2. Data and classifiers

- **FACET**:
  - Download the **FACET** dataset (images + annotations) from the original authors / project page (Kumar et al., 2021).  
  - Create the following structure under `DATA_DIR` (default `data/` – configurable in `config.py` or via `ALA_DATA_DIR`):
    - `facet/annotations/annotations.csv`
    - `facet/new_annotations.csv` (optional, if you have a preprocessed split)
    - `facet/image/*.jpg` (all FACET person images)
  - The tasks `tasks.facet.main` and `tasks.judge.main` expect exactly this layout.
- **FairFace (for image embeddings)**:
  - Download the **FairFace** dataset (e.g. from the official GitHub/Kaggle release).
  - Place it under `DATA_DIR/fairface/` with the same CSV layout as in the original FairFace code:
    - `fairface_label_train.csv`, `fairface_label_val.csv`
    - image files referenced by the CSVs.
  - The embedding scripts below will look for FairFace at `DATA_DIR/fairface/`.
- **Counterfactual (SocialCounterfactuals)**:
  - The toxicity code uses the Hugging Face dataset  
    **`Intel/SocialCounterfactuals`**  
    and will download it automatically via `datasets.load_dataset("Intel/SocialCounterfactuals")` (see `tasks/counterfactual/main.py`).
- **Classifiers and token bias**:
  - This repo already includes **pretrained** LLaVA / PaliGemma gender, race, and toxicity models, plus their importance JSONs, under `nlp_classification/`.
  - For Qwen, run the training scripts in `nlp_classification/` to create:
    - Gender: `gender_model_qwen_pytorch_generated/`, `importance_dict_qwen_pytorch_generated.json`
    - Toxicity: `toxicity_model_qwen_pytorch/`, `importance_toxicity_dict_qwen_pytorch.json`
    - Race: `race_model_qwen_pytorch_generated/`, `importance_race_dict_qwen_pytorch_generated.json`
  - You can also point `NLP_CLASSIFICATION_DIR` (or `ALA_NLP_CLASSIFICATION_DIR`) to an external folder with the same layout.

See `nlp_classification/README.md` for detailed training commands and file layout.

### 3. Embeddings (for SFID / CLIP-CLIP / DEAR)

- Image embeddings (e.g. for image gender classifier): `embedding/fairface_{model}_train.pt` under the task directory or path set in config.
- Decoder embeddings for SFID/CLIP-CLIP: `embedding/fairface_{model}_train_decoder.pt`.
- For LLaVA and PaliGemma on FACET, you can generate FairFace-based embeddings directly in this repo:
  - **LLaVA (FACET)**:
    ```bash
    cd Adaptive-Logit-Adjustment-ICLR2026
    python -m tasks.facet.llava_fairface_embedding
    ```
    This script expects `DATA_DIR/fairface/` to contain `fairface_label_train.csv` and `fairface_label_val.csv`, and will write:
    - `tasks/facet/embedding/fairface_llava_train.pt`
    - `tasks/facet/embedding/fairface_llava_train_decoder.pt`
  - **PaliGemma (FACET)**:
    ```bash
    cd Adaptive-Logit-Adjustment-ICLR2026
    python -m tasks.facet.paligemma_fairface_embedding
    ```
    This script writes:
    - `tasks/facet/embedding/fairface_paligemma_train.pt`
    - `tasks/facet/embedding/fairface_paligemma_train_decoder.pt`
    - `tasks/facet/embedding/fairface_paligemma_val_decoder.pt`


---

## Usage

### FACET (gender or race)

```bash
# Gender debiasing, logit mode, LLaVA
python -m tasks.facet.main --model llava --mode logit --gpu_id 0 --lam 5.0 --debiasing_target gender

# Gender, Qwen, neutralization
python -m tasks.facet.main --model qwen --mode logit --gpu_id 0 --lam 5.0 --debiasing_target gender --neutral

# Race debiasing
python -m tasks.facet.main --model qwen --mode logit --gpu_id 0 --lam 5.0 --debiasing_target race
```

### Counterfactual (toxicity)

```bash
# Toxicity debiasing (s_scale = -1)
python -m tasks.counterfactual.main --model qwen --mode logit --gpu_id 0 --lam 1.0 --target gender
python -m tasks.counterfactual.main --model paligemma --mode logit --gpu_id 0 --lam 7.0
```

### Judge (accuracy)

```bash
python -m tasks.judge.main --model qwen --mode logit --gpu_id 0 --lam 0.9 --debiasing_target gender
```

### Debiasing modes

- **naive**: No debiasing.
- **logit**: Adaptive logit adjustment (image signal + text classifier + token β).
- **prompt**: Prompt modification based on image attribute (e.g. “Describe this photo of a man/woman” or “Do not include toxicity”).
- **sfid** / **clipclip**: Require decoder embeddings; see legacy `vqa_bias`/`vqa_bias_qwen` for scripts.
- **dear**: Requires trained DEAR adaptor; see legacy repos for training scripts.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{adaptive-logit-adjustment-iclr2026,
  title     = {Adaptive Logit Adjustment for Debiasing Vision-Language Models},
  booktitle = {ICLR},
  year      = {2026},
}
```

---

## License

MIT. See `LICENSE`.
