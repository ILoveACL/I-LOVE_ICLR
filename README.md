# NuAlign: A Numerical Distance Perception Alignment for Large Language Models as Evaluators
Official implementation of **NuAlign**, a two-stage training framework for numerical scoring with autoregressive large language models.

NuAlign is designed for automatic evaluation tasks in which an LLM directly generates a numerical score. The method first learns the task-specific mapping between answers and score sequences, and then introduces numerical-distance information to modulate likelihood optimization while retaining the native autoregressive scoring interface.



---

## Overview

Standard causal language-model training optimizes token-level cross-entropy, but it does not explicitly distinguish numerical errors by their severity. For example, predicting a score far from the human score is not directly treated as more severe than predicting a nearby score purely because of the numerical distance.

NuAlign addresses this with two stages:

### Stage I — Semantic Score Alignment

Stage I performs parameter-efficient supervised fine-tuning with the standard causal language-model objective. The goal is to first establish a task-aware mapping from the evaluation context to the human score sequence.

In the released code, Stage I:

- formats each example as a scoring prompt;
- masks the prompt portion and supervises the assistant response;
- fine-tunes the backbone with LoRA;
- saves PEFT checkpoints during training.

### Stage II — Numerical Distance Perception

Stage II starts from the Stage-I model and introduces numerical-distance-aware optimization.

For each training example, the current model prediction is reconstructed from the score-region logits. The absolute deviation between the reconstructed score and the human score is then used to modulate the likelihood gradient of the score tokens. The implementation uses

```python
re_loss = torch.exp(en_loss_re - en_loss_re.detach()) * re_adv
```

where `re_adv` is the numerical score deviation. The exponential term has forward value 1 while preserving the desired likelihood-gradient path during backpropagation.

The semantic likelihood loss and the numerical-distance component are then balanced adaptively at the mini-batch level.

---

## Repository Structure

```text
I-LOVE_ICLR/
├── README.md
├── Stage I/
│   ├── train.py
│   ├── trainer.py
│   ├── train_utils.py
│   ├── argument_utils.py
│   ├── inference.py
│   ├── merge_lora.py
│   └── Datasets/
│       ├── data_utils.py
│       └── qa_dataset.py
└── Stage II/
    ├── train.py
    ├── trainer.py
    ├── train_utils.py
    ├── argument_utils.py
    ├── loss.py
    ├── inference.py
    ├── merge_lora.py
    └── Datasets/
        ├── data_utils.py
        └── qa_dataset.py
```

### Main files

| File | Description |
| --- | --- |
| `train.py` | Training entry point. Builds the LoRA configuration, loads the model/tokenizer, and launches the trainer through Hugging Face Accelerate. |
| `trainer.py` | Main training loop. Stage I uses the standard causal LM objective; Stage II implements numerical-distance perception and adaptive weighting. |
| `argument_utils.py` | Command-line arguments for model paths, datasets, batch sizes, LoRA, optimizer settings, and training configuration. |
| `train_utils.py` | Model/tokenizer loading, optional 4-bit quantization, LoRA attachment, device utilities, and random-seed initialization. |
| `Datasets/data_utils.py` | Prompt tokenization, label masking, dataset caching, and deterministic shuffling. |
| `Datasets/qa_dataset.py` | Dataset loading and task-specific prompt construction. |
| `loss.py` | Stage-II token-level causal LM loss used to retain unreduced token losses. |
| `merge_lora.py` | Merges a trained LoRA adapter into its base model and writes a standalone model checkpoint. |
| `inference.py` | Batched greedy inference with vLLM. |

---

## Installation

The released code depends on PyTorch and the Hugging Face training stack. A typical environment can be created with:

```bash
conda create -n nualign python=3.12 -y
conda activate nualign

pip install torch
pip install transformers peft accelerate deepspeed datasets numpy
pip install vllm
```

Optional 4-bit loading uses `bitsandbytes`:

```bash
pip install bitsandbytes
```

Exact package versions may depend on the CUDA/PyTorch environment. The experiments reported in the paper use PyTorch-based training with Hugging Face Transformers, PEFT/LoRA, Accelerate, and vLLM for inference.

---

## Data Format

The code loads JSON/JSONL data through Hugging Face `datasets`.

A minimal example is:

```json
{
  "question": "What is a binary search tree?",
  "answer": "A tree in which the left child is smaller and the right child is larger.",
  "human_score": 4
}
```

Some task configurations additionally use:

```json
{
  "info": "Optional context or source passage",
  "question": "Question text",
  "answer": "Answer to be graded",
  "human_score": 7,
  "example": [
    ["Example answer 1", 8],
    ["Example answer 2", 4]
  ]
}
```

The `example` field is optional. When present, each item is expected to contain an example answer and its score.

### Adapting a new scoring dataset

The released repository contains task-specific prompt examples. Before training on a new dataset, edit:

```text
Stage I/Datasets/qa_dataset.py
Stage II/Datasets/qa_dataset.py
```

to match:

1. the dataset field names;
2. the scoring rubric;
3. the valid score range;
4. the desired output template.

The expected target format used by the current implementation is a short numerical response such as:

```text
Final score: 4
```

or its dataset-specific equivalent.

---

## Training Pipeline

### 1. Stage I: Semantic Score Alignment

Enter the Stage-I directory:

```bash
cd "Stage I"
```

Launch training:

```bash
accelerate launch train.py \
  --train_data /path/to/train.jsonl \
  --eval_data /path/to/dev.jsonl \
  --pre_trained_model /path/to/base_model \
  --output_dir /path/to/stage1_output \
  --lora \
  --rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lr 3e-4 \
  --seed 3407
```

Useful options include:

```text
--train_batch_size
--per_device_batch_size
--max_length
--num_train_epochs
--gradient_checkpointing
--use_bit
--lr_scheduler_type
--num_warmup_steps
--target_modules
```

Stage I uses standard causal language-model supervision after masking the prompt tokens.

---

### 2. Merge the Stage-I LoRA adapter

The Stage-II loader expects a normal causal-LM checkpoint. Therefore, a convenient workflow is to merge the Stage-I LoRA adapter before starting Stage II.

```bash
python merge_lora.py \
  --lora_model /path/to/stage1_output/model.pt.stepXXXX \
  --output_dir /path/to/stage1_merged
```

The merged model is written under:

```text
/path/to/stage1_merged/<base-model-name>-finetune/
```

Use this directory as the Stage-II initialization.

---

### 3. Stage II: Numerical Distance Perception

Enter the Stage-II directory:

```bash
cd "../Stage II"
```

Launch Stage-II training from the Stage-I model:

```bash
accelerate launch train.py \
  --train_data /path/to/train.jsonl \
  --eval_data /path/to/dev.jsonl \
  --pre_trained_model /path/to/stage1_merged/<base-model-name>-finetune \
  --output_dir /path/to/stage2_output \
  --lora \
  --rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lr 3e-4 \
  --seed 3407
```

Stage II computes unreduced token-level cross-entropy and reconstructs the score-region prediction from the current logits. The reconstructed numerical deviation is used as a stop-gradient modulation signal for the likelihood update.

A simplified view of the implementation is:

```text
gold score sequence
        │
        ▼
teacher-forced logits
        │
        ├── token-level causal LM loss
        │
        ▼
argmax score reconstruction
        │
        ▼
numerical deviation |predicted score - human score|
        │
        ▼
distance-modulated likelihood gradient
        │
        ▼
adaptive combination with semantic likelihood
```

---

## Inference

After training, merge the final LoRA adapter if necessary:

```bash
python merge_lora.py \
  --lora_model /path/to/stage2_output/model.pt.stepXXXX \
  --output_dir /path/to/stage2_merged
```

Then run vLLM inference:

```bash
python inference.py \
  --model_dir /path/to/stage2_merged/<model-name>-finetune \
  --data_file /path/to/test.jsonl \
  --output_file /path/to/predictions.jsonl \
  --rank 1 \
  --bsz 32 \
  --seed 3407 \
  --temperature 0 \
  --top_k 1 \
  --top_p 1
```

The output file contains records of the form:

```json
{
  "question": "...",
  "answer": "...",
  "human": 4,
  "llm": "4"
}
```

---

## Reproducibility

The code exposes the random seed through `--seed` and initializes Python, NumPy, PyTorch, Transformers, and accelerator random states.

For the experiments reported in the paper, results are aggregated over multiple random seeds. To reproduce a multi-seed experiment, run the complete Stage-I → Stage-II pipeline independently for each seed.

Example:

```bash
for SEED in 3406 3407 3408
do
    # Stage I
    accelerate launch train.py ... --seed ${SEED}

    # merge Stage I checkpoint

    # Stage II
    accelerate launch train.py ... --seed ${SEED}
done
```

Keep training, validation, and test splits fixed across methods when comparing NuAlign against baselines.

---

## Important Implementation Notes

The current repository is a research-code release and retains several dataset/model-specific constants from the experiments. Users adapting the code should check the following before running a new configuration.

### 1. Prompt templates are dataset-specific

`Datasets/qa_dataset.py` and `inference.py` contain prompt templates and score ranges used by particular experiments. Replace them with the prompt/rubric for the target dataset.

### 2. Token IDs are model/tokenizer-specific

The current preprocessing/training code contains special token IDs such as:

```python
ignore_index = 151643
end_index = 77091
```

These values depend on the tokenizer/model used in the original experiments. If a different model family or tokenizer is used, verify the corresponding special-token and score-region alignment before training.

### 3. Stage-II score parsing is task-specific

`Stage II/trainer.py` extracts the first numerical substring from the reconstructed score span. The current public implementation also contains a fallback constant inherited from a dataset-specific experimental configuration.

When adapting the code to another score range, update this behavior to the target dataset. In particular, verify the parsing rule and valid score range rather than reusing a fixed fallback value across datasets.

### 4. Score-region offsets should be verified

The current implementation identifies score positions using offsets relative to the output template. These offsets are tied to the prompt/tokenization format. If the prompt or tokenizer changes, inspect the tokenized sequence and update the score-region boundaries accordingly.

### 5. Distributed execution

The data pipeline contains explicit `torch.distributed` synchronization calls. The released code was written for the authors' distributed training environment. When adapting the repository to a different number of GPUs, verify the Accelerate/Distributed configuration accordingly.

---

## Method Notes

NuAlign differs from directly attaching a regression head to an LLM. Both stages retain the autoregressive score-generation interface.

The key Stage-II signal is based on the reconstructed **complete score**, rather than independently assigning numerical costs to individual digit tokens. This is particularly relevant when scores contain multiple digits or decimal places, where equal token-level errors can correspond to different errors in the complete numerical value.

The released implementation uses the reconstructed deviation only as a modulation signal; score extraction and numerical parsing are non-differentiable.

---

## Citation

Citation information will be updated after the paper is publicly available.

```bibtex
@inproceedings{nualign2027,
  title     = {NuAlign: A Numerical Distance Perception Alignment for Large Language Models as Automatic Evaluators},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```

---

## Code Release

This repository is released to support reproducibility and inspection of the training procedure described in the paper. The code is currently organized around the authors' experimental pipeline rather than as a general-purpose library.

For issues related to dataset adaptation, tokenizer-specific constants, or reproducing a particular experimental configuration, please open a GitHub issue with:

- model/backbone;
- dataset and score range;
- command used;
- relevant training log;
- error traceback, if any.

