## Installment

### Requirements

### create conda environment

```python
conda create -n evophage_env python=3.12.3
conda activate evophage_env
```

### install required package

```python
pip install torch==2.3.0

pip install tqdm numpy beartype biopython einops

pip install MEGABYTE-pytorch==0.2.1
```

## Model Architecture

### Model Specifications

- Total Parameters: 36513286
- Hierarchical Levels: 2-stage Transformer
- Maximum Sequence Length: 8192
- Vocabulary Size: 6 (A, T, C, G, **,#)

### Layer Configuration

```text
Stage 1: 512 dimensions, 8 layers, 256 sequence length
Stage 2: 256 dimensions, 8 layers, 32 sequence length
```

## Quick Start

### Training

```python
from megaDNA.megadna import MEGADNA


model = MEGADNA(
    num_tokens=6,
    dim=(512, 256),
    depth=(8, 8),
    max_seq_len=(256, 32)
)


model = train_megadna_model(
    custom_sequences=your_sequences,
    output_dir='megadna_trained_model',
    epochs=45,
    batch_size=1,
    lr=1e-4
)
```

### Hyperparameters

- Batch Size: 1
- Learning Rate: 1e-4
- Sequence Length: 8192
