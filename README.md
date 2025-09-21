
# Age Style Transfer with T5 and Classifier

This project explores **language style transfer across age groups** using Transformer-based models, primarily focusing on T5. It provides both **supervised** and **unsupervised** training pipelines to perform age-aware rewriting of text, along with a **style classifier** for evaluation.

## Overview

The goal of this project is to **transform the writing style of a sentence to match a target age group**, while preserving the semantic meaning. We approach this task using two main components:

1. **Style Classifier** – A BERT-based classifier to evaluate the age style of generated text.
2. **T5-based Style Transfer** – A sequence-to-sequence model that performs age-style rewriting under both supervised and unsupervised training settings.

## Components

### 1. Style Classifier

We fine-tune a BERT-based model for **age group classification**, using textual samples labeled by age. The classifier helps:
- Evaluate how well the generated sentences reflect the target style.
- Provide an auxiliary loss during training (in later versions).

### 2. T5 Style Transfer

#### Unsupervised Setting

- **Input**: A styled sentence + source/target age.
- **Output**: The same sentence (self-supervised reconstruction).
- **Prompt Format**:  
  `"transfer from [input_age] to [target_age]: [styled_sentence]"`

- **Goal**: Teach the model to rewrite in the target style, even without explicit paired data.

#### Supervised Setting

- **Input**: Neutral sentences describing an object/event.
- **Output**: The same description rewritten in the style of a specific age group.
- **Paired Training**: Different age groups describing the *same thing* are treated as supervised pairs.


## Dataset

The dataset includes:
- `input.sentences`: The original sentence (neutral or stylized).
- `persona.age`: The associated age group (e.g., "teen", "middle", "senior").
- `output.sentences`: The target-styled sentence.

In the **supervised setting**, paired samples are created by matching different age groups’ descriptions of the same item. In the **unsupervised setting**, the model is trained to paraphrase within and across age styles using prompt-based guidance.


## Technologies

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Datasets](https://huggingface.co/docs/datasets/)
- [scikit-learn](https://scikit-learn.org/) (for metrics)

## Structure

```bash
├── Classifier/
├── Transfer_T5_gpu/
├── pastel_data_preprocessing.ipynb
└── README.md
```

