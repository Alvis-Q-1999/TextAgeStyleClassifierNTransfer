
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
**StyleRemover**

- **Input**: A styled sentence .
- **Output**: A neutral style sentence.
- **Prompt Format**:  
  `"transfer from [input_age] to neutral style: [styled_sentence]"`

**StyleApplier**

- **Input**: A neutral style sentence .
- **Output**: A new styled sentence.
- **Prompt Format**:  
  `"transfer from neutral style to [target_age] style: [neutral_sentence]"`


#### Supervised Setting

- **Input**: A sentence in style A.
- **Target**: A sentence in style B describing the same object.
- **Output**: A rewrited sentence in style B.
- **Prompt Format**:  
  `"transfer from [input_age] to [target_age] style: [style-A sentence]"`


## Dataset

The dataset includes:
- `input.sentences`: The original sentence (neutral or stylized).
- `persona.age`: The associated age group (e.g., "18-24", "25-34", "35-44").
- `output.sentences`: The neutral-styled sentence.

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

