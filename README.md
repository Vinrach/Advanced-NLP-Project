<div align="center">

# Neural Word Embeddings with Skip-Gram & Negative Sampling

### Learning Semantic Word Representations from Context

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/NLTK-NLP-154F5C?style=for-the-badge">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-Evaluation-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
</p>

</div>

---

## Overview

This project implements a **Skip-Gram neural word embedding model** inspired by
the work of Mikolov et al. on distributed representations of words and phrases.

The model learns dense vector representations of words by using surrounding
context to predict target-context relationships.

The implementation explores:

- Skip-Gram architecture
- Negative sampling
- Frequent-word subsampling
- Context-window based training
- Dense word embeddings
- Cosine similarity
- Word analogy reasoning
- Vector compositionality
- Embedding visualization using PCA and t-SNE

The project was developed as part of a Master's-level Advanced NLP project.

---

## Project Goals

The primary objective is to understand and implement how neural language models
learn meaningful representations of words from their surrounding context.

The project focuses on four major questions:

1. **How can words be represented as dense numerical vectors?**
2. **How can contextual information be used to learn semantic relationships?**
3. **Can negative sampling make Skip-Gram training computationally practical?**
4. **Can learned embeddings capture semantic similarity and vector relationships?**

---

## Model Architecture

The implementation follows the Skip-Gram approach:

```text
                  Training Corpus
                         │
                         ▼
                  Text Preprocessing
                         │
                         ▼
                    Tokenization
                         │
                         ▼
                 Vocabulary Mapping
                         │
                         ▼
               Context Window Creation
                         │
                         ▼
                ┌───────────────────┐
                │   Target Word     │
                │   "economy"       │
                └─────────┬─────────┘
                          │
                          ▼
                 Input Word Embedding
                          │
                          ▼
                 ┌─────────────────┐
                 │ Negative        │
                 │ Sampling        │
                 └────────┬────────┘
                          │
                          ▼
                 Context Prediction
                          │
                          ▼
                   Learned Embedding
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         Similarity    Analogies   Visualization
