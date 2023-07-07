# App for RoBERTa model fine-tuned on SQuAD v2

This is a streamlit app which demonstrates the answer-retrieval capabilities of a fine-tuned RoBERTa (Robustly optimized Bidirectional Encoder Representations from Transformers) model.  [Click here to visit the 🤗 model card](https://huggingface.co/etweedy/roberta-base-squad-v2) to see more details about this model.

* The model is a fine-tuned version of [roberta-base for QA](https://huggingface.co/roberta-base)
* It was fine-tuned for context-based extractive question answering on the [SQuAD v2 dataset](https://huggingface.co/datasets/squad_v2), a dataset of English-language context-question-answer triples designed for extractive question answering training and benchmarking.
* Version 2 of SQuAD (Stanford Question Answering Dataset) contains the 100,000 examples from SQuAD Version 1.1, along with 50,000 additional "unanswerable" questions, i.e. questions whose answer cannot be found in the provided context.
* The original RoBERTa (Robustly Optimized BERT Pretraining Approach) model was introduced in [this paper](https://arxiv.org/abs/1907.11692) and [this repository](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)
