import torch
import streamlit as st
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from lib.utils import preprocess_examples, make_predictions, get_examples

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# TO DO:
# - make it pretty
# - add support for multiple questions corresponding to same context
# - add examples
# What else??

# Initialize session state variables
if 'response' not in st.session_state:
    st.session_state['response'] = ''
if 'context' not in st.session_state:
    st.session_state['context'] = ''
if 'question' not in st.session_state:
    st.session_state['question'] = ''
    
# Build trainer using model and tokenizer from Hugging Face repo
@st.cache_resource(show_spinner=False)
def get_model():
    repo_id = 'etweedy/roberta-base-squad-v2'
    model = AutoModelForQuestionAnswering.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer

def fill_in_example(i):
    st.session_state['response'] = ''
    st.session_state['question'] = ex_q[i]
    st.session_state['context'] = ex_c[i]

def clear_boxes():
    st.session_state['response'] = ''
    st.session_state['question'] = ''
    st.session_state['context'] = ''

with st.spinner('Loading the model...'):
    model, tokenizer = get_model()

st.header('RoBERTa Q&A model')

st.markdown('''
This app demonstrates the answer-retrieval capabilities of a fine-tuned RoBERTa (Robustly optimized Bidirectional Encoder Representations from Transformers) model.
''')
with st.expander('Click to read more about the model...'):
    st.markdown('''
* [Click here](https://huggingface.co/etweedy/roberta-base-squad-v2) to visit the Hugging Face model card for this fine-tuned model.
* To create this model, the [RoBERTa base model](https://huggingface.co/roberta-base) was fine-tuned on Version 2 of [SQuAD (Stanford Question Answering Dataset)](https://huggingface.co/datasets/squad_v2), a dataset of context-question-answer triples.
* The objective of the model is "extractive question answering", the task of retrieving the answer to the question from a given context text corpus.
* SQuAD Version 2 incorporates the 100,000 samples from Version 1.1, along with 50,000 'unanswerable' questions, i.e. samples in the question cannot be answered using the context given.
* The original base RoBERTa model was introduced in [this paper](https://arxiv.org/abs/1907.11692) and [this repository](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta).  Here's a citation for that base model:
```bibtex
@article{DBLP:journals/corr/abs-1907-11692,
  author    = {Yinhan Liu and
               Myle Ott and
               Naman Goyal and
               Jingfei Du and
               Mandar Joshi and
               Danqi Chen and
               Omer Levy and
               Mike Lewis and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {RoBERTa: {A} Robustly Optimized {BERT} Pretraining Approach},
  journal   = {CoRR},
  volume    = {abs/1907.11692},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.11692},
  archivePrefix = {arXiv},
  eprint    = {1907.11692},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-11692.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
''')

st.markdown('''
Please type or paste a context paragraph and question you'd like to ask about it.  The model will attempt to answer the question, or otherwise will report that it cannot.  Your results will appear below the question field when the model is finished running.

Alternatively, you can try an example by clicking one of the buttons below:
''')

ex_q, ex_c = get_examples()
example_container = st.container()
input_container = st.container()
response_container = st.container()

with example_container:
    ex_cols = st.columns(len(ex_q)+1)
    for i in range(len(ex_q)):
        with ex_cols[i]:
            st.button(
                label = f'Try example {i+1}',
                key = f'ex_button_{i+1}',
                on_click = fill_in_example,
                args=(i,),
            )
    with ex_cols[-1]:
        st.button(
            label = "Clear all fields",
            key = "clear_button",
            on_click = clear_boxes,
        )
            
# Form for user inputs
with input_container:
    with st.form(key='input_form',clear_on_submit=False):
        context = st.text_area(
            label='Context',
            value=st.session_state['context'],
            key='context_field',
            label_visibility='hidden',
            placeholder='Enter your context paragraph here.',
            height=300,
        )
        question = st.text_input(
            label='Question',
            value=st.session_state['question'],
            key='question_field',
            label_visibility='hidden',
            placeholder='Enter your question here.',
        )
        query_submitted = st.form_submit_button("Submit")
        if query_submitted:
            st.session_state['question'] = question
            st.session_state['context'] = context
            with st.spinner('Generating response...'):
                data_raw = Dataset.from_dict(
                    {
                        'id':[0],
                        'context':[st.session_state['context']],
                        'question':[st.session_state['question']],
                    }
                )
                data_proc = data_raw.map(
                    preprocess_examples,
                    remove_columns = data_raw.column_names,
                    batched = True,
                    fn_kwargs = {
                        'tokenizer':tokenizer,
                    }
                )
                predicted_answers = make_predictions(model, tokenizer,
                                                    data_proc, data_raw,
                                                    n_best = 20)
                answer = predicted_answers[0]['prediction_text']
                confidence = predicted_answers[0]['confidence']
                st.session_state['response'] = f"""
                    Answer: {answer}\n
                    Confidence: {confidence:.2%}
                """
with response_container:
    st.write(st.session_state['response'])
            
