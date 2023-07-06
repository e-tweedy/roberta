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
    
ex_q, ex_c = get_examples()

for i in range(len(ex_q)):
    st.sidebar.button(
        label = f'Try example {i+1}',
        key = f'ex_button_{i+1}',
        on_click = fill_in_example,
        args=(i,),
    )
st.sidebar.button(
    label = 'Clear boxes',
    key = 'clear_button',
    on_click = clear_boxes,
)

st.header('RoBERTa Q&A model')

st.markdown('''
This app demonstrates the answer-retrieval capabilities of a finetuned RoBERTa (Robustly optimized Bidirectional Encoder Representations from Transformers) model.  The [RoBERTa base model](https://huggingface.co/roberta-base) was fine-tuned on version 2 of the [SQuAD (Stanford Question Answering Dataset) dataset](https://huggingface.co/datasets/squad_v2), a dataset of context-question-answer triples.  The objective of the model is to retrieve the answer to the question from the context paragraph.

Version 2 incorporates the 100,000 samples from Version 1.1, along with 50,000 'unanswerable' questions, i.e. samples in the question cannot be answered using the context given.

Please type or paste a context paragraph and question you'd like to ask about it.  The model will attempt to answer the question, or otherwise will report that it cannot.

Alternatively, you can try some of the examples provided on the sidebar to the left.
''')
input_container = st.container()
st.divider()
response_container = st.container()
    
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
        st.session_state['context'] = context
        question = st.text_input(
            label='Question',
            value=st.session_state['question'],
            key='question_field',
            label_visibility='hidden',
            placeholder='Enter your question here.',
        )
        st.session_state['question'] = question
        query_submitted = st.form_submit_button("Submit")
        if query_submitted:
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
            
