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
from lib.utils import preprocess_examples, make_predictions

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

with st.spinner('Loading the model...'):
    model, tokenizer = get_model()

input_container = st.container()
st.divider()
response_container = st.container()
    
# Form for user inputs
with input_container:
    with st.form(key='input_form',clear_on_submit=False):
        context = st.text_area(
            label='Context',
            value='',
            key='context_field',
            label_visibility='hidden',
            placeholder='Enter your context paragraph here.',
            height=300,
        )
        question = st.text_input(
            label='Question',
            value='',
            key='question_field',
            label_visibility='hidden',
            placeholder='Enter your question here.',
        )
        query_submitted = st.form_submit_button("Submit")
        if query_submitted:
            with st.spinner('Generating response...'):
                data_raw = Dataset.from_dict(
                    {
                        'id':[0],
                        'context':[context],
                        'question':[question]
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
            
