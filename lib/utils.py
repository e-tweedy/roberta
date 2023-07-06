import numpy as np
from scipy.special import softmax
import collections
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

def preprocess_examples(examples, tokenizer , max_length = 384, stride = 128):
    """
    Preprocesses and tokenizes examples in preparation for inference
    
    Parameters:
    -----------
    examples : datasets.Dataset
        The dataset of examples.  Must have columns:
        'id', 'question', 'context'
    tokenizer : transformers.AutoTokenizer
        The tokenizer for the model
    max_length : int
        The max length for context truncation
    stride : int
        The stride for context truncation

    Returns:
    --------
    inputs : dict
        The tokenized and processed data dictionary with
        keys 'input_ids', 'attention_mask', 'offset_ids', 'example_id'
        All values are lists of length = # of inputs output by tokenizer
            inputs['input_ids'][k] : list
                token ids corresponding to tokens in feature k
            inputs['attention_mask'][k] : list
                attention mask for feature k
            inputs['offset_ids'][k] : list
                offset ids for feature k
            inputs['example_id'][k] : int
                id of example from which feature k originated
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples['context'],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def make_predictions(model,tokenizer,inputs,examples,
                    n_best = 20,max_answer_length=30):
    """
    Generates a list of prediction data based on logits

    Parameters:
    -----------
    model : transformers.AutoModelForQuestionAnswering
        The trained model
    tokenizer : transformers.AutoTokenizer
        The model's tokenizer
    inputs : dict
        The tokenized and processed data dictionary with
        keys 'input_ids', 'attention_mask', 'offset_ids', 'example_id'
        All values are lists of length = # of inputs output by tokenizer
            inputs['input_ids'][k] : list
                token ids corresponding to tokens in feature k
            inputs['attention_mask'][k] : list
                attention mask for feature k
            inputs['offset_ids'][k] : list
                offset ids for feature k
            inputs['example_id'][k] : int
                id of example from which feature k originated
    examples : datasets.Dataset
        The dataset of examples.  Must have columns:
        'id', 'question', 'context'
    n_best : int
        The number of top start/end (by logit) indices to consider
    max_answer_length : int
        The maximum length (in characters) allowed for a candidate answer

    Returns:
    --------
    predicted_answers : list(dict)
        predicted_answers[k] has keys 'id','prediction_text','confidence'
        predicted_answers[k]['id'] : int
            The unique id of the example
        predicted_answers[k]['prediction_text'] : str
            The predicted answer as a string
        predicted_answers[k]['confidence'] : float
            The predicted probability corresponding to the answer, i.e. the
            corresponding output of a softmax function on logits          
    """
    assert n_best <= len(inputs['input_ids'][0]), 'n_best cannot be larger than max_length'
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.us_available():
        device = "cuda"
    else:
        device = "cpu"
    data_for_model = inputs.remove_columns(["example_id", "offset_mapping"])
    data_for_model.set_format("torch",device=device)
    dl = DataLoader(
        data_for_model,
        collate_fn=default_data_collator,
        batch_size=len(inputs)
    )
    model = model.to(device)
    for batch in dl:
        outputs = model(**batch)
        
    start_logits = outputs.start_logits.cpu().detach().numpy()
    end_logits = outputs.end_logits.cpu().detach().numpy()       
    example_to_inputs = collections.defaultdict(list)
    for idx, feature in enumerate(inputs):
        example_to_inputs[feature["example_id"]].append(idx)
    
    predicted_answers = []
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_inputs[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = inputs[feature_index]['offset_mapping']
            
            start_indices = np.argsort(start_logit)[-1:-n_best-1:-1].tolist()
            end_indices = np.argsort(end_logit)[-1 :-n_best-1: -1].tolist()

            for start_index in start_indices:
                for end_index in end_indices:
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if(
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    if (offsets[start_index] is None)^(offsets[end_index] is None):
                        continue
                    if (offsets[start_index] is None)&(offsets[end_index] is None):
                        answers.append(
                            {
                                    "text": '',
                                    "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
                    else:
                        answers.append(
                            {
                                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
            answer_logits = [a['logit_score'] for a in answers]
            answer_probs = softmax(answer_logits)
            
            if len(answers)>0:
                best_answer = max(answers, key=lambda x:x['logit_score'])
                predicted_answers.append(
                    {'id':example_id, 'prediction_text':best_answer['text'], 'confidence':answer_probs[0]}
                )
            else:
                predicted_answers.append({'id':example_id, 'prediction_text':'','confidence':answer_probs[0]})
            for pred in predicted_answers:
                if pred['prediction_text'] == '':
                    pred['prediction_text'] = "I don't have an answer based on the context provided."
    return predicted_answers