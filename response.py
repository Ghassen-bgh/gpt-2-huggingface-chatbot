# response.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

def get_response(user_input):
    inputs = tokenizer.encode_plus(
    user_input,
    add_special_tokens=True,
    padding="longest",
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    response_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

