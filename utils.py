from transformers import GPT2LMHeadModel, GPT2Tokenizer

def initialize_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer
