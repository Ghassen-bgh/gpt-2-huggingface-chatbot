import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def initialize_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return model


def initialize_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def preprocess_data(tokenizer, text):
    encoded_inputs = tokenizer(text, truncation=True, padding=True)
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    return input_ids, attention_mask



def train(model, input_ids, attention_mask):
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Convert to tensor and add batch dimension
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # Convert to tensor and add batch dimension

    model.train()
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    return loss



def main():
    tokenizer = initialize_tokenizer()
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = initialize_model()
    
    text = """
    Hello?
    Hi, how can I help you today?
    """
    input_ids, attention_mask = preprocess_data(tokenizer, text)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        
        loss = train(model, input_ids, attention_mask)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")


if __name__ == '__main__':
    main()
