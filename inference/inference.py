from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32
    )
    return model, tokenizer

def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    inputs = data.get("inputs")
    tokens = tokenizer(inputs, return_tensors="pt")
    output = model.generate(**tokens, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)
