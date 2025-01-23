from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Model:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=500, do_sample=False, temperature=0.0):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
