from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.metrics import roc_auc_score

class Model:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_text(self, prompt, max_new_tokens=500, do_sample=False, temperature=0.0):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def linear_classifier(self, positive_set, negative_set, depth):
        device = self.model.device
        hidden_size = self.model.config.hidden_size
        
        # set up contrastive dataset
        positive_encodings = self.tokenizer(positive_set, return_tensors="pt", padding=True, truncation=True)
        negative_encodings = self.tokenizer(negative_set, return_tensors="pt", padding=True, truncation=True)
        positive_input_ids = positive_encodings['input_ids'].to(device)
        negative_input_ids = negative_encodings['input_ids'].to(device)
        
        # get activations at a given layer
        with torch.no_grad():
            positive_outputs = self.model(positive_input_ids, output_hidden_states=True)
            positive_activations = positive_outputs.hidden_states[depth]  # Shape: [batch, seq_len, hidden_size]
            positive_activations = positive_activations.mean(dim=1)  # Average over sequence length
            
            negative_outputs = self.model(negative_input_ids, output_hidden_states=True)
            negative_activations = negative_outputs.hidden_states[depth]
            negative_activations = negative_activations.mean(dim=1)
    
        # train linear classifier
        X = torch.cat([positive_activations, negative_activations], dim=0)
        y = torch.cat([
            torch.ones(positive_activations.shape[0]),
            torch.zeros(negative_activations.shape[0])
        ]).to(device)
        
        classifier = torch.nn.Linear(hidden_size, 1).to(device)
        optimizer = torch.optim.Adam(classifier.parameters())
        criterion = torch.nn.BCEWithLogitsLoss()
        
        num_epochs = 100
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            logits = classifier(X).squeeze()
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            scores = torch.sigmoid(classifier(X).squeeze())
            scores_np = scores.cpu().numpy()
            labels_np = y.cpu().numpy()
            auroc = roc_auc_score(labels_np, scores_np)

        return classifier, auroc