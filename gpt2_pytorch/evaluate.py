import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

# Load your dataset
data_dir = 'edu_fineweb10B'
val_dataset = FineWebDataset(data_dir, split='val')
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
config = GPT2Config(vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12, max_position_embeddings=1024, type_vocab_size=2, initializer_range=0.02)
model = GPT2Model(config)
model.eval()

# Evaluation loop
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for batch in val_loader:
        inputs, labels = batch, batch
        outputs = model(inputs)
        val_loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        print(f"Validation Loss: {val_loss.item()}")
