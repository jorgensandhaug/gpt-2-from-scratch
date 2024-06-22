import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model
import wandb
from dataset import FineWebDataset

# Initialize Weights & Biases
wandb.init(project="gpt2")
# Load your dataset
data_dir = 'edu_fineweb10B'
train_dataset = FineWebDataset(data_dir, split='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = FineWebDataset(data_dir, split='val')  # Assuming you have a validation split
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model and optimizer
config = GPT2Config(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    max_position_embeddings=1024,
    type_vocab_size=2,
    initializer_range=0.02
)
model = GPT2Model(config)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Log hyperparameters
wandb.config.update({
    "learning_rate": 5e-5,
    "batch_size": 32,
    "num_epochs": 3,
})

# Training and validation loop
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    total_train_loss = 0.0
    total_val_loss = 0.0

    # Training loop
    model.train()
    for batch in train_loader:
        inputs, labels = batch, batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    wandb.log({"train_loss": avg_train_loss, "epoch": epoch})

    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch, batch
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

    print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

# Finish the wandb run
wandb.finish()
