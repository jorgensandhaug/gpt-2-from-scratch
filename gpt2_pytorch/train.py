import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from model import GPT2Model, GPT2Config
import wandb
from dataset import FineWebDataset
import time

# Initialize Weights & Biases
wandb.init(project="gpt2")

# Define parameters
data_dir = 'edu_fineweb10B'
B = 32  # Batch size for DataLoader
T = 1024  # Sequence length
process_rank = 0  # Rank of the current process (e.g., in multi-GPU training)
num_processes = 1  # Total number of processes

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load your datasets
train_dataset = FineWebDataset(data_dir, B, T, process_rank, num_processes, split='train')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset = FineWebDataset(data_dir, B, T, process_rank, num_processes, split='val') 
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize the model and optimizer
config = GPT2Config(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    max_position_embeddings=T,
)
model = GPT2Model(config).to(device)
lr = 6e-4
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# Training and validation loop
num_epochs = 1000
previous_best_val_loss = float('inf')
# max_num_batches_train = 1000
max_num_batches_val = 100
validation_every = 10

# Log hyperparameters
wandb.config.update({
    "learning_rate": lr,
    "batch_size": B, "num_epochs": num_epochs,
})


def run_validation(model, val_loader, step):
    global previous_best_val_loss
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i > max_num_batches_val:
                break
        
            inputs, labels = batch[0].to(device), batch[1].to(device)
            inputs = inputs.reshape(B, T)
            labels = labels.reshape(B, T)
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / min(len(val_loader), max_num_batches_val)
    wandb.log({"avg_val_loss": avg_val_loss, "step": step})

    if avg_val_loss < previous_best_val_loss:
        previous_best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "gpt2.pth")
        print(f"Model saved to gpt2.pth")

for epoch in range(num_epochs):
    total_train_loss = 0.0

    # Training loop
    print(f"Running training loop for epoch {epoch}, number of batches: {len(train_loader)}")
    model.train()
    for i, batch in enumerate(train_loader):
        # if i > max_num_batches_train:
        #     break

        time1 = time.time()
        inputs, labels = batch[0].to(device), batch[1].to(device)
        inputs = inputs.reshape(B, T)
        labels = labels.reshape(B, T)
        time2 = time.time()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        time3 = time.time()
        wandb.log({"train_loss": loss.item(), "time_loading": time2 - time1, "time_training": time3 - time2})

        if i % validation_every == 0:
            run_validation(model, val_loader, step=epoch*len(train_loader) + i)
    
    avg_train_loss = total_train_loss / min(len(train_loader), max_num_batches_train)
    wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch})

    print(f"Epoch {epoch}, Avg Train Loss: {avg_train_loss}")

# Finish the wandb run
wandb.finish()
