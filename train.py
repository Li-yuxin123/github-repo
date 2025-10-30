import os 
import yaml   
import torch  
from tqdm import tqdm 
import torch.nn as nn  
from Bio import SeqIO
import torch.optim as optim  
from torch.utils.data import DataLoader, Dataset  
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
import re
import wandb
global SEED 
SEED = 2025
import time

from megaDNA.megadna import MEGADNA

class DNADataset(Dataset):
    """
    Custom PyTorch Dataset for handling DNA sequences.
    It converts sequences to numerical tokens and prepares them for autoregressive modeling.
    """
    def __init__(self, sequences, seq_length=8192):
        self.seq_length = seq_length
        self.sequences = sequences
        self.nucleotide_to_token = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'N': 0}  # pad_id=0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Truncate or pad the sequence to the desired length
        if len(seq) > self.seq_length:
            seq = seq[:self.seq_length]
        else:
            # Pad with 'N' (which will be converted to a random base)
            seq = seq.ljust(self.seq_length, 'N')
            
        # tokens = torch.tensor(self.convert_string_to_list(seq), dtype=torch.long)
        # 转换为token
        tokens = torch.tensor([self.nucleotide_to_token.get(c.upper(), 0) for c in seq], 
                             dtype=torch.long)
        
        # Create input (x) and target (y) for next-token prediction
        x = tokens[:-1]
        y = tokens[1:]
        
        return x, y

    @staticmethod
    def convert_string_to_list(s):
        """
        Converts a DNA string (A, T, C, G) to a list of numerical tokens.
        'N' characters are mapped to a random base.
        """
        np.random.seed(SEED)
        s = s.upper()
        # Simple mapping: A=1, T=2, C=3, G=4. 'N' gets a random choice.
        return np.array([1 if c == 'A' else 2 if c == 'T' else 3 if c == 'C' else 4 if c == 'G' else np.random.choice([1, 2, 3, 4]) for c in s])


        
def train_megadna_model(config_path, custom_sequences, output_dir, epochs=10, batch_size=2, lr=1e-4):
    """
    Trains the StripedHyena model and saves checkpoints every 0.5 epochs.

    Args:
        config_path (str): Path to the model configuration YAML file.
        custom_sequences (list): List of DNA sequences for training.
        output_dir (str): Directory to save model checkpoints and loss data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
    """
    # 创建日志文件路径
    log_file = os.path.join(output_dir, "training.log")

    def log_message(message):
        """同时输出到终端和日志文件"""
        tqdm.write(message)  # 输出到终端
        with open(log_file,'a',encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 记录训练开始
    log_message("=" * 50)
    log_message("Starting MEGADNA training")
    log_message(f"Output directory: {output_dir}")
    log_message(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")

    epoch_losses = []
    batch_losses = []
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # global_config = dotdict(config)
    # model = StripedHyena(global_config)
    # MEGADNA 初始化
    model = MEGADNA(
    num_tokens=6,           
    dim=(512,256),                
    depth=(8, 8),           
    max_seq_len=(256, 32)  
    )
    # # print("Loaded configuration:", global_config)
    # log_message(f"Loaded configuration: {global_config}")
    # global_config.hidden_size = global_config.hidden_size
    # global_config.num_filters = global_config.num_filters
    # global_config.num_layers = global_config.num_layers
    # global_config.num_attention_heads = global_config.num_attention_heads

    # # print("Hidden Size:", global_config.hidden_size)
    # # print("Number of Filters:", global_config.num_filters)
    # # print("Number of Layers:", global_config.num_layers)
    # # print("Number of Attention Heads:", global_config.num_attention_heads)
    # log_message(f"Hidden Size: {global_config.hidden_size}")
    # log_message(f"Number of Filters: {global_config.num_filters}")
    # log_message(f"Number of Layers: {global_config.num_layers}")
    # log_message(f"Number of Attention Heads: {global_config.num_attention_heads}")

    dataset = DNADataset(custom_sequences, seq_length=8192)
    #print(len(dataset))
    log_message(f"sequence size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    model = model.to(torch.float32)

    # --- Parameter Counting Section ---
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {num_params:,}")
    log_message(f"Total trainable parameters: {num_params:,}")

    # print("\n--- Layer-wise Trainable Parameters ---")
    log_message("\n--- Layer-wise Trainable Parameters ---")
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        total_trainable_params += n

    block_pattern = re.compile(r'blocks\.(\d+)\.')
    max_block = -1
    for name, _ in model.named_parameters():
        m = block_pattern.search(name)
        if m:
            max_block = max(max_block, int(m.group(1)))

    for idx in range(max_block + 1):
        # print(f"\n========== blocks.{idx} ==========")
        log_message(f"\n========== blocks.{idx} ==========")
        total = 0
        for name, param in model.named_parameters():
            if param.requires_grad and f"blocks.{idx}." in name:
                n = param.numel()
                total += n
                # print(f"{name:<80} | shape={str(tuple(param.shape)):<20} | params={n:>10,}")
                log_message(f"{name:<80} | shape={str(tuple(param.shape)):<20} | params={n:>10,}")
        # print(f"Total trainable params in blocks.{idx}: {total:,}")
        log_message(f"Total trainable params in blocks.{idx}: {total:,}")
    
    # --- Training Loop ---
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        epoch_batch_losses = []
        num_batches = len(dataloader)
        mid_epoch_batch = num_batches // 2

        # Use enumerate to track batch index
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            x, y = x.to(device), y.to(device)
            x = x.long()
            # logits, _ = model(x)

            # loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = model(x, return_value='loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_batch_losses.append(loss.item() * x.size(0))
            #print(loss.item())

            # # --- Mid-epoch checkpoint saving ---
            # # Check if the current batch is the mid-epoch batch
            # if batch_idx + 1 == mid_epoch_batch:
            #     avg_loss_mid = total_loss / (batch_idx + 1)
            
            #     if not os.path.exists(output_dir):
            #         os.makedirs(output_dir)
            
            #     # Save checkpoint for the 0.5 epoch mark
            #     checkpoint_path = os.path.join(output_dir, f"evo_model_epoch_{epoch + 0.5}.pt")
            #     torch.save({
            #         'epoch': epoch + 0.5,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': avg_loss_mid,
            #     }, checkpoint_path)
            #     # Print confirmation outside the tqdm progress bar
            #     #tqdm.write(f"Checkpoint saved to {checkpoint_path}")
            #     log_message(f"Checkpoint saved to {checkpoint_path}")


        # --- End-of-epoch processing ---
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        batch_losses.extend(epoch_batch_losses)

        # tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        epoch_time = time.time() - epoch_start
        log_message(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Duration: {epoch_time:.2f} s")

        # --- End-of-epoch checkpoint saving ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        checkpoint_path = os.path.join(output_dir, f"megadna_model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,  # Make epoch number consistent with filename
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        log_message(f"Checkpoint saved to {checkpoint_path}")

        # --- Save loss data ---
        loss_data_path = os.path.join(output_dir, "epoch_losses_data.npy")
        np.save(loss_data_path, {'epoch_losses': epoch_losses})

        batch_loss_data_path = os.path.join(output_dir, "batch_losses_data.npy")
        np.save(batch_loss_data_path, {'batch_losses': batch_losses})
        
        # tqdm.write(f"Checkpoint saved to {checkpoint_path}")
    # 记录训练结束
    log_message("Training Completed")
    log_message("=" * 50)

    return model


config_path = "config/hyena.yml"  
    
custom_sequences =[]

for record in SeqIO.parse("combined_contigs_virus_filtered_96K_sampled1K.fna", "fasta"):
#for record in SeqIO.parse("../autodl-fs/phage_sequences.fasta", "fasta"):
    custom_sequences.append(str(record.seq))
      
output_dir = "megadna_trained_model" 
    
epochs = 45
batch_size = 1  
learning_rate = 1e-4  

model = train_megadna_model(  
    config_path=config_path,  
    custom_sequences=custom_sequences,  
    output_dir=output_dir,  
    epochs=epochs,  
    batch_size=batch_size,  
    lr=learning_rate
)