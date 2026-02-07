# import torch
# import numpy as np
# import os
# import time
# from layer import TransformerLM
# from cs336_basics.data_loader import get_batch, save_checkpoint

# config = {
#     "vocab_size": 10000,
#     "batch_size": 32,
#     "num_layers": 15,
#     "d_model": 1024,
#     "d_ff": 4096,
#     "context_length": 128,
#     "max_steps": 5000,
#     "learning_rate": 3e-4,
#     "eval_interval": 500,
#     "checkpoint_path": "checkpoints/model_latest.pt",
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "dtype": np.uint16 
# }

# train_data = np.memmap('train.bin', dtype=config["dtype"], mode='r')
# val_data = np.memmap('val.bin', dtype=config["dtype"], mode='r')


# model = TransformerLM(config["vocab_size"], config["context_length"], config["num_layers"], ) # Initialize with your hyperparams
# model.to(config["device"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

# for step in range(config["max_steps"]):
#     # 1. Periodically Evaluate
#     if step % config["eval_interval"] == 0:
#         val_loss = evaluate(model, val_data) # Helper to average loss over N batches
#         print(f"Step {step}: Val Loss {val_loss:.4f}")
        
#         save_checkpoint(model, optimizer, step, config["checkpoint_path"])

#     # 3. Training Step
#     xb, yb = get_batch(train_data, config["batch_size"], config["context_length"], device=config["device"])
#     logits, loss = model(xb, yb)
    
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
    
#     optimizer.step()

#     if step % 100 == 0:
#         print(f"Step {step}: Train Loss {loss.item():.4f}")