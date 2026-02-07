import torch, numpy as np

# def load_data(x: torch.Tensor, batch_size: int, context_length: int, device: str):
#     return np.load()

def get_batch(x, batch_size, context_length, device):
    max_idx = len(x) - context_length 
    ix = torch.randint(0, max_idx, (batch_size,))
    
    inputs_list = []
    targets_list = []
    
    for i in ix:
        inputs_list.append(torch.from_numpy(x[i : i + context_length].astype(np.int64)))
        targets_list.append(torch.from_numpy(x[i + 1 : i + context_length + 1].astype(np.int64)))
    
    inputs = torch.stack(inputs_list)
    targets = torch.stack(targets_list)
    
    return inputs.to(device), targets.to(device)


def save_checkpoint(model, optimizer, iteration, out):
    training_state = {
        "params": model.state_dict(),
        "optim": optimizer.state_dict(),
        "it": iteration
    }
    torch.save(training_state, out)


def load_checkpoint(src, model, optimizer):
    training_state = torch.load(src)

    model.load_state_dict(training_state["params"])
    optimizer.load_state_dict(training_state["optim"])
    
    return training_state["it"] 