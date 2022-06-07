import torch.nn

def adam(parameters, learning_rate):
    return torch.optim.Adam(parameters, lr=learning_rate)
