import gc
import torch

def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
