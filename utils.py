import gc
import torch

def unload_model(model):
    model.to("cpu")
