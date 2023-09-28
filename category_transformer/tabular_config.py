from dataclasses import dataclass
import torch

@dataclass
class TabularConfig:
    output_size: int = 4 # Number of classes
    n_layer: int = 8 # Number of decoder blocks
    n_head: int = 32 # Nmber of attention heads
    n_embd: int  = 128 # Embedding of all features 
    n_features: int = 14 # Number of non-category features
    dropout: float = 0.2
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    classification_weights: torch.tensor = torch.tensor([1.0, 1.0, 1.0, 1.0]) # This will be the weights for the classification loss
    embedding_config: list = None  # Initialize as None

    def __post_init__(self): # Initialize categorical values
        if self.embedding_config is None:
            self.embedding_config = [
                {"nr_classes": 4751, "embedding_dimension": 16},  # vendors
                {"nr_classes": 4, "embedding_dimension": 1},     # booking year
                {"nr_classes": 4, "embedding_dimension": 1}      # document year
            ]

