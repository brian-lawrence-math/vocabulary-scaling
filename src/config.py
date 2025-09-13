from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    num_words: int = 1 << 8     # 16, 64, 256, 1024
    embed_dim: int = 24        # 2, 4, 6, 8, 12, 16, 24, 32
                        # embed_dim must be divisible by num_heads
    attn_layers: int = 4       # 1, 2, 4, 6, 10
    num_heads: int = 2
    epochs: int = 1000 # 10000 or 20000 recommended.
         # epochs should be a multiple of test_interval

    lr: float = 0.1 # Chosen by experiment
    # In initial experiment, lr = 0.1, 0.2, 0.5, 1.0 all reasonable
    # But with small numbers of parameters, lr = 0.5 is too big
    label_smoothing: float = 0.1 # Chosen by experiment
    # In fact, label_smoothing = 0.0, 0.1, 0.2 all seem to work well

    print_loss_interval: int = 500  # how often to print training loss
    test_interval: int = 500        # how often to test on all vocabulary items
                        # do not make this value too small:
                        # testing imposes significant performance costs

    pos_embed_k: int = 2 # use 2*k dimension for positional embedding

    batch_size: int = 100

    log_file: Optional[str] = None

