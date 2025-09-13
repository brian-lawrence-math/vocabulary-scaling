import argparse
import math
import torch
from torch import nn
import pickle
import os
import sys

from data import Data, START, STOP, PAD
from model import Seq2Seq
from config import Config

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.data = Data()
        self.data.load(num_words=config.num_words)
        self.model = Seq2Seq(tokens_in = self.data.n_tokens, tokens_out = self.data.n_tokens, config = self.config)
        self.best_count = 0
        if config.log_file is None:
            self.out_file = sys.stdout
        else:
            self.out_file = config.log_file

        # write info both to logfile and to (subprocess) stdout
        print(f"num_words:{config.num_words}")
        print(f"tot_params:{self.model.n_params()}")
        print(f"embed_dim:{config.embed_dim}")
        print(f"attn_layers:{config.attn_layers}")
        with open(config.log_file, "w") as logfile:
            logfile.write("Training with parameters:\n")
            logfile.write(f"  num_words = {config.num_words}\n")
            logfile.write(f"  embed_dim = {config.embed_dim}\n")
            logfile.write(f"  attn_layers = {config.attn_layers}\n")
            logfile.write(f"  tot params: {self.model.n_params()}\n")

    def run(self):
        config = self.config

        lr = config.lr
        batch_size = config.batch_size
        losses = []

        cml_loss = 0
        cml_count = 0
        count_targ = config.print_loss_interval

        data = self.data
        model = self.model

        model.train()
        with open(config.log_file, "a") as logfile:
            for epoch in range(config.epochs):
                es, en = data.random_items(batch_size)
                x, y = data.encode_list(es), data.encode_list(en)
                z = model(x, y[:,:-1])
                loss = nn.functional.cross_entropy(
                    z.flatten(0, 1), 
                    y[:,1:].flatten(0, 1), 
                    ignore_index = data.ctoi[PAD], 
                    label_smoothing=config.label_smoothing
                )
                losses.append(loss.item())
                
                cml_loss += loss.item()
                cml_count += 1
                if cml_count == count_targ:
                    avg_loss = cml_loss / cml_count
                    logfile.write(f"Loss after {epoch + 1} epochs: {avg_loss:0.3f}\n")
                    cml_loss = 0
                    cml_count = 0
                if (epoch + 1) % config.test_interval == 0:
                    logfile.write(self.score() + "\n")
                    logfile.flush()

                if self.best_count == config.num_words:
                    logfile.write(f"SUCCESS: training complete in {epoch + 1} epochs.")
                    print(f"best_count:{self.best_count}")
                    return
                if math.isnan(cml_loss):
                    logfile.write(f"FAILURE: loss is nan.")
                    print(f"best_count:{self.best_count}")
                    return
                
                loss.backward()
                for n, p in model.named_parameters():
                    p.data -= lr * p.grad
                    p.grad.zero_()
            print(f"best_count:{self.best_count}")
            logfile.write(f"best count:{self.best_count}\n")

    def translate(self, x_str, y_str=""):
        model = self.model
        data = self.data

        model.eval()
        MAX_OUTPUT_LEN = 15
        while(len(y_str) < MAX_OUTPUT_LEN):
            x, y = data.encode(x_str), data.encode(y_str)
            logits = model(x, y)[len(y_str)]
            next_token = logits.argmax()
            next_char = data.itoc[next_token.item()]
            if next_char == STOP:
                break
            y_str += next_char
        return y_str
    
    def score(self):
        data = self.data
        model = self.model

        model.eval()
        count = 0
        tot_count = len(data.data)
        for i, (es, en) in enumerate(data.data):
            en_pred = self.translate(es)
            count += en_pred == en
        self.best_count = max(self.best_count, count)
        return f"Correct translations: {count} / {tot_count}"
    
    def debug(self, x_str, y_str):
        """
        Tool to help debug translation
        """
        print(f"------------- DEBUGGING {x_str} {y_str} -------------")
        data = self.data
        model = self.model

        x, y = data.encode(x_str), data.encode(y_str)
        logits = torch.softmax(model(x, y)[len(y_str)], -1, torch.float)
        sorted, idxs = torch.sort(-logits)
        for i in range(4):
            print((self.data.itoc[idxs[i].item()], -sorted[i].item()))

    def save_model(self, fname="weights.pickle"):
        with open(fname, "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, fname="weights.pickle"):
        with open(fname, "rb") as file:
            self.model = pickle.load(file)

def run_experiment(config):
    trainer = Trainer(config)
    trainer.run()

if __name__ == "__main__":
    # populate config from command line args
    parser = argparse.ArgumentParser()
    int_attrs = [        
        "num_words", 
        "embed_dim", 
        "attn_layers", 
        "num_heads", 
        "epochs"
    ]
    float_attrs = [
        "lr",
        "label_smoothing"
    ]
    str_attrs = ["log_file"]
    for attr in int_attrs + float_attrs + str_attrs:
        parser.add_argument("--" + attr)
    args = parser.parse_args()
    config = Config()
    for attr in int_attrs:
        if getattr(args, attr) is not None:
            setattr(config, attr, int(getattr(args, attr)))
    for attr in float_attrs:
        if getattr(args, attr) is not None:
            setattr(config, attr, float(getattr(args, attr)))
    for attr in str_attrs:
        if getattr(args, attr) is not None:
            setattr(config, attr, getattr(args, attr))
    
    if config.log_file is not None:
        config.log_file = os.path.join("../logs", config.log_file)

    trainer = Trainer(config)
    trainer.run()

