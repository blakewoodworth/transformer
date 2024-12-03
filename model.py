import torch
import math
from utils import *

class RMSNorm(torch.nn.Module):
    def __init__(self, d:int, eps:float=1e-5):
        super().__init__()
        self.d = d
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(self.d))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        return (x * self.weight) / rms

class FFN(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(torch.nn.functional.gelu(self.w1(x)))

class CausalMultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, context_length:int, attn_pdrop:float=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_model//num_heads
        self.attn_pdrop = attn_pdrop
        self.context_length = context_length
        self.attn = torch.nn.MultiheadAttention(d_model, num_heads, attn_pdrop, bias=False)
        device = get_device()
        self.mask = torch.torch.triu(torch.ones(context_length, context_length), diagonal=1)>0
        self.mask = self.mask.to(device)

    def forward(self, X:torch.Tensor):
        Xt = X.transpose(0,1)
        out,_ = self.attn(query=Xt, key=Xt, value=Xt, need_weights=False, attn_mask=self.mask[:X.shape[-2],:X.shape[-2]])
        return out.transpose(0,1)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, context_length:int, attn_pdrop:float=0.0, residual_pdrop:float=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.context_length = context_length

        self.rms_1 = RMSNorm(d_model, eps=1e-5)
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, context_length, attn_pdrop)
        self.rms_2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, X:torch.Tensor):
        out1 = X + torch.nn.functional.dropout(self.attn(self.rms_1(X)), p=self.attn_pdrop, inplace=True)
        return out1 + torch.nn.functional.dropout(self.ffn(self.rms_2(out1)), p=self.attn_pdrop, inplace=True)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size:int, d_model:int, context_length:int, pdrop:float=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length
        self.pdrop = pdrop

        self.token_embeddings = torch.nn.Embedding(vocab_size, d_model)
        self.position_embeddings = torch.nn.parameter.Parameter(torch.empty(context_length, d_model))
        torch.nn.init.xavier_normal_(self.position_embeddings)

    def forward(self, X:torch.Tensor):
        return torch.nn.functional.dropout(self.token_embeddings(X) + self.position_embeddings[:X.shape[-1],:], p=self.pdrop, inplace=True)

class TransformerLM(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, vocab_size:int, context_length:int, num_layers:int, 
        embed_pdrop:float=0.0, attn_pdrop:float=0.0, residual_pdrop:float=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.embed_pdrop = embed_pdrop
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.embedding = PositionalEmbedding(vocab_size, d_model, context_length, embed_pdrop)

        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.rms_out = RMSNorm(d_model, eps=1e-5)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)

        # zero out second layer of ffn weights and LM head weights
        for name, param in self.named_parameters():
            if name[-len('ffn.w2.weight'):] == 'ffn.w2.weight':
                param.data.zero_()
            elif name == 'lm_head.weight':
                param.data.zero_()

    def forward(self, X):
        X = self.embedding(X)
        for layer in self.layers:
            X = layer(X)
        return self.lm_head(self.rms_out(X))

def make_model(args):
    model_hyperparameters = {
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'num_layers': args.num_layers,
        'embed_pdrop': args.embed_pdrop,
        'attn_pdrop': args.attn_pdrop,
        'residual_pdrop': args.residual_pdrop
    }
    model = TransformerLM(**model_hyperparameters)
    if int(torch.__version__[0]) >= 2:
        model = torch.compile(model)
    return model




