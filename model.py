import torch
import math

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

# def my_scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, dropout_p:float=0.0,is_causal:bool=True):
#     # Q,K,V: [batch_size x num_heads x seq_len x feature_dim]
#     assert is_causal
#     attn_bias = torch.zeros(Q.shape[-2], Q.shape[-2], dtype=Q.dtype)
#     mask = torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool).tril(diagonal=0)
#     attn_bias.masked_fill_(mask.logical_not(), float('-inf')).to(Q.dtype)
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     attn_weight = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(Q.shape[0]) + attn_bias.to(device)
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#     return torch.matmul(attn_weight, V)

# class CausalMultiheadSelfAttention(torch.nn.Module):
#     def __init__(self, d_model:int, num_heads:int, context_length:int, attn_pdrop:float=0.0):
#         super().__init__()
#         self.d_model = d_model
#         self.H = num_heads
#         self.d_v = d_model//num_heads
#         self.context_length = context_length
#         self.attn_pdrop = attn_pdrop
#         self.input_proj = torch.nn.Linear(self.d_model, 3*self.H*self.d_v, bias=False)
#         self.output_proj = torch.nn.Linear(self.H*self.d_v, self.d_model, bias=False)

#         # if int(torch.__version__[0]) == 2:
#         self.attn_fn = torch.nn.functional.scaled_dot_product_attention
#         # else:
#         #     self.attn_fn = my_scaled_dot_product_attention

#     def forward(self, X:torch.Tensor):
#         Q, K, V = self.input_proj(X).split(self.H*self.d_v, dim=-1)
#         mat_shape = list(Q.shape)
#         mat_shape[-1] = self.H
#         mat_shape.append(self.d_v)
        
#         Q = Q.view(mat_shape).transpose(-3,-2) # [batch_size x num_heads x seq_len x feature_dim]
#         K = K.view(mat_shape).transpose(-3,-2)
#         V = V.view(mat_shape).transpose(-3,-2)

#         pdrop = self.attn_pdrop if self.training else 0.0
#         attention = self.attn_fn(Q,K,V,dropout_p=pdrop,is_causal=True)

#         mat_shape = mat_shape[:-2] + [self.H*self.d_v]
#         attention = attention.transpose(-3,-2).view(mat_shape)
#         return self.output_proj(attention)

# for Torch 1.8.1
class CausalMultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, context_length:int, attn_pdrop:float=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_model//num_heads
        self.attn_pdrop = attn_pdrop
        self.context_length = context_length
        self.attn = torch.nn.MultiheadAttention(d_model, num_heads, attn_pdrop, bias=False)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.mask = torch.tril(torch.ones(context_length, context_length)).to(device)

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
        self.out_linear = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, X):
        X = self.embedding(X)
        for layer in self.layers:
            X = layer(X)
        return self.out_linear(self.rms_out(X))









