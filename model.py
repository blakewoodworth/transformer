import torch

class FFN(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(torch.nn.functional.gelu(self.w1(x)))

class CausalMultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, attn_pdrop:float|None=None):
        super().__init__()
        self.d_model = d_model
        self.H = num_heads
        self.d_v = d_model//num_heads
        self.attn_pdrop = attn_pdrop
        self.input_proj = torch.nn.Linear(self.d_model, 3*self.H*self.d_v, bias=False)
        self.output_proj = torch.nn.Linear(self.H*self.d_v, self.d_model, bias=False)

    def forward(self, X:torch.Tensor):
        Q, K, V = self.input_proj(X).split(self.H*self.d_v, dim=-1)
        mat_shape = list(Q.shape)
        mat_shape[-1] = self.H
        mat_shape.append(self.d_v)
        
        Q = Q.view(mat_shape).transpose(-3,-2)
        K = K.view(mat_shape).transpose(-3,-2)
        V = V.view(mat_shape).transpose(-3,-2)

        attention = torch.nn.functional.scaled_dot_product_attention(Q,K,V,dropout_p=self.attn_pdrop,is_causal=True)
        mat_shape = mat_shape[:-2] + [self.H*self.d_v]
        attention = attention.transpose(-3,-2).view(mat_shape)
        return self.output_proj(attention)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, attn_pdrop:float|None=None, residual_pdrop:float|None=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop

        self.rms_1 = torch.nn.RMSNorm(d_model, eps=1e-5)
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.rms_2 = torch.nn.RMSNorm(d_model, eps=1e-5)
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

        # self.token_embeddings = torch.nn.Embedding(vocab_size, d_model)
        # self.position_embeddings = torch.nn.parameter.Parameter(torch.empty(context_length, d_model))
        # torch.nn.init.xavier_normal_(self.position_embeddings)

        self.embedding = PositionalEmbedding(vocab_size, d_model, context_length, embed_pdrop)

        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.rms_out = torch.nn.RMSNorm(d_model, eps=1e-5)
        self.out_linear = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, X):
        # X = torch.nn.functional.dropout(self.token_embeddings(X) + self.position_embeddings[:X.shape[-1],:], p=self.embed_pdrop, inplace=True)
        X = self.embedding(X)
        for layer in self.layers:
            X = layer(X)
        logits = self.out_linear(self.rms_out(X))
        return torch.nn.functional.softmax(logits, dim=-1)









