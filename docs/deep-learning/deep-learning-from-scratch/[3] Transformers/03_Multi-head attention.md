[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/deep-learning-from-scratch/[3] Transformers/03_Multi-head attention.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: PyTorch CUDA 12.1
    language: python
    name: pytorchcu121
---

## Input sequence: "Dream big and work for it"

```python
import torch

inputs = torch.tensor(
    [[0.72, 0.45, 0.31], # Dream    (x^1)
     [0.75, 0.20, 0.55], # big      (x^2)
     [0.30, 0.80, 0.40], # and      (x^3)
     [0.85, 0.35, 0.60], # work     (x^4)
     [0.55, 0.15, 0.75], # for      (x^5)
     [0.25, 0.20, 0.85]] # it       (x^6)
)

# Corresponding words
words = ['Dream', 'big', 'and', 'work', 'for', 'it']
```

## Class for implementing causal attention

```python
from torch import nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ values.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
```

```python
d_in = inputs.shape[-1]
d_out = 2
print(d_in, d_out)
```

```python
batch = torch.stack((inputs,), dim=0)
print(batch.shape)
```

## Class for implementing multi-head attention

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

```python
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
```

```python
context_vecs = mha(batch)
print(context_vecs)
print('context_vecs.shape:', context_vecs.shape)
```

## Implementing multi-head attention with weight splits

Instead of maintaining two separate classes, MultiHeadAttentionWrapper and CausalAttention, we can combine both of these concepts into a single MultiHeadAttention class.

* Step 1: Reduce the projection dim to match desired output dim
* Step 2: Use a linear layer to combine head outputs
* Step 3: Tensor shape: (b, num_tokens, d_out)
* Step 4: We implicitly split the matrix by adding a num_heads dimension. Then we unroll last dim: (b, num_tokens, head_dim)
* Step 5: Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
* Step 6: Compute dot product of each head
* Step 7: Mask truncated to the number of tokens
* Step 8: Use the mask to fill attention scores
* Step 9: Tensor shape: (b, num_tokens, n_heads, head_dim)
* Step 10: Combine heads, where self.d_out = self.num_heads * self.head_dim
* Step 11: Add an optional linear projection 

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_in) # Linear layer to combine
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(
                context_length, context_length
            ), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x) # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a 'num_heads' dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3) # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the maks to fill attention scores
        attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
```

```python
torch.manual_seed(123)

# Define the tensor with 3 rows and 6 columns
inputs = torch.tensor([
    [0.43, 0.15, 0.89, 0.55, 0.87, 0.66], # Row 1
    [0.57, 0.85, 0.64, 0.22, 0.58, 0.33], # Row 2
    [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]  # Row 3
])

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
print(batch)

batch_size, context_length, d_in = batch.shape
d_out = 6
context_length = inputs.shape[0]
dropout = 0.0
num_heads = 2

mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads, True)

context_vecs = mha(batch)
print(context_vecs)
print('context_vecs.shape:', context_vecs.shape)
```

## Stepwise implementation of multi-head attention

```python
import torch

# Input: (batch=1, seq_len=3, d_model=6)
x = torch.tensor([[
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], # The 
    [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], # Kid
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Smiles
]])

batch_size, seq_len, d_model = x.shape
```

```python
# Define 6x6 projection matrices (d_model x d_model)
torch.manual_seed(0) # for reprducibility
Wq = torch.randn(d_model, d_model)
Wk = torch.randn(d_model, d_model)
Wv = torch.randn(d_model, d_model)

# Compute Q, K, V
# Shape logic: (B, T, d_model) @ (d_model, d_model) -> (B, T, d_model)
Q = x @ Wq
K = x @ Wk
V = x @ Wv

# Print Q, K, and V
print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)

# Print dimensionalities
print('X shape:', x.shape)      # (1, 3, 6)
print('Wq shape:', Wq.shape)    # (6, 6)
print('Wk shape:', Wk.shape)    # (6, 6)
print('Wv shape:', Wv.shape)    # (6, 6)
print('Q shape:', Q.shape)      # (1, 3, 6)
print('K shape:', K.shape)      # (1, 3, 6)
print('V shape:', V.shape)      # (1, 3, 6)
```

```python
# Print values
torch.set_printoptions(precision=2)
print("\nWq:\n", Wq)
print("\nWk:\n", Wk)
print("\nWv:\n", Wv)

print("\nQ:\n", Q)
print("\nK:\n", K)
print("\nV:\n", V)
```

```python
num_heads = 2
head_dim = 3

Q = Q.view(1, 3, num_heads, head_dim)
K = K.view(1, 3, num_heads, head_dim)
V = V.view(1, 3, num_heads, head_dim)

print('Q after unrolling:', Q)
print('K after unrolling:', K)
print('V after unrolling:', V)
```

```python
Q = Q.transpose(1, 2)
K = K.transpose(1, 2)
V = V.transpose(1, 2)

print('Q after grouping by heads:', Q)
print('K after grouping by heads:', K)
print('V after grouping by heads:', V)
```

```python
K_T = K.transpose(2, 3)
print('K_T shape:', K_T)
```

```python
attn_scores = Q @ K_T
print('Attention scores shape:', attn_scores.shape)
print('Attention scores:\n', attn_scores)
```

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print('Causal mask:\n', mask)

attn_scores.masked_fill_(mask, -torch.inf)
print('Attention scores after masking:\n', attn_scores)
```

```python
torch.set_printoptions(precision=3, sci_mode=False)
head_dim = 3
attn_weights = torch.softmax(attn_scores / head_dim**0.5, dim=-1)
print('Attention weights shape:', attn_weights.shape)
print('Attention weights:\n', attn_weights)
```

```python
dropout = torch.nn.Dropout(0.1)
attn_weights = dropout(attn_weights)
print('Attention weights after dropout:\n', attn_weights)
```

```python
context_vec = attn_weights @ V
print('Context vector shape:', context_vec.shape)
print('Context vec:\n', context_vec)
```

### Step 11: Reformat and concatenate

```python
context_vec = context_vec.transpose(1, 2)
print('Context vector after swapping dimensions 1 and 2:', context_vec.shape)
print('Context vector:\n', context_vec)
```

```python
context_vec = context_vec.reshape(batch_size, seq_len, num_heads * head_dim)
print('Context vector after concatenating heads:', context_vec.shape)
print('Context vector:\n', context_vec)
```
