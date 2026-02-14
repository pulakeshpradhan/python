[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/deep-learning-from-scratch/[3] Transformers/02_Causal attention.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: torch
    language: python
    name: python3
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

## Class for implementing self attention

```python
from torch import nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
```

## Context vectors corresponding to inputs

```python
d_in = inputs.shape[-1]
d_out = 2

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
```

## Final attention weights

```python
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
```

## Lower triangular matrix (mask)

```python
context_length = inputs.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
```

## Attention weights after applying mask

```python
masked_simple = attn_weights * mask_simple
print(masked_simple)
```

## Attention weights normalized

```python
row_sums = masked_simple.sum(dim=-1, keepdim=True)
print(row_sums.shape)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
```

## Attention scores

```python
print(attn_scores)
```

## Upper triangular matrix

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)
```

## Ones are converted to -ve infinity

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
```

## Attention weights after taking softmax

```python
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
```

## Ones matrix

```python
example = torch.ones(inputs.shape[0], inputs.shape[0]) #B
print(example)
```

## Random dropout with 50% probability

```python
torch.manual_seed(123)
dropout = nn.Dropout(0.5) #A
dropout(example)
```

## Attention weights after dropout mask

```python
torch.manual_seed(123)
print(dropout(attn_weights))
```
