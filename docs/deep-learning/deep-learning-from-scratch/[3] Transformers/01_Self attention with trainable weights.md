[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/deep-learning-from-scratch/[3] Transformers/01_Self attention with trainable weights.ipynb)

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

## We want to generate the context vector for 2nd token

```python
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 #C = dimensionality of the context vector
print(x_2)
print(d_in)
print(d_out)
```

## Randomly initializing Wq, Wk, Wv matrices

```python
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```

```python
print(W_query)
```

```python
print(W_key)
```

```python
print(W_value)
```

```python
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
print(key_2)
print(value_2)
```

## Calculating Q, K, and V using X, Wq, Wk, Wv

```python
keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query

print('keys.shape', keys.shape)
print('values.shape', values.shape)
print('queries.shape', queries.shape)

print('keys:', keys)
print('queries', queries)
print('values', values)
```

## Keys corresponding to second token and the attention of second token to itself

```python
keys_2 = keys[1]
attn_score_22 = query_2 @ keys_2
print(attn_score_22)
```

## All attention scores for query number 2

```python
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)
```

## Attention scores (NOT Weights) matrix

```python
attn_scores = queries @ keys.T # omega
print(attn_scores)
```

## Scale by 1/sqrt(d) and then take softmax

```python
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
print(attn_weights)
```

## Softmax peaks when the numbers are scaled

```python
import torch
# Define the tensor
tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

# Apply softmax
# Apply softmax without scaling
softmax_result = torch.softmax(tensor, dim=-1)
print("Softmax without scaling:", softmax_result)

# Multiply the tensor by 8 and the apply softmax
scaled_tensor = tensor * 8
softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)
print("Softmax after scaling (tensor * 8):", softmax_scaled_result)
```

## Scaling has to be such that the variance of Q*K.T is close to 1

```python
import numpy as np

# Function to compute variance before and after scaling
def compute_variance(dim, num_trials=1000):
    dot_products = []
    scaled_dot_products = []

    # Generate multiple random vectors and compute dot products
    for _ in range(num_trials):
        q = np.random.randn(dim)
        k = np.random.randn(dim)

        # Compute dot product
        dot_product = q @ k
        dot_products.append(dot_product)

        # Scale the dot product by sqrt (dim)
        scaled_dot_product = dot_product / (dim)
        scaled_dot_products.append(scaled_dot_product)
          
    # Calculate the variance of the dot products
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling

# For dimension 5
variance_before_5, variance_after_5 = compute_variance(5)
print(f"Variance before scaling (dim=5): {variance_before_5}")
print(f"Variance after scaling (dim=5): {variance_after_5}")

# For dimension 100
variance_before_100, variance_after_100 = compute_variance(100)
print(f"Variance before scaling (dim=100): {variance_before_100}")
print(f"Variance after scaling (dim=100): {variance_after_100}")
```

## Context vector corresponding to 2nd input token

```python
context_vec_2 = attn_weights_2 @ values
context_vec = attn_weights @ values
print(context_vec_2)
print(context_vec)
```

## Python class for doing this whole operation

```python
import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
```

```python
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
```

```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)  
        queries = self.W_query(x) 
        values = self.W_value(x)  

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        context_vec = torch.matmul(attn_weights, values)  # [batch, seq_len, d_out]
        return context_vec

# Example test
torch.manual_seed(123)
sa_v2 = SelfAttention_v1(d_in, d_out)
print(sa_v2(inputs))
```
