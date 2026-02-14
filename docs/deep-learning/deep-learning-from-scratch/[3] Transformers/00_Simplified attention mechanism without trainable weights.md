[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/deep-learning-from-scratch/[3] Transformers/00_Simplified attention mechanism without trainable weights.ipynb)

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

# Input text

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

## Magnitude of vectors

```python
# Calculate the magnitude of each vector
magnitudes = torch.norm(inputs, dim=1)

# Print the magnitudes
print('Magnitudes of the vectors:')
for word, magnitude in zip(words, magnitudes):
    print(f'{word}: {magnitude.item():.4f}')
```

## Plotting the 3D vectors

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extract x, y, z coordinates
x_coords = inputs[:, 0].numpy()
y_coords = inputs[:, 1].numpy()
z_coords = inputs[:, 2].numpy()

# 3D plot with vectors from origin
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'c', 'm', 'y']

for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()
```

## Dot product between 2nd input token and all words

```python
query = inputs[1] # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here)

print(attn_scores_2)
```

## Normalize the attention weights

```python
# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

# print('Attention weights:', attn_weights_2_tmp)
# print('Sum:', attn_weights_2_tmp.sum())
```

## Converting to softmax probabilities

```python
from torch import  nn

# Convert attn_scores_2 to softmax
softmax = nn.Softmax(dim=-1)
attn_weights_2_tmp = softmax(attn_scores_2)

print('Attention weights:', attn_weights_2_tmp)
print('Sum:', attn_weights_2_tmp.numpy().sum())
```

## Attention scores for all queries

```python
attn_scores = inputs @ inputs.T # @ for efficient matrix multiplication
print(attn_scores)
```

## Attention scores converted to attention weights

```python
attn_weights = softmax(attn_scores)
print(attn_weights)
```

## Sum of each row is 1

```python
row_2_sum = sum([0.1624, 0.1803, 0.1336, 0.2059, 0.1715, 0.1462])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
```

## Print the context vector for all queries

```python
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
```

# Context vector corresponding to the 2nd input

```python
context_vec_2 = all_context_vecs[1]
print('Previous 2nd context vector:', context_vec_2)
```

## Adding context vector just for plotting

```python
# Append context_vec_2 to inputs
inputs = torch.cat((inputs, context_vec_2.unsqueeze(0)), dim=0)

# Add 'context_vector' to the words list
words.append('context_vector_for_big')

print('Updated input tensor:')
print(inputs)
print('\nUpdated input tensor:')
print(words)
```

```python
# Extract x, y, z coordinates
x_coords = inputs[:, 0].numpy()
y_coords = inputs[:, 1].numpy()
z_coords = inputs[:, 2].numpy()

# 3D plot with vectors from origin
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()
```
