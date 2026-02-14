[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/deep-learning-from-scratch/[4] Vision Transformer (ViT)/Coding ViT from scratch.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="CON9ZsA0oaKl" -->
<img src="https://imgs.search.brave.com/J6D4toInIozipgRabyeqg22NqJ4zmhQMjnOoOgWXjeQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tYXVj/aGVyLmhvbWUuaGRt/LXN0dXR0Z2FydC5k/ZS9QaWNzL3Zpc2lv/blRyYW5zZm9ybWVy/T3ZlcnZpZXcucG5n">
<!-- #endregion -->

<!-- #region id="0rAK-ME8gwNz" -->
## Import libraires
<!-- #endregion -->

```python executionInfo={"elapsed": 72, "status": "ok", "timestamp": 1764507984616, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="zyZwqi9ig3qa"
import torch
from torch import nn
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
```

<!-- #region id="gJQiiGHriU0q" -->
## Define the transformation
<!-- #endregion -->

```python executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1764507984992, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="H-UMiRnuiZJj"
# Transformation for PIL to tensor format
transform = transforms.Compose(
    [transforms.ToTensor()]
)
```

<!-- #region id="B0ZGBtzHh8Fn" -->
## Download the data
<!-- #endregion -->

```python executionInfo={"elapsed": 62, "status": "ok", "timestamp": 1764507985390, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="3hG2tdouiE8j"
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
val_dataset = MNIST(root='./data', train=False, transform=transform, download=False)
```

<!-- #region id="B0okpNADjbOV" -->
## Define the parameters
<!-- #endregion -->

```python executionInfo={"elapsed": 37, "status": "ok", "timestamp": 1764507985687, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="t_JALZ9Xjesk"
num_classes = 10
batch_size = 64
num_channels = 1
img_size = 28
patch_size = 7
num_patches = (img_size // patch_size) ** 2
embedding_dim = 64
attention_heads = 4
transformer_blocks = 4
mlp_hidden_nodes = 128
learning_rate = 0.001
epochs = 5
```

<!-- #region id="FwclLkgKjNbM" -->
## Define the dataloader
<!-- #endregion -->

```python executionInfo={"elapsed": 46, "status": "ok", "timestamp": 1764507986033, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="BHZGRdmTjRvk"
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

<!-- #region id="44aj-sqYlAql" -->
## Part 1: Patch embedding
<!-- #endregion -->

```python executionInfo={"elapsed": 41, "status": "ok", "timestamp": 1764507986386, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="IPxvmWmElE9E"
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(num_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1764507986495, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="VGgBf_u5mY41" outputId="b4e5a15d-be14-4b9f-d331-be4fd61089eb"
# Sample a data point from the train_loader
data_point, label = next(iter(train_loader))
print("Shape of data point:", data_point.shape)

patch_embed = nn.Conv2d(num_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
patch_embed_output = patch_embed(data_point)
print(patch_embed(data_point).shape)

patch_embed_output_flatten = patch_embed_output.flatten(2)
print(patch_embed_output_flatten.shape)

patch_embed_output_flatten_transpose = patch_embed_output_flatten.transpose(1, 2)
print(patch_embed_output_flatten_transpose.shape)
```

<!-- #region id="Zjx1EpGmn27m" -->
## Part 2: Transformer encoder
<!-- #endregion -->

```python executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1764507986865, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="ifbBeSn4pvCs"
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_nodes),
            nn.GELU(),
            nn.Linear(mlp_hidden_nodes, embedding_dim)
        )

    def forward(self, x):
        residual1 = x
        x = self.layer_norm1(x)
        x = self.multihead_attention(query=x, key=x, value=x)[0]
        x = x + residual1

        residual2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual2

        return x
```

<!-- #region id="OY2HhjzLslPf" -->
## Part 3: MLP head
<!-- #endregion -->

```python executionInfo={"elapsed": 35, "status": "ok", "timestamp": 1764507987593, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="3iKf7XvIsq3T"
class MLP_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.mlp_head(x)

        return x
```

<!-- #region id="6H_oQraEteRG" -->
## Part 4: Vision Transformer
<!-- #endregion -->

```python executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1764507988352, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="ExzeQqW2thmb"
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))
        self.transformer_blocks = nn.Sequential(*[TransformerEncoder() for _ in range(transformer_blocks)])
        self.mlp_head = MLP_head()

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.size(0)
        class_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.transformer_blocks(x)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x
```

<!-- #region id="ZqSmOLzfxWL8" -->
## Training
<!-- #endregion -->

```python executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1764507989567, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="OX3VVVRBxYBV"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VisionTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 82519, "status": "ok", "timestamp": 1764508072628, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="km1OSgV5xvd9" outputId="ff5a1b85-3a13-4076-e833-a8c0c852bae2"
# Training
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_epoch = 0
    total_epoch = 0

    print(f'\nEpoch {epoch+1}')

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        accuracy = 100.0 * (correct / labels.size(0))

        correct_epoch += correct
        total_epoch += labels.size(0)

        if batch_idx % 100 ==0:
            print(f'Batch {batch_idx+1:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%')

    epoch_acc = 100.0 * correct_epoch / total_epoch
    print(f'==> Epoch {epoch+1} Summary: Total Loss = {total_loss:.4f}, Accuracy = {epoch_acc:.2f}%')

```

<!-- #region id="8lx3pBcW4caW" -->
## Validation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1439, "status": "ok", "timestamp": 1764508083819, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="E4Nyn3iO4Gl2" outputId="6de047f8-5564-4c3e-862e-13dbbe73d284"
# Validation
model.eval()
val_loss = 0
correct_val = 0
total_val = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct_val += (preds == labels).sum().item()
        total_val += labels.size(0)

val_acc = 100.0 * correct_val / total_val
avg_val_loss = val_loss / len(val_loader)

print(f'>> Validation: Loss = {avg_val_loss:.4f}, Accuracy = {val_acc:.2f}%')

```

<!-- #region id="S4SSAPFM4bG-" -->
## Plotting
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 582} executionInfo={"elapsed": 679, "status": "ok", "timestamp": 1764508208261, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="5oD_ISZY4al-" outputId="6f5efe38-b162-49e9-e00f-614774265c63"
import matplotlib.pyplot as plt
import random

def show_random_predictions(model, dataloader, class_names=None, num_images=10):
    model.eval()
    images_shown = 0

    # grab one big batch so we can pick random samples from it
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    # run through model
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    # randomly pick indices
    idxs = random.sample(range(len(images)), num_images)

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(idxs):
        img = images[idx].cpu().squeeze(0)  # MNIST-like (1, H, W)
        pred = preds[idx].item()
        true = labels[idx].item()

        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Pred: {pred}\nTrue: {true}",
                  color="green" if pred == true else "red")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

show_random_predictions(model, val_loader)

```
