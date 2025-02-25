import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from itertools import product, combinations
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

# Set up a directory to save outputs.
SAVE_DIR = "local_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# 1. Simple CNN (MNIST) with Flexible Layer Analysis
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        """
        A simple CNN for MNIST.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # For layer-wise analysis.
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1   = nn.Linear(32 * 7 * 7, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =============================================================================
# 2. Differentiable Vietoris–Rips Module (Supports up to 2–simplices)
# =============================================================================
class DifferentiableVietorisRips(nn.Module):
    def __init__(self, epsilon, vr_dim=2, sharpness=10.0):
        """
        Constructs a soft, differentiable Vietoris–Rips complex.
        Generates 0–simplices, 1–simplices, and if vr_dim>=2, 2–simplices.
        Args:
            epsilon: Distance threshold.
            vr_dim: Maximum simplex dimension.
            sharpness: Controls steepness of the sigmoid.
        """
        super(DifferentiableVietorisRips, self).__init__()
        self.epsilon   = epsilon
        self.vr_dim    = vr_dim
        self.sharpness = sharpness

    def forward(self, W):
        N = W.size(0)
        device = W.device
        masks = []

        # 0–simplices: each filter alone.
        for i in range(N):
            mask = torch.zeros(N, device=device)
            mask[i] = 1.0
            masks.append(mask)

        # Use NumPy for distance computation.
        W_np = W.detach().cpu().numpy()

        # 1–simplices: pairs.
        for i, j in combinations(range(N), 2):
            dist = np.linalg.norm(W_np[i] - W_np[j])
            val  = 1.0 / (1.0 + np.exp(self.sharpness * (dist - self.epsilon)))
            if val > 1e-6:
                mask = torch.zeros(N, device=device)
                mask[i] = val
                mask[j] = val
                masks.append(mask)

        # 2–simplices: triplets.
        if self.vr_dim >= 2:
            for i, j, k in combinations(range(N), 3):
                d_ij = np.linalg.norm(W_np[i] - W_np[j])
                d_jk = np.linalg.norm(W_np[j] - W_np[k])
                d_ik = np.linalg.norm(W_np[i] - W_np[k])
                if (d_ij <= self.epsilon) and (d_jk <= self.epsilon) and (d_ik <= self.epsilon):
                    mask = torch.zeros(N, device=device)
                    mask[i] = 1.0
                    mask[j] = 1.0
                    mask[k] = 1.0
                    masks.append(mask)

        M = torch.stack(masks, dim=0)
        return M

# =============================================================================
# 3. Outstanding Mapping Architecture using Transformer Encoder
# =============================================================================
class OutstandingMapping(nn.Module):
    def __init__(self, input_dim, context_dim, nhead=4, num_layers=1):
        """
        A mapping module based on a Transformer encoder.
        Args:
            input_dim: Input dimension (e.g., 25).
            context_dim: Desired embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
        """
        super(OutstandingMapping, self).__init__()
        self.linear_in = nn.Linear(input_dim, context_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=context_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(context_dim, context_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = x.unsqueeze(1)  # (num_simplices, 1, context_dim)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)    # (num_simplices, context_dim)
        x = self.linear_out(x)
        return x

# =============================================================================
# 4. HyperCategory Framework with Outstanding Mapping
# =============================================================================
class HyperCategoryFramework(nn.Module):
    def __init__(self, base_model, layer_name, context_dim, num_classes, epsilon, vr_dim):
        """
        Implements HCNN = (W, VR_ε(W), C, Φ) using an outstanding mapping.
        Args:
            base_model: Underlying neural network.
            layer_name: Name of the convolutional layer to analyze (e.g., "conv1" or "conv2").
            context_dim: Dimension of context embeddings.
            num_classes: Number of output classes.
            epsilon: VR threshold.
            vr_dim: Maximum simplex dimension.
        """
        super(HyperCategoryFramework, self).__init__()
        self.base_model   = base_model
        self.layer_name   = layer_name
        self.context_dim  = context_dim
        self.epsilon      = epsilon
        self.vr_dim       = vr_dim
        self.num_classes  = num_classes

        self.vr_complex   = DifferentiableVietorisRips(epsilon=epsilon, vr_dim=vr_dim, sharpness=10.0)
        self.φ            = OutstandingMapping(input_dim=self._weight_dim(), context_dim=context_dim)
        self.Ω            = Parameter(torch.randn(context_dim, context_dim))
        self.Λ            = nn.Linear(context_dim, context_dim)
        self.classifier_adjust = nn.Linear(context_dim, num_classes)

    def _weight_dim(self):
        layer = getattr(self.base_model, self.layer_name)
        return layer.weight[0].numel()

    def forward(self, x):
        y = self.base_model(x)
        W = self._get_differentiable_weights()
        M = self.vr_complex(W)
        contexts = self._functorial_mapping(W, M)
        H = self._hypercomposition(contexts, M)
        cat_proj = self._categorical_projection(H)
        adjustment = self.classifier_adjust(cat_proj)
        return y + adjustment

    def _get_differentiable_weights(self):
        layer = getattr(self.base_model, self.layer_name)
        weights = layer.weight
        return weights.view(weights.size(0), -1)

    def _functorial_mapping(self, W, M):
        s = M
        s_sum = s.sum(dim=1, keepdim=True)
        s_norm = s_sum + 1e-6
        pooled = (s.unsqueeze(2) * W.unsqueeze(0)).sum(dim=1) / s_norm
        contexts = self.φ(pooled)
        return contexts

    def _hypercomposition(self, contexts, M):
        orders = M.sum(dim=1, keepdim=True)
        is_pair = torch.sigmoid((orders - 2.0) * 10.0)
        transformed = F.gelu(torch.matmul(contexts, self.Ω) + self.Λ(contexts))
        H = is_pair * transformed + (1 - is_pair) * contexts
        return H

    def _categorical_projection(self, H):
        proj = torch.matmul(H.mean(dim=0, keepdim=True), self.Ω.T)
        return proj

# =============================================================================
# 5. Adaptive Margin-based Topological Regularizer
# =============================================================================
class TopologicalRegularizer(nn.Module):
    def __init__(self, base_margin=1.0, adapt_factor=0.5):
        """
        Enforces a contrastive loss on overlapping context embeddings using an adaptive margin.
        Args:
            base_margin: Base margin value.
            adapt_factor: Fraction of the average pairwise distance to add to the margin.
        """
        super(TopologicalRegularizer, self).__init__()
        self.base_margin = base_margin
        self.adapt_factor = adapt_factor

    def forward(self, contexts, M):
        num = M.size(0)
        M1 = M.unsqueeze(1)
        M2 = M.unsqueeze(0)
        inter = (M1 * M2).sum(dim=2)
        union = M1.sum(dim=2) + M2.sum(dim=2) - inter + 1e-6
        weights = inter / union
        diff = contexts.unsqueeze(1) - contexts.unsqueeze(0)
        diff_norm = torch.norm(diff, p=2, dim=2)
        avg_distance = diff_norm.mean()
        adaptive_margin = self.base_margin + self.adapt_factor * avg_distance
        pos_mask = (weights > 0.05).float()
        loss_matrix = torch.clamp(adaptive_margin - diff_norm, min=0) ** 2
        loss = (loss_matrix * pos_mask).sum() / (pos_mask.sum() + 1e-6)
        return loss

# =============================================================================
# 6. Hyperparameter Tuning (Grid Search)
# =============================================================================
def hyperparameter_tuning(device, base_model, layer_name, context_dim, num_classes, vr_dim, dataloader, num_batches=5):
    epsilons = [0.3, 0.5, 0.7]
    base_margins = [0.5, 1.0, 1.5]
    adapt_factors = [0.1, 0.5, 1.0]
    best_combo = None
    best_loss = -1e9
    for epsilon, base_margin, adapt_factor in product(epsilons, base_margins, adapt_factors):
        total_loss = 0.0
        count = 0
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            model_tune = HyperCategoryFramework(
                base_model=base_model,
                layer_name=layer_name,
                context_dim=context_dim,
                num_classes=num_classes,
                epsilon=epsilon,
                vr_dim=vr_dim
            ).to(device)
            model_tune.eval()
            with torch.no_grad():
                _ = model_tune(inputs)
                W = model_tune._get_differentiable_weights()
                M = model_tune.vr_complex(W)
                contexts = model_tune._functorial_mapping(W, M)
                reg = TopologicalRegularizer(base_margin=base_margin, adapt_factor=adapt_factor).to(device)
                loss = reg(contexts, M)
                total_loss += loss.item()
                count += 1
        avg_loss = total_loss / (count if count > 0 else 1)
        print(f"Tuning: eps={epsilon}, margin={base_margin}, adapt={adapt_factor} --> Cat Loss: {avg_loss:.4f}")
        if avg_loss > best_loss:
            best_loss = avg_loss
            best_combo = (epsilon, base_margin, adapt_factor)
    print(f"Selected: eps={best_combo[0]}, margin={best_combo[1]}, adapt={best_combo[2]}, Cat Loss: {best_loss:.4f}")
    return best_combo

# =============================================================================
# 7. Visualization Functions for Interpretability
# =============================================================================
def visualize_context_embeddings(contexts, save_prefix="layer"):
    from sklearn.manifold import TSNE
    contexts_np = contexts.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    contexts_2d = tsne.fit_transform(contexts_np)
    plt.figure(figsize=(8,6))
    plt.scatter(contexts_2d[:,0], contexts_2d[:,1], c='blue', alpha=0.7)
    plt.title("t-SNE of Learned Context Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    path = os.path.join(SAVE_DIR, f"{save_prefix}_context_embeddings.png")
    plt.savefig(path)
    plt.show()
    print(f"Saved t-SNE embedding figure to {path}")

def visualize_vr_complex(W, M, save_prefix="layer"):
    N = W.size(0)
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    M_np = M.detach().cpu().numpy()
    for idx, mask in enumerate(M_np):
        idxs = np.where(mask>0.5)[0]
        if len(idxs)==2:
            G.add_edge(idxs[0], idxs[1])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Vietoris–Rips Complex (0– and 1–simplices)")
    path = os.path.join(SAVE_DIR, f"{save_prefix}_vr_complex_graph.png")
    plt.savefig(path)
    plt.show()
    print(f"Saved VR complex graph figure to {path}")

    # Store 2–simplices adjacency
    triangles = []
    for idx, mask in enumerate(M_np):
        idxs = np.where(mask>0.5)[0]
        if len(idxs)==3:
            triangles.append([int(i) for i in idxs])
    tri_path = os.path.join(SAVE_DIR, f"{save_prefix}_2simplices.json")
    with open(tri_path, "w") as f:
        json.dump(triangles, f, indent=2)
    print(f"Saved 2–simplices adjacency to {tri_path}")

def visualize_conv_filters(layer_weights, save_prefix="layer"):
    import math
    N = layer_weights.size(0)
    grid_size = int(math.ceil(N**0.5))
    plt.figure(figsize=(grid_size*1.5, grid_size*1.5))
    for i in range(N):
        plt.subplot(grid_size, grid_size, i+1)
        filt = layer_weights[i, 0].detach().cpu().numpy()
        plt.imshow(filt, cmap="gray")
        plt.axis("off")
    path = os.path.join(SAVE_DIR, f"{save_prefix}_filters.png")
    plt.savefig(path)
    plt.show()
    print(f"Saved filter images to {path}")

def print_hypercategory_structure(W, M, save_prefix="layer"):
    N = W.size(0)
    M_np = M.detach().cpu().numpy()
    lines = []
    lines.append("=== Hypercategory Summary ===")
    lines.append(f"Number of base objects (filters) = {N}")
    lines.append("Objects (0–simplices): filters 0..N-1")
    one_simp = []
    two_simp = []
    for idx, mask in enumerate(M_np):
        idxs = np.where(mask>0.5)[0]
        if len(idxs)==2:
            one_simp.append(list(idxs))
        elif len(idxs)==3:
            two_simp.append(list(idxs))
    lines.append(f"\nNumber of 1–simplices (pairs) = {len(one_simp)}")
    for pair in one_simp:
        lines.append(f"  - 1-simplex: {pair}")
    lines.append(f"\nNumber of 2–simplices (triplets) = {len(two_simp)}")
    for tri in two_simp:
        lines.append(f"  - 2-simplex: {tri}")
    lines.append("\n=== Naive Compositional Overlaps ===")
    for tri in two_simp:
        tri_sorted = sorted(tri)
        i, j, k = tri_sorted
        lines.append(f"  2-simplex {tri_sorted} => possible composition: [({i},{j}) + ({j},{k})] => ({i},{k})")
    text_path = os.path.join(SAVE_DIR, f"{save_prefix}_hypercategory.txt")
    with open(text_path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Hypercategory summary saved to {text_path}")

# =============================================================================
# MAIN TRAINING AND ANALYSIS
# =============================================================================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create base model.
    base_model = SimpleCNN().to(device)
    # Analyze all layers.
    layers_to_analyze = ["conv1", "conv2"]

    # Dictionary to store hypercategory summaries per layer.
    hypercategory_summaries = {}

    # Hyperparameter Tuning & Analysis for each layer.
    context_dim = 256
    num_classes = 10
    vr_dim = 2  # up to 2–simplices

    for layer_name in layers_to_analyze:
        print(f"\n=== Analyzing layer: {layer_name} ===")
        best_epsilon, best_margin, best_adapt = hyperparameter_tuning(
            device, base_model, layer_name, context_dim, num_classes, vr_dim, dataloader, num_batches=3
        )

        model = HyperCategoryFramework(
            base_model=base_model,
            layer_name=layer_name,
            context_dim=context_dim,
            num_classes=num_classes,
            epsilon=best_epsilon,
            vr_dim=vr_dim
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        regularizer = TopologicalRegularizer(base_margin=best_margin, adapt_factor=best_adapt).to(device)

        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                task_loss = F.cross_entropy(outputs, targets)
                W = model._get_differentiable_weights()
                M = model.vr_complex(W)
                contexts = model._functorial_mapping(W, M)
                cat_loss = regularizer(contexts, M)
                total_loss = task_loss + cat_loss
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
                if batch_idx % 100 == 0:
                    print(f"Layer [{layer_name}] Epoch [{epoch+1}/3], Batch [{batch_idx}/{len(dataloader)}], "
                          f"Task Loss: {task_loss.item():.4f}, Cat Loss: {cat_loss.item():.4f}, "
                          f"Total Loss: {total_loss.item():.4f}")
            print(f"Layer [{layer_name}] Epoch [{epoch+1}/3], Average Loss: {running_loss/len(dataloader):.4f}")

        # Save model and regularizer for this layer.
        os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "checkpoints", f"hcnn_{layer_name}.pth"))
        torch.save(regularizer.state_dict(), os.path.join(SAVE_DIR, "checkpoints", f"regularizer_{layer_name}.pth"))
        print(f"Saved model and regularizer for {layer_name} to disk.")

        # Visualization and Hypercategory Summary.
        model.eval()
        with torch.no_grad():
            W = model._get_differentiable_weights()
            M = model.vr_complex(W)
            contexts = model._functorial_mapping(W, M)
        visualize_context_embeddings(contexts, save_prefix=layer_name)
        visualize_vr_complex(W, M, save_prefix=layer_name)

        layer_obj = getattr(base_model, layer_name)
        if layer_obj.weight.size(1) == 1:
            visualize_conv_filters(layer_obj.weight, save_prefix=layer_name)
        else:
            w_sub = layer_obj.weight[:, 0, :, :].unsqueeze(1)
            visualize_conv_filters(w_sub, save_prefix=layer_name)

        print_hypercategory_structure(W, M, save_prefix=layer_name)
        # Optionally, save the context embeddings and VR masks to disk.
        torch.save(W, os.path.join(SAVE_DIR, f"{layer_name}_weights.pt"))
        torch.save(M, os.path.join(SAVE_DIR, f"{layer_name}_vr_masks.pt"))
        np.save(os.path.join(SAVE_DIR, f"{layer_name}_contexts.npy"), contexts.detach().cpu().numpy())
        print(f"Saved weights, VR masks, and context embeddings for {layer_name}.")

        hypercategory_summaries[layer_name] = {
            "num_filters": int(W.size(0)),
            "num_1simplices": sum(1 for mask in M.detach().cpu().numpy() if np.sum(mask > 0.5)==2),
            "num_2simplices": sum(1 for mask in M.detach().cpu().numpy() if np.sum(mask > 0.5)==3)
        }

    # Save the overall hypercategory summary.
    summary_path = os.path.join(SAVE_DIR, "hypercategory_summary.json")
    with open(summary_path, "w") as f:
        json.dump(hypercategory_summaries, f, indent=2)
    print("All analyses complete. Results saved to:", SAVE_DIR)
