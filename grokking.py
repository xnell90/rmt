import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from tqdm import tqdm

# 1. Configuration & Hyperparameters (Based on Liu et al., "Omnigrok")
SEED = 42
TRAIN_SIZE = 1000         # Small subset to force initial memorization
BATCH_SIZE = 200
LR = 1e-3
WEIGHT_DECAY = 0.01       # Crucial driver for grokking
INIT_SCALE = 8.0          # Large initialization delays generalization
STEPS = 100000             # Total optimization steps
LOG_INTERVAL = 1000

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# 2. Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Subsample training set to 1,000 examples
train_subset = Subset(full_train_dataset, range(TRAIN_SIZE))

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. Define MLP Architecture with Scaled Initialization
class GrokkingMLP(nn.Module):
    def __init__(self, init_scale=8.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )
        # Scale initial weights to induce initial memorization
        with torch.no_grad():
            for param in self.parameters():
                param.mul_(init_scale)

    def forward(self, x):
        return self.net(x)

model = GrokkingMLP(init_scale=INIT_SCALE).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()

# Helper function to evaluate accuracy and weight norm
@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total

def get_weight_norm(model):
    return sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5

# 4. Training Loop (Step-based)
history = {"step": [], "train_acc": [], "test_acc": [], "weight_norm": []}
train_iter = iter(train_loader)

print("Starting training... Watch for train_acc hitting 1.0 early, followed by test_acc jumping later.")

for step in tqdm(range(STEPS), desc='Progress', disable=True):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    model.train()
    x, y = x.to(device), y.to(device)
    y_onehot = F.one_hot(y, num_classes=10).float()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y_onehot)
    loss.backward()
    optimizer.step()

    # Log metrics periodically
    if step % LOG_INTERVAL == 0 or step == 1:
        train_acc = evaluate(train_loader)
        test_acc = evaluate(test_loader)
        w_norm = get_weight_norm(model)

        history["step"].append(step)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["weight_norm"].append(w_norm)

        print(f"Step {step:5d}/{STEPS} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Weight Norm: {w_norm:.2f}")

# 5. Plot the Grokking Phase Transition
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xscale("log")
ax1.set_xlabel("Optimization Steps (Log Scale)")
ax1.set_ylabel("Accuracy", color="tab:blue")
ax1.plot(history["step"], history["train_acc"], label="Train Accuracy", color="tab:blue", linewidth=2)
ax1.plot(history["step"], history["test_acc"], label="Test Accuracy (Grokking)", color="tab:orange", linewidth=2)
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, which="both", linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
ax2.set_ylabel("L2 Weight Norm", color="tab:green")
ax2.plot(history["step"], history["weight_norm"], label="Weight Norm", color="tab:green", linestyle=":", linewidth=2)
ax2.tick_params(axis="y", labelcolor="tab:green")

fig.suptitle("Grokking on MNIST (Omnigrok Setup)", fontsize=14)
fig.legend(loc="center left", bbox_to_anchor=(0.15, 0.5))
plt.tight_layout()
plt.savefig("mnist_grokking.png", dpi=150)
print("Saved plot to mnist_grokking.png")
plt.show()
