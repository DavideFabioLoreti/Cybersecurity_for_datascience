"""
Adversarial Attacks e Difese su Deep Learning Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import foolbox as fb
from tqdm import tqdm

# =====================================================
# 1. SETUP
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 100
EPOCHS = 5

# Trasformazioni
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


# =====================================================
# 2. MODELLO
# =====================================================
def create_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model.to(device)


# =====================================================
# 2b. PGD Adversarial Attack (per difesa)
# =====================================================
def pgd_attack(model, images, labels, eps=0.03, alpha=0.007, steps=10):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    delta = torch.zeros_like(images).uniform_(-eps, eps).to(device)
    delta.requires_grad = True

    for _ in range(steps):
        outputs = model(images + delta)
        cost = loss(outputs, labels)
        cost.backward()

        delta.data = (delta + alpha * delta.grad.sign()).clamp(-eps, eps)
        delta.grad.zero_()

    return (images + delta).detach()


# =====================================================
# 3. TRAIN STANDARD + ADVERSARIAL TRAINING OPZIONALE
# =====================================================
def train_model(model, epochs=EPOCHS, adversarial_training=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # Adversarial Training
            if adversarial_training:
                x_adv = pgd_attack(model, x, y)
                x = 0.5 * x + 0.5 * x_adv  # mix clean + adversarial

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

            pbar.set_postfix({'acc': 100 * correct / total})

        scheduler.step()
    return model


# =====================================================
# 4. EVALUATION + ATTACCHI ADVERSARIAL
# =====================================================
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, preds = model(x).max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def run_adversarial_attacks(model, test_loader, epsilons=[0.01,0.03,0.05]):
    preprocessing = dict(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010], axis=-3)
    fmodel = fb.PyTorchModel(model, bounds=(-3,3), preprocessing=preprocessing)

    attacks = {
        'FGSM': fb.attacks.FGSM(),
        'PGD': fb.attacks.LinfPGD(steps=10, rel_stepsize=0.25, random_start=True),
        'DeepFool': fb.attacks.L2DeepFoolAttack(steps=50),
        'Carlini-Wagner (C&W)': fb.attacks.L2CarliniWagnerAttack(steps=50)
    }

    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    results = {}

    for name, attack in attacks.items():
        results[name] = []

        if name in ["DeepFool", "Carlini-Wagner (C&W)"]:
            print(f"\n>>> Running {name}...")
            adversarials, _, success = attack(fmodel, imgs, labels, epsilons=[None])
            acc = 100 * (1 - success.float().mean().item())
            results[name].append(acc)
            print(f"{name} → Accuracy post-attacco: {acc:.1f}%")
            continue

        for eps in epsilons:
            adversarials, _, success = attack(fmodel, imgs, labels, epsilons=[eps])
            acc = 100*(1 - success.float().mean().item())
            results[name].append(acc)
            print(f"{name} ε={eps:.3f} → Accuracy={acc:.1f}%")

    return results


# =====================================================
# 5. MAIN
# =====================================================
def main():
    print("\n=== TRAINING STANDARD ===")
    model = create_model()
    model = train_model(model, adversarial_training=False)
    acc_clean = evaluate_model(model, testloader)
    print(f"Clean Accuracy: {acc_clean:.2f}%")

    adv_results = run_adversarial_attacks(model, testloader)

    print("\n=== TRAINING ROBUST (PGD Adversarial Training) ===")
    robust_model = create_model()
    robust_model = train_model(robust_model, adversarial_training=True)
    acc_clean_robust = evaluate_model(robust_model, testloader)
    print(f"Clean Accuracy (Robust Model): {acc_clean_robust:.2f}%")

    adv_results_robust = run_adversarial_attacks(robust_model, testloader)

    print("\n=== RESULTS SUMMARY ===")
    print("Standard Model:", adv_results)
    print("Robust Model:", adv_results_robust)


if __name__ == "__main__":
    main()
