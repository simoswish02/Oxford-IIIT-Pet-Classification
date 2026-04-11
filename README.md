# Oxford-IIIT Pet Classification

Deep learning project for fine-grained classification of 37 dog and cat breeds using the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The project is structured in two parts:
1. **Custom CNN (PetNet)** — a lightweight convolutional neural network designed from scratch.
2. **Transfer Learning (ResNet-18)** — a pretrained ResNet-18 fine-tuned progressively on the target dataset.

---

## Project Structure

- **Part 1 – Custom CNN (PetNet)**  
  A modular CNN built from scratch using stacked `ConvBlock` units (Conv → BatchNorm → ReLU), global average pooling, and a fully connected classification head.  
  Trained with MixUp augmentation, label smoothing, AdamW optimizer, and OneCycleLR scheduler.

- **Part 2 – Transfer Learning (ResNet-18)**  
  A pretrained ResNet-18 adapted for 37-class classification via progressive layer unfreezing and layer-specific learning rates.  
  Systematic ablation study across input resolution, augmentation strength, regularization, and unfreezing depth.

---

## Results

| Configuration | Test Accuracy |
|---|---|
| PetNet (custom CNN, baseline) | ~52% |
| ResNet-18 – Original hyperparameters | 62.57% |
| ResNet-18 – Resized inputs (224×224) | 85.32% |
| ResNet-18 – Lowered augmentations | 88.03% |
| ResNet-18 – No label smoothing & MixUp | 89.17% |
| ResNet-18 – Unfreezing up to `layer1` | **89.71%** |

---

## Key Techniques

- **MixUp augmentation** for improved generalization
- **Label smoothing** via `CrossEntropyLoss`
- **AdamW** optimizer with weight decay
- **OneCycleLR** learning rate scheduler
- **Progressive layer unfreezing** with layer-specific learning rates
- **Ablation studies** on architecture, augmentation, regularization, and fine-tuning depth

---

## Requirements

- Python 3.x
- PyTorch
- torchvision
- torchinfo (for FLOPs/param counting)

---

## Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 categories of dogs and cats, ~200 images per class.

---

## Usage

Open and run `assignment_module_two.ipynb` in Google Colab or a local Jupyter environment with GPU support.
