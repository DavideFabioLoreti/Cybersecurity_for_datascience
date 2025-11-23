# Adversarial Robustness Evaluation on Deep Learning Models

Questo repository contiene l'implementazione sperimentale di valutazione della **robustezza adversarial** su modelli Deep Learning, con confronto tra modello standard e modello robusto addestrato con PGD adversarial training.

---

## Contesto

I modelli di deep learning, pur performanti sui dati puliti, sono vulnerabili a **adversarial attacks**: piccole perturbazioni impercettibili che causano misclassificazioni. Questo progetto esplora:

- Quanto un modello standard è vulnerabile.
- Quanto l'adversarial training può migliorare la robustezza.
- L'efficacia di diversi attacchi: FGSM, PGD, DeepFool e Carlini-Wagner (C\&W).

Il dataset utilizzato è **CIFAR-10** (60.000 immagini RGB 32x32 su 10 classi), e il modello base è una **ResNet-18** modificata per CIFAR-10.

---

## Struttura del progetto

