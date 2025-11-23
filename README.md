## Struttura del progetto

Il repository contiene i seguenti file principali:

- `cybersecurity DL adversarial attack.py` – Script principale che esegue:
  - Addestramento del modello standard su dati puliti.
  - Addestramento del modello robusto con **PGD adversarial training**.
  - Valutazione dei modelli su dati puliti e adversarial examples generati con:
    - FGSM (ε = 0.01, 0.03, 0.05)
    - PGD (10 step, ε = 0.01, 0.03, 0.05)
    - DeepFool (adaptive)
    - Carlini-Wagner (50 step)
  - Stampa delle accuratezze residue per tutti i casi.


---

## Risultati principali

| Attacco / Metriche          | Modello Standard | Modello Robusto |
|----------------------------|----------------|----------------|
| **Clean accuracy**         | 79.53%         | 69.88%         |
| FGSM ε=0.01               | 7.0%           | 13.0%          |
| FGSM ε=0.03               | 7.0%           | 11.0%          |
| FGSM ε=0.05               | 7.0%           | 10.0%          |
| PGD (10 step) ε=0.01      | 7.0%           | 13.0%          |
| PGD (10 step) ε=0.03      | 6.0%           | 11.0%          |
| PGD (10 step) ε=0.05      | 6.0%           | 9.0%           |
| DeepFool (adaptive)        | 0.0%           | 0.0%           |
| C&W (50 step)              | 0.0%           | 0.0%           |

---

## Analisi dei risultati

- Il **modello robusto** mostra maggiore resistenza agli attacchi L\(_\infty\) (FGSM e PGD).  
- Gli **attacchi ottimizzati L\(_2\)** (DeepFool, C&W) rimangono altamente efficaci su entrambi i modelli.  
- È evidente il **trade-off tra accuratezza pulita e robustezza**: il modello robusto sacrifica circa 10 punti percentuali di accuracy sui dati puliti per aumentare la resistenza agli attacchi.

---

## Osservazioni qualitative

- Le perturbazioni sul modello robusto sono **più visibili**, ma riducono le misclassificazioni da attacchi L\(_\infty\).  
- Classi semanticamente simili, come **cat, dog e deer**, risultano più vulnerabili, suggerendo che gli adversarial examples sfruttano correlazioni semantiche interne al modello.

---

## Dipendenze principali

- [PyTorch](https://pytorch.org/)  
- [Torchvision](https://pytorch.org/vision/stable/index.html)  
- [Foolbox](https://foolbox.readthedocs.io/)  
- [TQDM](https://tqdm.github.io/)

---

## Licenza

Rilasciato sotto **MIT License**.

---

## Riferimenti

- Rauber, J., Brendel, W., & Bethge, M. (2017). *Foolbox: A Python toolbox to benchmark the robustness of machine learning models*. arXiv:1707.04131  
- Madry, A., et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR
