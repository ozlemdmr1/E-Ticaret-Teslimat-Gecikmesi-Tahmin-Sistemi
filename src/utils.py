import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    return (correct / len(y_true)) * 100

def get_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[TN, FP],
                   [FN, TP]])
    return cm, TN, FP, FN, TP

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Tahmin: Zamanında (0)', 'Tahmin: Gecikti (1)'],
                yticklabels=['Gerçek: Zamanında (0)', 'Gerçek: Gecikti (1)'])
    plt.title('Confusion Matrix (ShipDelay AI)', fontweight='bold')
    plt.tight_layout()
    plt.show()

def get_risk_level(prob):
    percentage = prob * 100
    if percentage >= 65:
        return "YÜKSEK 🔴", "Öncelikli kargo / Ekspres rota ataması önerilir."
    elif percentage >= 40:
        return "ORTA 🟡", "Depo çıkışı ve kurye ataması takibe alınmalı."
    else:
        return "DÜŞÜK 🟢", "Standart sevkiyat akışı."