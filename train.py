import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import ShipDelayDataset
from src.model import ShipDelayNN
from src.utils import calculate_accuracy, get_confusion_matrix, plot_confusion_matrix

# 1. Veri Hazırlığı
data_pipeline = ShipDelayDataset('data/shipping.csv')
X_train, y_train, X_test, y_test, test_raw_df = data_pipeline.load_and_preprocess()

# 2. Model, Loss & Optimizer
input_dim = X_train.shape[1]
model = ShipDelayNN(input_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop
epochs = 100
train_losses = []

print("🚀 Model Eğitimi Başlıyor...\n")
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:03d}/{epochs}] ---> Loss: {loss.item():.4f}")

# 4. Loss Grafiği
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='navy')
plt.title('Epoch vs Loss (Training Curve)')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.grid(True)
plt.legend()
plt.show()

# 5. Test Değerlendirmesi
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

y_true_np = y_test.numpy().flatten()
y_pred_np = preds.numpy().flatten()

acc = calculate_accuracy(y_test, preds)
print(f"\n🎯 Test Seti Accuracy: %{acc:.2f}")

# 6. Confusion Matrix & Heatmap
cm, TN, FP, FN, TP = get_confusion_matrix(y_true_np, y_pred_np)
print(f"TN: {TN} | FP: {FP} | FN: {FN} | TP: {TP}")
plot_confusion_matrix(cm)

# 7. Model Ağırlıklarını Saklama
torch.save(model.state_dict(), 'shipdelay_model.pth')
print("💾 Model 'shipdelay_model.pth' olarak kaydedildi.")