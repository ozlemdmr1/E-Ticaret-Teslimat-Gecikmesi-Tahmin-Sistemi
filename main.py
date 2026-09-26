import torch
import pandas as pd
import numpy as np

from src.dataset import ShipDelayDataset
from src.model import ShipDelayNN
from src.utils import get_risk_level

def run_cli():
    # Pipeline yükle
    pipeline = ShipDelayDataset('data/shipping.csv')
    X_train, _, _, _, _ = pipeline.load_and_preprocess()
    
    model = ShipDelayNN(X_train.shape[1])
    try:
        model.load_state_dict(torch.load('shipdelay_model.pth'))
    except:
        print("⚠️ Eğitilmiş model bulunamadı! Lütfen önce 'train.py' dosyasını çalıştırın.")
        return

    model.eval()

    print("\n" + "="*50)
    print("      📦 SHIPDELAY AI — TAHMİN BİRİMİ 📦      ")
    print("="*50)

    cost = float(input("Ürün Fiyatı ($) [ör. 220]: "))
    weight = float(input("Ürün Ağırlığı (gram) [ör. 3500]: "))
    discount = float(input("İndirim Oranı ($) [ör. 15]: "))
    rating = float(input("Müşteri Puanı (1-5) [ör. 4]: "))
    prior = float(input("Geçmiş Satın Alma Sayısı [ör. 3]: "))
    calls = float(input("Müşteri Hizmetleri Çağrı Sayısı [ör. 2]: "))

    print("\nGönderim Yöntemi (1: Flight, 2: Ship, 3: Road): ")
    m_choice = input("Seçim: ").strip()
    mode_map = {'1': 'Flight', '2': 'Ship', '3': 'Road'}
    mode = mode_map.get(m_choice, 'Ship')

    print("\nÜrün Önem Seviyesi (1: low, 2: medium, 3: high): ")
    i_choice = input("Seçim: ").strip()
    imp_map = {'1': 'low', '2': 'medium', '3': 'high'}
    importance = imp_map.get(i_choice, 'low')

    warehouse = input("\nDepo Bloğu (A, B, C, D, F): ").strip().upper()
    if warehouse not in ['A', 'B', 'C', 'D', 'F']:
        warehouse = 'F'

    # DataFrame Oluştur
    raw_dict = {
        'Customer_care_calls': [calls],
        'Customer_rating': [rating],
        'Cost_of_the_Product': [cost],
        'Prior_purchases': [prior],
        'Discount_offered': [discount],
        'Weight_in_gms': [weight],
        'Product_importance': [importance],
        'Warehouse_block': [warehouse],
        'Mode_of_Shipment': [mode],
        'Gender': ['F']
    }
    
    df_raw = pd.DataFrame(raw_dict)
    
    # Categorical Map
    df_raw['Product_importance'] = df_raw['Product_importance'].map({'low': 0, 'medium': 1, 'high': 2})
    df_encoded = pd.get_dummies(df_raw, columns=['Warehouse_block', 'Mode_of_Shipment', 'Gender'], drop_first=True)
    
    # Sütun Hizalama
    full_df = pd.DataFrame(0, index=[0], columns=pipeline.feature_columns)
    for col in df_encoded.columns:
        if col in full_df.columns:
            full_df[col] = df_encoded[col]

    # Scaling & Tensor
    scaled = pipeline.scaler.transform(full_df.values)
    tensor_in = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor_in)).item()

    risk_label, tip = get_risk_level(prob)

    print("\n" + "="*50)
    print("                 SHIPDELAY AI                   ")
    print("="*50)
    print(f" Gecikme İhtimali : %{prob*100:.1f}")
    print(f" Risk Seviyesi    : {risk_label}")
    print("-" * 50)
    print(f" Operasyon Notu   : {tip}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_cli()