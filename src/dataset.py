import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class ShipDelayDataset:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_and_preprocess(self):
        # 1. Veriyi Oku
        df = pd.read_csv(self.csv_path)
        
        # Sütun isimlerindeki boşlukları temizle
        df.columns = df.columns.str.strip()
        
        # ID kolonunu çıkar (varsa)
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
            
        # Target kolonunu tespit et
        possible_target_cols = ['Reached.on.Time_Y.N', 'Reached_on_Time_Y_N', 'Reached.on.Time_Y_N', 'Reached_on_Time_Y.N']
        target_col = None
        for col in possible_target_cols:
            if col in df.columns:
                target_col = col
                break
                
        if target_col is None:
            target_col = df.columns[-1]
            
        print(f"🎯 Hedef (Target) Sütun: {target_col}")
        
        # Target ve Features ayır
        y = df[target_col].values.astype(np.float32)
        X_df = df.drop(columns=[target_col]).copy()
        
        # Categorical Encoding
        importance_map = {'low': 0, 'medium': 1, 'high': 2}
        if 'Product_importance' in X_df.columns:
            X_df['Product_importance'] = X_df['Product_importance'].map(importance_map).fillna(0)
            
        # One-Hot Encoding
        cat_cols = [c for c in ['Warehouse_block', 'Mode_of_Shipment', 'Gender'] if c in X_df.columns]
        if cat_cols:
            X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
            
        # Tüm sayısal olmayan sütunları / bool değerleri sayıya dönüştür
        X_df = X_df.astype(np.float32)
        
        self.feature_columns = X_df.columns.tolist()
        X = X_df.values
        
        print(f"📊 Okunan Toplam Satır Sayısı: {len(X)} | Sütun Sayısı: {X.shape[1]}")
        
        # Train-Test Split (%80 Train, %20 Test)
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        train_size = int(len(X) * 0.8)
        
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y[train_idx], y[test_idx]
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)
        
        # PyTorch Tensors
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
        
        X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(1)
        
        return X_train, y_train, X_test, y_test, X_df.iloc[test_idx]