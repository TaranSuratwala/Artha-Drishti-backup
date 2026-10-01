import os
import joblib
import torch
import numpy as np

base_dir = r"C:\Users\Lenovo\Documents\VS CODE codes(files)\helloworld\unified_models"
feature_cols_path = os.path.join(base_dir, "feature_cols.pkl")
scaler_path = os.path.join(base_dir, "feature_scaler.pkl")
medians_path = os.path.join(base_dir, "feature_medians.pkl")
qbins_path = os.path.join(base_dir, "training_quantiles.pkl")
model_path = os.path.join(base_dir, "unified_model.pth")

if not os.path.exists(model_path):
    print("Model not found")
    exit(1)

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
feature_cols = joblib.load(feature_cols_path)
saved_input_dim = checkpoint.get('input_dim', len(feature_cols))

print(f"saved_input_dim={saved_input_dim}, len(feature_cols)={len(feature_cols)}")

if saved_input_dim != len(feature_cols):
    print("Fixing mismatch...")
    # Truncate feature_cols
    feature_cols = feature_cols[:saved_input_dim]
    joblib.dump(feature_cols, feature_cols_path)
    print("Updated feature_cols.pkl")
    
    # Truncate scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if hasattr(scaler, 'center_'):
            scaler.center_ = scaler.center_[:saved_input_dim]
        if hasattr(scaler, 'scale_'):
            scaler.scale_ = scaler.scale_[:saved_input_dim]
        if hasattr(scaler, 'n_features_in_'):
            scaler.n_features_in_ = saved_input_dim
        joblib.dump(scaler, scaler_path)
        print("Updated feature_scaler.pkl")
        
    # Truncate medians
    if os.path.exists(medians_path):
        medians = joblib.load(medians_path)
        medians = medians[:saved_input_dim]
        joblib.dump(medians, medians_path)
        print("Updated feature_medians.pkl")
        
    # Truncate qbins
    if os.path.exists(qbins_path):
        qbins = joblib.load(qbins_path)
        if hasattr(qbins, 'shape') and len(qbins.shape) == 2:
            qbins = qbins[:, :saved_input_dim]
            joblib.dump(qbins, qbins_path)
            print("Updated training_quantiles.pkl")
            
    print("Fix applied successfully!")
else:
    print("No mismatch found. Maybe just updating n_features_in_ in scaler?")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != saved_input_dim:
            print("Fixing n_features_in_...")
            scaler.n_features_in_ = saved_input_dim
            joblib.dump(scaler, scaler_path)
            print("Updated feature_scaler.pkl")
