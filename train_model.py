import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_and_preprocess_data():
    print("Loading datasets...")
    collision_path = "dft-road-casualty-statistics-collision-provisional-2025 (1).csv"
    casualty_path = "dft-road-casualty-statistics-casualty-provisional-2025.csv"
    vehicle_path = "dft-road-casualty-statistics-vehicle-provisional-2025.csv"
    
    if not all(os.path.exists(p) for p in [collision_path, casualty_path, vehicle_path]):
        raise FileNotFoundError("Missing one or more required CSV files in workspace.")
        
    df = pd.read_csv(collision_path, low_memory=False)
    cas_df = pd.read_csv(casualty_path, low_memory=False)
    veh_df = pd.read_csv(vehicle_path, low_memory=False)
    print(f"Original collisions shape: {df.shape}")
    
    # 1. Date/Time Parsing
    print("Parsing date and time...")
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['hour'] = df['datetime'].dt.hour
    
    # 2. Map Categorical Codes to Labels
    print("Decoding categorical codes...")
    severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
    day_map = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
    urban_rural_map = {1: "Urban", 2: "Rural", 3: "Unallocated"}
    light_map = {
        1: "Daylight",
        4: "Darkness - lights lit",
        5: "Darkness - lights unlit",
        6: "Darkness - no lights",
        7: "Darkness - lights detail unknown"
    }
    weather_map = {
        1: "Fine no high winds",
        2: "Raining no high winds",
        3: "Snowing no high winds",
        4: "Fine + high winds",
        5: "Raining + high winds",
        6: "Snowing + high winds",
        7: "Fog or mist",
        8: "Other",
        9: "Unknown"
    }
    surface_map = {
        1: "Dry",
        2: "Wet or damp",
        3: "Snow",
        4: "Frost or ice",
        5: "Flood over 3cm deep",
        6: "Oil or wet mud",
        7: "Road sign"
    }
    
    # Clean missing codes (-1) to NaN
    df['urban_or_rural_area'] = df['urban_or_rural_area'].replace(-1, np.nan)
    df['light_conditions'] = df['light_conditions'].replace(-1, np.nan)
    df['weather_conditions'] = df['weather_conditions'].replace(-1, np.nan)
    df['road_surface_conditions'] = df['road_surface_conditions'].replace(-1, np.nan)
    df['speed_limit'] = df['speed_limit'].replace(-1, np.nan)
    
    # Decode
    df['collision_severity_label'] = df['collision_severity'].map(severity_map)
    df['day_of_week_label'] = df['day_of_week'].map(day_map)
    df['urban_or_rural_label'] = df['urban_or_rural_area'].map(urban_rural_map)
    df['light_conditions_label'] = df['light_conditions'].map(light_map)
    df['weather_conditions_label'] = df['weather_conditions'].map(weather_map)
    df['road_surface_conditions_label'] = df['road_surface_conditions'].map(surface_map)
    
    # Drop rows where target severity is null
    df = df.dropna(subset=['collision_severity_label'])
    df['is_severe'] = df['collision_severity'].isin([1, 2]).astype(int)
    
    # 3. Feature Engineering from Vehicle Table (Vulnerable Road Users & Driver Demographics)
    print("Engineering features from vehicle table...")
    # Motorcycle codes: 2, 3, 4, 5, 97
    # Pedal cycle: 1
    # HGV/Bus: 11, 20, 21
    veh_df['is_motorcycle'] = veh_df['vehicle_type'].isin([2, 3, 4, 5, 97]).astype(int)
    veh_df['is_pedal_cycle'] = (veh_df['vehicle_type'] == 1).astype(int)
    veh_df['is_hgv_or_bus'] = veh_df['vehicle_type'].isin([11, 20, 21]).astype(int)
    veh_df['age_of_driver'] = veh_df['age_of_driver'].replace(-1, np.nan)
    
    veh_agg = veh_df.groupby('collision_index').agg({
        'is_motorcycle': 'max',
        'is_pedal_cycle': 'max',
        'is_hgv_or_bus': 'max',
        'age_of_driver': ['min', 'max']
    })
    veh_agg.columns = ['has_motorcycle', 'has_pedal_cycle', 'has_hgv_or_bus', 'driver_age_min', 'driver_age_max']
    veh_agg = veh_agg.reset_index()
    
    # 4. Feature Engineering from Casualty Table (Vulnerable Pedestrians & Casualty Demographics)
    print("Engineering features from casualty table...")
    # Pedestrian casualty class = 3
    cas_df['is_pedestrian'] = (cas_df['casualty_class'] == 3).astype(int)
    cas_df['age_of_casualty'] = cas_df['age_of_casualty'].replace(-1, np.nan)
    
    cas_agg = cas_df.groupby('collision_index').agg({
        'is_pedestrian': 'max',
        'age_of_casualty': ['min', 'max']
    })
    cas_agg.columns = ['has_pedestrian', 'casualty_age_min', 'casualty_age_max']
    cas_agg = cas_agg.reset_index()
    
    # 5. Merge Features
    print("Merging tables...")
    model_df = df.merge(veh_agg, on='collision_index', how='left')
    model_df = model_df.merge(cas_agg, on='collision_index', how='left')
    
    # Fill missing binary flags with 0
    for col in ['has_motorcycle', 'has_pedal_cycle', 'has_hgv_or_bus', 'has_pedestrian']:
        model_df[col] = model_df[col].fillna(0).astype(int)
        
    print(f"Processed shape: {model_df.shape}")
    print(f"Class distribution of 'is_severe':\n{model_df['is_severe'].value_counts(normalize=True)}")
    
    return model_df

def train_and_evaluate():
    df = load_and_preprocess_data()
    
    # Features & Targets
    NUM_FEATURES = [
        'speed_limit', 'hour', 'number_of_vehicles', 
        'driver_age_min', 'driver_age_max', 
        'casualty_age_min', 'casualty_age_max'
    ]
    CAT_FEATURES = [
        'urban_or_rural_label',
        'light_conditions_label',
        'weather_conditions_label',
        'road_surface_conditions_label',
        'day_of_week_label'
    ]
    BIN_FEATURES = ['has_motorcycle', 'has_pedal_cycle', 'has_hgv_or_bus', 'has_pedestrian']
    
    X = df[NUM_FEATURES + CAT_FEATURES + BIN_FEATURES]
    y = df['is_severe']
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Define Column Transformer for Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_FEATURES),
            ('cat', categorical_transformer, CAT_FEATURES)
        ],
        remainder='passthrough' # Leave binary features as-is
    )
    
    # Define models
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.3f}")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    }
    
    best_f1 = 0
    best_model_name = None
    best_pipeline = None
    results = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        mean_cv_f1 = np.mean(scores)
        print(f"{name} 5-Fold CV F1 Score: {mean_cv_f1:.4f}")
        
        # Fit on full training set
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"{name} Test F1 Score: {f1:.4f}")
        print(f"{name} Test ROC AUC: {roc_auc:.4f}")
        
        results[name] = {
            'cv_f1_mean': mean_cv_f1,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist()
        }
        
        # Keep best model based on CV F1 score
        if mean_cv_f1 > best_f1:
            best_f1 = mean_cv_f1
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest Model selected: {best_model_name}")
    
    # Re-evaluate best model
    best_y_pred = np.array(results[best_model_name]['y_pred'])
    best_y_prob = np.array(results[best_model_name]['y_prob'])
    
    print("\n--- Best Model Classification Report ---")
    report = classification_report(y_test, best_y_pred, target_names=['Slight', 'Severe'])
    print(report)
    
    report_dict = classification_report(y_test, best_y_pred, target_names=['Slight', 'Severe'], output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Feature Importances extraction
    preprocessor_fitted = best_pipeline.named_steps['preprocessor']
    classifier_fitted = best_pipeline.named_steps['classifier']
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = cat_encoder.get_feature_names_out(CAT_FEATURES).tolist()
    all_feature_names = NUM_FEATURES + encoded_cat_features + BIN_FEATURES
    
    # Extract importances
    if hasattr(classifier_fitted, 'feature_importances_'):
        importances = classifier_fitted.feature_importances_.tolist()
    else:
        importances = []
    
    feature_importance_list = sorted(
        [{"feature": name, "importance": imp} for name, imp in zip(all_feature_names, importances)],
        key=lambda x: x['importance'],
        reverse=True
    )
    
    # Generate curves data for plotting
    fpr, tpr, _ = roc_curve(y_test, best_y_prob)
    precision, recall, _ = precision_recall_curve(y_test, best_y_prob)
    
    # Save the pipeline
    joblib.dump(best_pipeline, 'road_safety_model.joblib')
    print("Saved best model pipeline to 'road_safety_model.joblib'")
    
    # Save metadata for dashboard
    metadata = {
        'model_name': best_model_name,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'feature_importances': feature_importance_list,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc_score(y_test, best_y_prob))
        },
        'pr_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'auc': float(auc(recall, precision))
        },
        'cv_results': {
            'RandomForest': {
                'cv_f1': results['RandomForest']['cv_f1_mean'],
                'test_f1': results['RandomForest']['test_f1'],
                'test_roc_auc': results['RandomForest']['test_roc_auc']
            },
            'XGBoost': {
                'cv_f1': results['XGBoost']['cv_f1_mean'],
                'test_f1': results['XGBoost']['test_f1'],
                'test_roc_auc': results['XGBoost']['test_roc_auc']
            }
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Saved model metadata to 'model_metadata.json'")
    
    # Create static diagnostic plots and save them
    plt.figure(figsize=(12, 5))
    
    # ROC Curve Plot
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metadata["roc_curve"]["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Confusion Matrix Plot
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Slight', 'Severe'], 
                yticklabels=['Slight', 'Severe'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_plots.png', dpi=150)
    plt.close()
    print("Saved diagnostic plots to 'model_evaluation_plots.png'")

if __name__ == '__main__':
    train_and_evaluate()
