# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Fichiers
DATA_CSV = 'pose_dataset.csv'        # généré par pose_collect.py
MODEL_PATH = 'pose_risk_model.pkl'   # modèle entraîné sauvegardé ici

# Colonnes (doivent correspondre à celles écrites par pose_collect.py)
FEATURES = [
    'Trunk','Neck','NeckLateral','HeadRoll','ShoulderTilt',
    'LeftShoulder','RightShoulder',
    'LeftElbow','RightElbow',
    'LeftHip','RightHip',
    'LeftKnee','RightKnee',
    'LeftThigh','RightThigh',
    'is_sitting'
]

LABEL_COL = 'label'  # 0 = Safe, 1 = Medium, 2 = High

def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"CSV introuvable: {DATA_CSV}. Lance d'abord pose_collect.py pour créer des données.")
    # Charger le dataset
    df = pd.read_csv(DATA_CSV)

    # Garder lignes avec un label
    if LABEL_COL not in df.columns:
        raise ValueError(f"Colonne '{LABEL_COL}' absente du CSV.")
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Vérifier features
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
    # Séparer X, y
    X = df[FEATURES]
    y = df[LABEL_COL]
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Pipeline: imputation -> normalisation -> RandomForest
    pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler(with_mean=False)),  # robuste si colonnes très clairsemées
        ('clf', RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1
        ))
    ])
    # Entraîner
    pipeline.fit(X_train, y_train)
    # Évaluer
    y_pred = pipeline.predict(X_test)
    print('Classification report:')
    print(classification_report(y_test, y_pred, digits=3))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Sauvegarder le modèle
    joblib.dump(pipeline, MODEL_PATH)
    print(f'✅ Modèle sauvegardé: {MODEL_PATH}')

    # Optionnel: importance des features (si dispo)
    try:
        clf = pipeline.named_steps['clf']
        importances = clf.feature_importances_
        order = np.argsort(importances)[::-1]
        print('\nTop features:')
        for idx in order:
            print(f'- {FEATURES[idx]}: {importances[idx]:.4f}')
    except Exception:
        pass

if __name__ == '__main__':
    main()