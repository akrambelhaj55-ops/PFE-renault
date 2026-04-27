"""
================================================================
PFE - Renault Group Tanger Melloussa
Système intelligent de gestion de l'eau
================================================================
Étape 1 : Pipeline ETL - Extraction, Nettoyage, Transformation
Auteur   : [Ton nom]
Date     : Avril 2026
Fichier  : etl_eau_renault.py
================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

FICHIER_EXCEL    = "Synthèse_Eaux_2026_VF.xlsm"
FEUILLE          = "database_Eaux"
FICHIER_SORTIE   = "dataset_eau_propre.csv"
SEUIL_OBJECTIF   = 1.25   # m³/véhicule (objectif Renault 2026)
TCM_MIN_PROD     = 100    # véhicules minimum pour considérer un jour de production


# ─────────────────────────────────────────────
# ÉTAPE 1 : EXTRACTION
# ─────────────────────────────────────────────

def extraire_donnees(fichier, feuille):
    """
    Lit le fichier Excel et retourne uniquement les lignes
    déjà remplies par le responsable (TCM non-null).
    Lecture incrémentale : si un CSV existant est présent,
    on ne charge que les nouvelles lignes.
    """
    print(f"[ETL] Lecture du fichier : {fichier}")
    df = pd.read_excel(fichier, sheet_name=feuille, header=0)
    df.columns = df.columns.str.strip()

    # Garder uniquement les lignes avec données réelles
    df = df[df['TCM'].notna()].copy()
    print(f"[ETL] {len(df)} lignes trouvées (de {df['Date'].min().date()} à {df['Date'].max().date()})")

    # Si un CSV existe déjà, on ne prend que les nouvelles lignes
    if os.path.exists(FICHIER_SORTIE):
        df_existant = pd.read_csv(FICHIER_SORTIE, parse_dates=['Date'])
        derniere_date = df_existant['Date'].max()
        df = df[df['Date'] > derniere_date]
        print(f"[ETL] Mode incrémental : {len(df)} nouvelle(s) ligne(s) depuis le {derniere_date.date()}")

    return df


# ─────────────────────────────────────────────
# ÉTAPE 2 : NETTOYAGE
# ─────────────────────────────────────────────

def nettoyer_donnees(df):
    """
    Applique les règles de nettoyage :
    - Valeurs aberrantes (KPI gonflé par faible TCM)
    - Ratios impossibles
    - Classification des types de journées
    """
    print("[ETL] Nettoyage des données...")

    # Classification des journées
    df['is_arret']    = (df['TCM'] < TCM_MIN_PROD).astype(int)
    df['is_weekend']  = (df['Date'].dt.dayofweek >= 5).astype(int)
    df['is_lundi']    = (df['Date'].dt.dayofweek == 0).astype(int)

    # Remplacer les 0 par NaN pour les colonnes de consommation
    # (un 0 = donnée manquante, pas une vraie consommation nulle)
    cols_conso = ['ED_Total', 'EOR_Total', 'EI_Looker', 'EP_Looker']
    for col in cols_conso:
        if col in df.columns:
            nb_zeros = (df[col] == 0).sum()
            if nb_zeros > 0:
                df[col] = df[col].replace(0, np.nan)
                print(f"  → {col} : {nb_zeros} zéros remplacés par NaN")

    nb_avant = len(df)
    return df


# ─────────────────────────────────────────────
# ÉTAPE 3 : TRANSFORMATION (Feature Engineering)
# ─────────────────────────────────────────────

def transformer_donnees(df):
    """
    Crée les nouvelles variables (features) nécessaires pour :
    - Le Machine Learning
    - Les KPIs Looker Studio
    - La détection d'anomalies
    """
    print("[ETL] Création des features...")

    # --- Features temporelles ---
    df['jour_semaine']  = df['Date'].dt.dayofweek   # 0=Lundi, 6=Dimanche
    df['mois']          = df['Date'].dt.month
    df['trimestre']     = df['Date'].dt.quarter
    df['nom_jour']      = df['Date'].dt.day_name()
    df['semaine_annee'] = df['Date'].dt.isocalendar().week.astype(int)

    # --- KPI principal : m³ / véhicule ---
    # Uniquement calculé pour les vrais jours de production
    df['KPI_m3_veh'] = np.where(
        df['TCM'] >= TCM_MIN_PROD,
        df['E. Appro Looker'] / df['TCM'],
        np.nan
    )

    # --- Ratios de performance ---
    df['ratio_EOR_ED'] = df['EOR_Total'] / df['ED_Total'].replace(0, np.nan)
    df['taux_EP']      = df['EP_Looker'] / df['E. Appro Looker'].replace(0, np.nan)
    df['taux_EI']      = df['EI_Looker'] / df['E. Appro Looker'].replace(0, np.nan)

    # --- Détection d'anomalies simple (règles métier) ---
    # Anomalie = KPI dépasse l'objectif de plus de 15%
    df['anomalie_kpi'] = np.where(
        (df['TCM'] >= TCM_MIN_PROD) & (df['KPI_m3_veh'] > SEUIL_OBJECTIF * 1.15),
        1, 0
    )

    # Anomalie = ratio recyclage EOR/ED trop faible (< 10%)
    df['anomalie_recyclage'] = np.where(
        df['ratio_EOR_ED'] < 0.10,
        1, 0
    )

    # --- Statistiques mobiles (contexte pour le ML) ---
    df = df.sort_values('Date').reset_index(drop=True)
    df['KPI_MA7']  = df['KPI_m3_veh'].rolling(window=7,  min_periods=3).mean()
    df['KPI_MA30'] = df['KPI_m3_veh'].rolling(window=30, min_periods=7).mean()
    df['conso_MA7'] = df['E. Appro Looker'].rolling(window=7, min_periods=3).mean()

    print(f"  → {len(df.columns)} colonnes dans le dataset final")
    print(f"  → Jours normaux (TCM≥{TCM_MIN_PROD}) : {(df['TCM']>=TCM_MIN_PROD).sum()}")
    print(f"  → Jours arrêt                        : {(df['is_arret']==1).sum()}")
    print(f"  → Anomalies KPI détectées            : {df['anomalie_kpi'].sum()}")
    print(f"  → Anomalies recyclage détectées      : {df['anomalie_recyclage'].sum()}")

    return df


# ─────────────────────────────────────────────
# ÉTAPE 4 : CHARGEMENT
# ─────────────────────────────────────────────

def charger_donnees(df):
    """
    Sauvegarde le dataset propre.
    En mode incrémental : ajoute au fichier existant.
    En production : INSERT dans PostgreSQL.
    """
    print("[ETL] Chargement du dataset...")

    if df.empty:
        print("[ETL] Aucune nouvelle donnée à charger.")
        return

    if os.path.exists(FICHIER_SORTIE):
        # Mode incrémental : ajouter sans écraser
        df.to_csv(FICHIER_SORTIE, mode='a', header=False, index=False)
        print(f"[ETL] {len(df)} ligne(s) ajoutée(s) à {FICHIER_SORTIE}")
    else:
        # Premier lancement : créer le fichier
        df.to_csv(FICHIER_SORTIE, index=False)
        print(f"[ETL] Fichier créé : {FICHIER_SORTIE} ({len(df)} lignes)")

    # ── En production, remplacer par : ──────────────────────────────
    # import psycopg2
    # conn = psycopg2.connect("dbname=eau_renault user=admin password=xxx")
    # df.to_sql("fact_eau_journalier", conn, if_exists="append", index=False)
    # ────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────

def run_pipeline():
    print("=" * 55)
    print(f" Pipeline ETL Eau - Renault Tanger")
    print(f" Exécution : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    df = extraire_donnees(FICHIER_EXCEL, FEUILLE)

    if df.empty:
        print("[ETL] Aucune nouvelle donnée disponible. Fin du pipeline.")
        return

    df = nettoyer_donnees(df)
    df = transformer_donnees(df)
    charger_donnees(df)

    print("=" * 55)
    print(" Pipeline terminé avec succès ✓")
    print("=" * 55)
    return df


if __name__ == "__main__":
    df_final = run_pipeline()
