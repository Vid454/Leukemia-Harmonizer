# @title
# ════════════════════════════════════════════════════════════
# LEUKEMIA HARMONISER — CELL 2: FULL SYSTEM
# ════════════════════════════════════════════════════════════
# SETUP:
#   1. In Colab: left sidebar > Secrets (key icon) > add NGROK_TOKEN and
#      enable "Notebook access"
#   2. Run this cell
#   3. Upload your pre-trained pkl models when prompted
#   4. Open the public URL — everything else happens in the UI
# ════════════════════════════════════════════════════════════

from google.colab import userdata
NGROK_TOKEN = userdata.get("NGROK_TOKEN")

# ── IMPORTS ───────────────────────────────────────────────────
import nest_asyncio
nest_asyncio.apply()

import asyncio, io, json, os, re, threading, time, traceback, uuid
import numpy as np
import pandas as pd
import joblib
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List

from rapidfuzz import process, fuzz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from google.colab import files
from pyngrok import ngrok
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── GLOBAL STATE ──────────────────────────────────────────────
MODELS          = {}   # model_key → bundle dict
RAW_DATASETS    = {}   # name → raw DataFrame
PROCESSED       = {}   # name → harmonized DataFrame
HARMONIZE_STATS = {}   # name → stats dict
PREDICTIONS     = {}   # name → {model_name: results, 'rows': [...]}
TRAINING_JOBS   = {}   # job_id → {'status','log','metrics'}

# ── HARMONIZER CONSTANTS ──────────────────────────────────────
TARGET_FEATURES = [
    'WBC','RBC','Hemoglobin','Hematocrit','MCV','MCH','MCHC',
    'RDW','Platelets','MPV','Neutrophils','Lymphocytes',
    'Monocytes','Eosinophils','Basophils','Age','Gender_enc'
]

VALID_RANGES = {
    'WBC':(100,500000),'RBC':(0.5,10.0),'Hemoglobin':(2.0,25.0),
    'Hematocrit':(5.0,70.0),'MCV':(50.0,150.0),'MCH':(10.0,50.0),
    'MCHC':(20.0,45.0),'RDW':(5.0,40.0),'Platelets':(1000,1500000),
    'MPV':(2.0,25.0),'Neutrophils':(0,400000),'Lymphocytes':(0,400000),
    'Monocytes':(0,50000),'Eosinophils':(0,50000),'Basophils':(0,10000),
    'Age':(0,120),
}

DEFAULTS = {
    'WBC':7500,'RBC':4.8,'Hemoglobin':13.5,'Hematocrit':41.0,
    'MCV':90.0,'MCH':29.5,'MCHC':33.5,'RDW':13.0,'Platelets':250000,
    'MPV':10.0,'Neutrophils':4500,'Lymphocytes':2500,'Monocytes':500,
    'Eosinophils':200,'Basophils':50,'Age':45
}

ALIAS_MAP = {
    'WBC':['wbc','wbc_count_ul','white_blood_cell_count','white blood cell count',
           'white_blood_cells','wbc_10e9_l','tlc','leukocytes','total wbc'],
    'RBC':['rbc','rbc_count_mill_ul','red_blood_cell_count','red blood cell count',
           'red_blood_cells','rbc_10e12_l','erythrocytes'],
    'Hemoglobin':['hemoglobin','hemoglobin_gdl','hemoglobin_g_per_dl','hgb',
                  'hgb_g_dl','haemoglobin','hb'],
    'Hematocrit':['hematocrit','hematocrit_pct','hematocrit_percent','hct',
                  'hct_percent','pcv'],
    'MCV':['mcv','mcv_fl','mcv_femtol','mean_corpuscular_vol',
           'mean corpuscular volume','mean_corpuscular_volume'],
    'MCH':['mch','mch_pg','mch_picog','mean_corp_hemoglobin',
           'mean corpuscular hemoglobin'],
    'MCHC':['mchc','mchc_gdl','mchc_g_dl'],
    'RDW':['rdw','rdw_pct','rdw_cv_pct','red_cell_dist_width',
           'red cell distribution width','rdwcv'],
    'Platelets':['platelets','platelet_count','plt','plt_10e9_l',
                 'platelets_per_ul','thrombocytes'],
    'MPV':['mpv','mpv_fl','mean_platelet_volume','mean platelet volume'],
    'Neutrophils':['neutrophils','absolute_neutrophil_count','neut_abs_10e9_l',
                   'neutrophils_per_ul','anc','neut'],
    'Lymphocytes':['lymphocytes','absolute_lymphocyte_count','lymph_abs_10e9_l',
                   'lymphocytes_per_ul','alc','lymph'],
    'Monocytes':['monocytes','absolute_monocyte_count','mono_abs_10e9_l',
                 'monocytes_per_ul','amc','mono'],
    'Eosinophils':['eosinophils','absolute_eosinophil_count','eos_abs_10e9_l',
                   'eosinophils_per_ul','aec','eos'],
    'Basophils':['basophils','absolute_basophil_count','baso_abs_10e9_l',
                 'basophils_per_ul','baso'],
    'Age':['age','age_at_enrollment','patient_age','age_years'],
    'Gender':['gender','sex','biological_sex','patient_sex'],
}

DROP_PATTERNS = [
    'mrn','admission','discharge','state','city','zip','race',
    'ethnicity','insurance','ward','physician','height_cm','weight_kg',
    'systolic','diastolic','sample_id','lab_note','equipment_serial',
    'analyzer_version','icd10','drg_code','specimen_id',
    'collection_date','processing_date','turnaround','ordering',
    'analyzer_serial','sample_quality','sample_volume','technician',
    'neut_pct','lymph_pct','mono_pct','eos_pct','baso_pct',
    'repeat_test','priority','instrument_flag','qc_passed','accession',
    'study_id','protocol_id','study_site','enrollment_date',
    'follow_up','education_level','employment','annual_income',
    'bmi','smoking','alcohol_use','exercise_freq','family_cancer',
    'blood_type','prior_chemo','radiation','comorbid','ecog',
    'trial_arm','ferritin','crp_mg','esr_mm','il6_pg','consent',
    'data_quality_flag','notes',
]

DIAG_KEYWORDS = [
    'diagnosis','label','target','outcome','leukemia_status',
    'cancer_type','result','classification','disease','class',
]

POSITIVE_KEYWORDS = [
    'leukemia','positive','1','cancer','malignant',
    'aml','all','cml','cll','yes','true',
]

# ── HARMONIZER FUNCTIONS ──────────────────────────────────────
def _should_drop(col_lower: str) -> bool:
    return any(pat in col_lower for pat in DROP_PATTERNS)

def _fix_scales(df: pd.DataFrame):
    conversions = []
    checks = {
        'WBC':(500,1000),'Platelets':(1000,1000),
        'Neutrophils':(50,1000),'Lymphocytes':(50,1000),
        'Monocytes':(20,1000),'Eosinophils':(20,1000),'Basophils':(5,1000),
    }
    for col,(thr,mult) in checks.items():
        if col in df.columns:
            med = df[col].dropna().median()
            if pd.notna(med) and 0 < med < thr:
                df[col] = df[col] * mult
                conversions.append(f'{col} ×{mult} (10⁹/L → cells/µL)')
    return df, conversions

def _validate_clean(df: pd.DataFrame):
    log = {}
    for col,(lo,hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        n = int(bad.sum())
        if n > 0:
            good_vals = df.loc[~bad & df[col].notna(), col]
            med = good_vals.median() if len(good_vals) > 0 else DEFAULTS.get(col, 0)
            df.loc[bad, col] = med
            log[col] = n
    return df, log

def harmonize_df(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    original_cols = list(df.columns)
    cols_lower    = {c.lower().strip(): c for c in df.columns}

    # ── detect diagnosis column before dropping ───────────────
    diag_col = None
    for col in original_cols:
        cl = col.lower().strip()
        if any(kw in cl for kw in DIAG_KEYWORDS):
            diag_col = col
            break

    # ── drop non-CBC columns ──────────────────────────────────
    to_drop = list({orig for low, orig in cols_lower.items()
                    if _should_drop(low)})
    df = df.drop(columns=to_drop, errors='ignore')
    n_dropped = len(to_drop)

    # ── fuzzy column mapping ──────────────────────────────────
    avail   = list(df.columns)
    a_lower = [c.lower().strip() for c in avail]
    mapping = {}
    mapping_detail = []
    used_targets = set()

    for target, aliases in ALIAS_MAP.items():
        if target in used_targets:
            continue
        for i, cl in enumerate(a_lower):
            if cl in aliases or cl == target.lower():
                if avail[i] not in mapping:
                    mapping[avail[i]] = target
                    used_targets.add(target)
                    mapping_detail.append({
                        'original': avail[i], 'mapped': target,
                        'match': 'Exact', 'confidence': 100
                    })
                break
        else:
            result = process.extractOne(
                target.lower(), a_lower, scorer=fuzz.token_sort_ratio
            )
            if result and result[1] >= 72:
                idx = a_lower.index(result[0])
                if avail[idx] not in mapping:
                    mapping[avail[idx]] = target
                    used_targets.add(target)
                    mapping_detail.append({
                        'original': avail[idx], 'mapped': target,
                        'match': 'Fuzzy', 'confidence': int(result[1])
                    })

    # mark unmapped originals as Dropped
    mapped_orig = {m['original'] for m in mapping_detail}
    for col in original_cols:
        if col not in mapped_orig:
            mapping_detail.append({
                'original': col, 'mapped': '—',
                'match': 'Dropped', 'confidence': 0
            })

    df = df.rename(columns=mapping)
    matched_features = [t for t in ALIAS_MAP if t in df.columns]
    df = df[[c for c in df.columns if c in ALIAS_MAP]].copy()

    # ── to numeric ────────────────────────────────────────────
    for col in df.columns:
        if col != 'Gender':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── scale fix ─────────────────────────────────────────────
    df, conversions = _fix_scales(df)

    # ── outlier cleaning ──────────────────────────────────────
    df, clean_log = _validate_clean(df)

    # ── gender ────────────────────────────────────────────────
    gender_map = {
        'm':'Male','f':'Female','male':'Male','female':'Female',
        '1':'Male','2':'Female','man':'Male','woman':'Female',
        'male ':'Male','female ':'Female',
    }
    if 'Gender' in df.columns:
        df['Gender'] = (df['Gender'].astype(str).str.strip().str.lower()
                        .map(gender_map).fillna('Male'))
    else:
        df['Gender'] = 'Male'

    # ── fill nulls ────────────────────────────────────────────
    n_nulls = int(df.isnull().sum().sum())
    for feat, default in DEFAULTS.items():
        if feat not in df.columns:
            df[feat] = default
        else:
            df[feat] = df[feat].fillna(default)

    # ── confidence ────────────────────────────────────────────
    cbc_feats  = [f for f in DEFAULTS if f != 'Age']
    real_feats = [f for f in cbc_feats if f in matched_features]
    confidence = round(len(real_feats) / len(cbc_feats) * 100, 1)
    n_dupes    = int(df.duplicated().sum())

    stats = {
        'matched': len(matched_features),
        'total': 17,
        'confidence': confidence,
        'dropped': n_dropped,
        'outliers_cleaned': sum(clean_log.values()),
        'nulls_filled': n_nulls,
        'scale_conversions': conversions,
        'duplicates': n_dupes,
        'clean_log': {k: int(v) for k, v in clean_log.items()},
        'mapping_detail': mapping_detail,
        'rows': len(df),
        'has_diagnosis': diag_col is not None,
        'diagnosis_col': diag_col,
    }
    return df, stats

# ── PREDICTION ENGINE ─────────────────────────────────────────
def run_predictions(df_harm: pd.DataFrame, models_override=None):
    target_models = models_override if models_override else MODELS
    results = {}
    for model_key, bundle in target_models.items():
        try:
            model     = bundle['model']
            features  = bundle['features']
            threshold = bundle.get('threshold', 0.2)
            le        = bundle['le_gender']
            scaler    = bundle.get('scaler')

            df = df_harm.copy()
            try:
                df['Gender_enc'] = le.transform(df['Gender'])
            except Exception:
                df['Gender_enc'] = 0

            # ensure all features exist
            for f in features:
                if f not in df.columns:
                    df[f] = 0

            X = df[features].fillna(0).values
            if scaler is not None:
                X = scaler.transform(X)

            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)

            results[model_key] = {
                'predictions':    preds.tolist(),
                'probabilities':  [round(float(p)*100, 2) for p in probs],
                'positive':       int(preds.sum()),
                'negative':       int(len(preds) - preds.sum()),
                'high_risk':      int((probs > 0.70).sum()),
                'inconclusive':   int(((probs >= 0.30) & (probs <= 0.70)).sum()),
                'low_risk':       int((probs < 0.30).sum()),
                'positive_rate':  round(float(preds.mean())*100, 2),
                'display_name':   bundle.get('display_name', model_key),
            }
        except Exception as e:
            results[model_key] = {'error': str(e), 'display_name': model_key}
    return results

def build_rows(df_harm: pd.DataFrame, preds: dict) -> list:
    rows = []
    model_keys = list(preds.keys())
    for i in range(len(df_harm)):
        row = {'row_index': i}
        for col in ['WBC','RBC','Hemoglobin','Hematocrit',
                    'Platelets','Neutrophils','Lymphocytes']:
            if col in df_harm.columns:
                v = df_harm[col].iloc[i]
                row[col] = round(float(v), 2) if pd.notna(v) else None
        row['Age']    = (int(df_harm['Age'].iloc[i])
                         if 'Age' in df_harm.columns else None)
        row['Gender'] = (df_harm['Gender'].iloc[i]
                         if 'Gender' in df_harm.columns else 'Unknown')
        votes = []
        max_p = 0.0
        for mk in model_keys:
            res = preds[mk]
            if 'error' in res:
                continue
            pred_val = res['predictions'][i]
            prob_val = res['probabilities'][i]
            row[f'{mk}_prediction'] = 'Positive' if pred_val == 1 else 'Negative'
            row[f'{mk}_probability'] = prob_val
            max_p = max(max_p, prob_val)
            votes.append(pred_val)
        if votes:
            s = sum(votes)
            if s == len(votes):
                row['consensus'] = 'Positive';  row['risk_level'] = 'High Risk'
            elif s == 0:
                row['consensus'] = 'Negative';  row['risk_level'] = 'Low Risk'
            else:
                row['consensus'] = 'Inconclusive'; row['risk_level'] = 'Inconclusive'
        else:
            row['consensus'] = 'Unknown'; row['risk_level'] = 'Unknown'
        row['max_probability'] = round(max_p, 2)
        rows.append(row)
    return rows

def _store_dataset(name, df_raw, df_harm, stats, preds):
    RAW_DATASETS[name]    = df_raw
    PROCESSED[name]       = df_harm
    HARMONIZE_STATS[name] = stats
    preds_store = {k: {kk: vv for kk, vv in v.items()
                       if kk not in ('predictions', 'probabilities')}
                   for k, v in preds.items()}
    preds_store['_rows'] = build_rows(df_harm, preds)
    PREDICTIONS[name]    = preds_store

def _refresh_predictions_all():
    """Re-run predictions on every stored dataset (called after model changes)."""
    for name, df_harm in PROCESSED.items():
        try:
            preds = run_predictions(df_harm)
            _store_preds_only(name, df_harm, preds)
        except Exception:
            pass

def _store_preds_only(name, df_harm, preds):
    preds_store = {k: {kk: vv for kk, vv in v.items()
                       if kk not in ('predictions', 'probabilities')}
                   for k, v in preds.items()}
    preds_store['_rows'] = build_rows(df_harm, preds)
    PREDICTIONS[name]    = preds_store

# ── LOAD UPLOADED MODELS ──────────────────────────────────────
print("=" * 60)
print("  LEUKEMIA HARMONISER — STARTUP")
print("=" * 60)
print("\nUpload your pre-trained model pkl files now.")
print("(leukemia_model.pkl, leukemia_model_nn.pkl, leukemia_model_rf.pkl)\n")

uploaded = files.upload()

for fname, content in uploaded.items():
    try:
        bundle = joblib.load(io.BytesIO(content))
        key_lower = fname.lower()
        if 'rf' in key_lower or 'forest' in key_lower:
            key = 'Random Forest'
            bundle['display_name'] = 'Random Forest'
        elif 'nn' in key_lower or 'neural' in key_lower:
            key = 'Neural Network'
            bundle['display_name'] = 'Neural Network'
        else:
            key = 'XGBoost'
            bundle['display_name'] = 'XGBoost'
        MODELS[key] = bundle
        print(f"  ✓ {key} loaded from {fname}")
    except Exception as e:
        print(f"  ✗ Failed to load {fname}: {e}")

print(f"\n  {len(MODELS)} model(s) ready: {list(MODELS.keys())}")

# ── FASTAPI APPLICATION ───────────────────────────────────────
app = FastAPI(title="Leukemia Harmoniser")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
#  REST ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/api/status")
def ep_status():
    return {
        'models_loaded': [
            {'key': k, 'display_name': v.get('display_name', k),
             'metrics': v.get('metrics', {})}
            for k, v in MODELS.items()
        ],
        'datasets_loaded': list(RAW_DATASETS.keys()),
        'total_records': sum(len(d) for d in RAW_DATASETS.values()),
        'status': 'ready',
    }

@app.get("/api/overview")
def ep_overview():
    if not RAW_DATASETS:
        return {'total_records':0,'total_positive':0,'total_inconclusive':0,
                'total_negative':0,'datasets':[],'models':[]}
    ds_list = []
    for name, df_raw in RAW_DATASETS.items():
        stats = HARMONIZE_STATS.get(name, {})
        preds = PREDICTIONS.get(name, {})
        def _g(mk, field, default=0):
            return preds.get(mk, {}).get(field, default)
        ds_list.append({
            'name': name,
            'rows': len(df_raw),
            'cols': df_raw.shape[1],
            'confidence': stats.get('confidence', 0),
            'has_diagnosis': stats.get('has_diagnosis', False),
            **{f'{mk.lower().replace(" ","_")}_positive':      _g(mk,'positive')
               for mk in MODELS},
            **{f'{mk.lower().replace(" ","_")}_positive_rate': _g(mk,'positive_rate')
               for mk in MODELS},
            'high_risk':     _g('XGBoost','high_risk'),
            'inconclusive':  max(_g(mk,'inconclusive') for mk in MODELS) if MODELS else 0,
            'low_risk':      _g('XGBoost','low_risk'),
        })
    total  = sum(d['rows'] for d in ds_list)
    pos    = sum(d.get('xgboost_positive', 0) for d in ds_list)
    inc    = sum(d.get('inconclusive', 0) for d in ds_list)
    return {
        'total_records': total,
        'total_positive': pos,
        'total_inconclusive': inc,
        'total_negative': max(0, total - pos - inc),
        'datasets': ds_list,
        'models': [{'key': k, 'display_name': v.get('display_name', k),
                    'metrics': v.get('metrics',{})} for k,v in MODELS.items()],
    }

@app.get("/api/datasets")
def ep_datasets():
    out = []
    for name, df_raw in RAW_DATASETS.items():
        stats = HARMONIZE_STATS.get(name, {})
        md    = stats.get('mapping_detail', [])
        out.append({
            'name': name,
            'rows': len(df_raw),
            'cols': df_raw.shape[1],
            'confidence': stats.get('confidence', 0),
            'status': 'Active',
            'has_diagnosis': stats.get('has_diagnosis', False),
            'cbc_columns':     [m['original'] for m in md if m['match'] != 'Dropped'],
            'dropped_columns': [m['original'] for m in md if m['match'] == 'Dropped'],
        })
    return out

@app.post("/api/upload")
async def ep_upload(file: UploadFile = File(...)):
    content = await file.read()
    df_raw  = None
    for enc in ['utf-8','latin-1','cp1252']:
        try:
            df_raw = pd.read_csv(io.BytesIO(content), encoding=enc)
            break
        except Exception:
            continue
    if df_raw is None:
        raise HTTPException(400, "Could not parse CSV file")

    # clean up Colab's "(1)" suffixes
    name = re.sub(r'\s*\(\d+\)\s*$', '', file.filename.replace('.csv','')).strip()

    try:
        df_harm, stats = harmonize_df(df_raw)
    except Exception as e:
        raise HTTPException(500, f"Harmonisation failed: {e}")

    # Store dataset — predictions NOT run yet (manual trigger)
    RAW_DATASETS[name]    = df_raw
    PROCESSED[name]       = df_harm
    HARMONIZE_STATS[name] = stats
    PREDICTIONS[name]     = {}   # empty until user triggers

    return {
        'name': name,
        'rows': len(df_raw),
        'cols': df_raw.shape[1],
        'has_diagnosis': stats.get('has_diagnosis', False),
        'stats': {k: v for k, v in stats.items()
                  if k not in ('mapping_detail',)},
    }

@app.post("/api/predict/{dataset_name}")
def ep_run_predictions(dataset_name: str):
    match = process.extractOne(
        dataset_name, list(PROCESSED.keys()), scorer=fuzz.token_sort_ratio
    )
    if not match or match[1] < 40:
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")
    name     = match[0]
    df_harm  = PROCESSED[name]
    preds    = run_predictions(df_harm)
    _store_preds_only(name, df_harm, preds)
    summary  = {k: {kk: vv for kk, vv in v.items()
                    if kk not in ('predictions','probabilities')}
                for k, v in preds.items() if 'error' not in v}
    return {'name': name, 'model_results': summary}

@app.get("/api/harmonisation/{dataset_name}")
def ep_harmonisation(dataset_name: str):
    match = process.extractOne(
        dataset_name, list(HARMONIZE_STATS.keys()), scorer=fuzz.token_sort_ratio
    )
    if not match or match[1] < 40:
        raise HTTPException(404, "Dataset not found")
    return {'name': match[0], **HARMONIZE_STATS[match[0]]}

@app.get("/api/predictions/{dataset_name}")
def ep_predictions(dataset_name: str, page: int = 0, page_size: int = 25,
                   risk_filter: str = "All", search: str = ""):
    if not PREDICTIONS:
        return {'total_rows':0,'rows':[],'model_stats':{},'predicted':False}
    match = process.extractOne(
        dataset_name, list(PREDICTIONS.keys()), scorer=fuzz.token_sort_ratio
    )
    if not match or match[1] < 40:
        raise HTTPException(404, "Dataset not found")
    name  = match[0]
    entry = PREDICTIONS[name]
    if not entry or '_rows' not in entry:
        return {'total_rows':0,'rows':[],'model_stats':{},'predicted':False,
                'dataset': name}
    rows  = entry['_rows']
    stats = {k: v for k, v in entry.items() if k != '_rows'}
    if risk_filter != "All":
        rows = [r for r in rows if r.get('risk_level') == risk_filter]
    if search:
        rows = [r for r in rows if search in str(r.get('row_index',''))]
    total = len(rows)
    paged = rows[page*page_size:(page+1)*page_size]
    return {
        'dataset': name, 'total_rows': total,
        'page': page, 'page_size': page_size,
        'rows': paged, 'model_stats': stats, 'predicted': True,
    }

# ── TRAINING ENDPOINT ─────────────────────────────────────────
class TrainRequest(BaseModel):
    dataset_name:  str
    algorithm:     str            # 'XGBoost' | 'Random Forest' | 'Neural Network'
    model_name:    str            # user-supplied display name
    balance_method: str = 'SMOTE'
    n_estimators:  int   = 200
    max_depth:     int   = 6
    learning_rate: float = 0.1
    job_id:        str   = ''

def _training_worker(req: TrainRequest):
    jid = req.job_id
    job = TRAINING_JOBS[jid]
    log = job['log']

    def L(msg):
        log.append(msg)

    try:
        match = process.extractOne(
            req.dataset_name, list(PROCESSED.keys()), scorer=fuzz.token_sort_ratio
        )
        if not match or match[1] < 40:
            raise ValueError(f"Dataset '{req.dataset_name}' not found")
        name    = match[0]
        stats   = HARMONIZE_STATS.get(name, {})
        df_harm = PROCESSED[name].copy()

        if not stats.get('has_diagnosis'):
            raise ValueError(
                "This dataset has no diagnosis/label column. "
                "Training requires a labelled dataset."
            )

        diag_col = stats['diagnosis_col']
        df_raw   = RAW_DATASETS[name]
        L(f"Dataset: {name}  ({len(df_harm):,} rows)")

        raw_labels = df_raw[diag_col].astype(str).str.lower().str.strip()
        y = raw_labels.apply(
            lambda v: 1 if any(kw in v for kw in POSITIVE_KEYWORDS) else 0
        ).values
        pos_n = int(y.sum())
        neg_n = int(len(y) - pos_n)
        L(f"Positive: {pos_n:,}  |  Negative: {neg_n:,}")
        if pos_n < 10:
            raise ValueError("Too few positive cases (<10) to train.")

        le = LabelEncoder()
        df_harm['Gender_enc'] = le.fit_transform(df_harm['Gender'])
        feature_cols = [f for f in TARGET_FEATURES if f in df_harm.columns]
        X = df_harm[feature_cols].fillna(0).values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        L(f"Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

        if req.balance_method == 'SMOTE':
            L("Applying SMOTE...")
            try:
                k = min(5, pos_n - 1)
                sm = SMOTE(random_state=42, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                L(f"After SMOTE: {len(X_tr):,} samples")
            except Exception as e:
                L(f"SMOTE failed ({e}) — using raw distribution")

        scaler    = None
        threshold = 0.2

        # ── XGBoost ───────────────────────────────────────────
        if req.algorithm == 'XGBoost':
            from xgboost import XGBClassifier
            cw = neg_n / max(pos_n, 1) if req.balance_method == 'Class Weight' else 1.0
            L(f"Training XGBoost  trees={req.n_estimators}  depth={req.max_depth}  lr={req.learning_rate}")
            steps = sorted(set([
                req.n_estimators // 4, req.n_estimators // 2,
                3 * req.n_estimators // 4, req.n_estimators
            ]))
            model = None
            for step in steps:
                if step < 1:
                    continue
                m = XGBClassifier(
                    n_estimators=step, max_depth=req.max_depth,
                    learning_rate=req.learning_rate,
                    scale_pos_weight=cw, random_state=42,
                    eval_metric='logloss', verbosity=0,
                )
                m.fit(X_tr, y_tr)
                acc = accuracy_score(y_te, m.predict(X_te))
                L(f"  iter {step}/{req.n_estimators}  acc={acc*100:.1f}%")
                model = m

        # ── Random Forest ─────────────────────────────────────
        elif req.algorithm == 'Random Forest':
            cw = 'balanced' if req.balance_method == 'Class Weight' else None
            depth = req.max_depth if req.max_depth > 0 else None
            L(f"Training Random Forest  trees={req.n_estimators}  depth={depth}")
            steps = sorted(set([
                max(1, req.n_estimators // 4), req.n_estimators // 2,
                3 * req.n_estimators // 4, req.n_estimators
            ]))
            model = None
            for step in steps:
                m = RandomForestClassifier(
                    n_estimators=step, max_depth=depth,
                    class_weight=cw, random_state=42, n_jobs=-1,
                )
                m.fit(X_tr, y_tr)
                acc = accuracy_score(y_te, m.predict(X_te))
                L(f"  trees {step}/{req.n_estimators}  acc={acc*100:.1f}%")
                model = m

        # ── Neural Network ────────────────────────────────────
        elif req.algorithm == 'Neural Network':
            L("Scaling features...")
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_te   = scaler.transform(X_te)
            L("Training MLP  layers=(128,64,32)  max_iter=500")
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu', max_iter=500,
                random_state=42, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=15,
                verbose=False,
            )
            model.fit(X_tr, y_tr)
            curve = model.loss_curve_
            n = len(curve)
            sample_points = [curve[int(i * n / 5)] for i in range(1, 6)]
            for idx, loss in enumerate(sample_points, 1):
                L(f"  epoch ~{idx*n//5}/{n}  loss={loss:.4f}")
        else:
            raise ValueError(f"Unknown algorithm: {req.algorithm}")

        # ── Evaluate ──────────────────────────────────────────
        L("Evaluating...")
        X_eval = X_te
        probs  = model.predict_proba(X_eval)[:, 1]
        preds  = (probs >= threshold).astype(int)
        acc    = round(accuracy_score(y_te, preds)  * 100, 2)
        roc    = round(roc_auc_score(y_te, probs),  4)
        f1     = round(f1_score(y_te, preds, zero_division=0)        * 100, 2)
        prec   = round(precision_score(y_te, preds, zero_division=0) * 100, 2)
        rec    = round(recall_score(y_te, preds, zero_division=0)    * 100, 2)
        L(f"Accuracy:  {acc}%")
        L(f"ROC-AUC:   {roc}")
        L(f"F1:        {f1}%")
        L(f"Precision: {prec}%  |  Recall: {rec}%")

        # ── Save as new named model (alongside existing) ──────
        model_key = req.model_name or f"{req.algorithm}_{jid[:6]}"
        bundle = {
            'model':        model,
            'features':     feature_cols,
            'threshold':    threshold,
            'le_gender':    le,
            'scaler':       scaler,
            'display_name': model_key,
            'algorithm':    req.algorithm,
            'trained_on':   name,
            'metrics': {
                'accuracy': acc, 'roc_auc': roc,
                'f1': f1, 'precision': prec, 'recall': rec,
            },
        }
        MODELS[model_key] = bundle

        # Re-run all dataset predictions with updated model set
        _refresh_predictions_all()

        L(f"✓ Model '{model_key}' saved — predictions updated on all datasets")
        job['status']  = 'complete'
        job['metrics'] = bundle['metrics']
        job['model_key'] = model_key

    except Exception as e:
        L(f"✗ ERROR: {e}")
        job['status'] = 'error'
        traceback.print_exc()

@app.post("/api/train")
def ep_train(req: TrainRequest):
    if not req.job_id:
        req.job_id = str(uuid.uuid4())[:8]
    TRAINING_JOBS[req.job_id] = {
        'status': 'running', 'log': [], 'metrics': {}, 'model_key': ''
    }
    threading.Thread(
        target=_training_worker, args=(req,), daemon=True
    ).start()
    return {'job_id': req.job_id, 'status': 'started'}

@app.get("/api/train/status/{job_id}")
def ep_train_status(job_id: str):
    job = TRAINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.delete("/api/models/{model_key}")
def ep_delete_model(model_key: str):
    if model_key not in MODELS:
        raise HTTPException(404, "Model not found")
    del MODELS[model_key]
    _refresh_predictions_all()
    return {'deleted': model_key}

# ── FRONTEND ──────────────────────────────────────────────────
# The HTML/JS/React frontend is defined as a multi-line string
# and served from the root endpoint.

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Leukemia Harmoniser</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:#F0F2F5;color:#111827;font-size:14px}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#F1F5F9}
::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:3px}
.mono{font-family:'DM Mono',monospace}
.fade{animation:fadeIn .22s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.drawer-panel{animation:drawerIn .2s ease}
@keyframes drawerIn{from{transform:translateX(100%)}to{transform:translateX(0)}}
.spin{animation:spin .8s linear infinite;display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
button{font-family:'DM Sans',sans-serif;cursor:pointer}
button:not(:disabled):hover{filter:brightness(.93)}
input,select{font-family:'DM Sans',sans-serif;color:#111827}
input:focus,select:focus{outline:2px solid #1A56DB;outline-offset:1px}
table{border-collapse:collapse;width:100%}
tr:hover td{background:rgba(26,86,219,.025)!important}
.page{display:none}.page.active{display:block}
.nav-btn{width:100%;display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;border:none;background:transparent;color:#94A3B8;font-size:13px;font-weight:400;text-align:left;margin-bottom:2px;transition:background .12s,color .12s;cursor:pointer}
.nav-btn.active{background:#1A56DB;color:#fff;font-weight:600}
.nav-btn:hover:not(.active){background:#1E293B;color:#CBD5E1}
.card{background:#fff;border:1px solid #E5E7EB;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.06);padding:24px}
.badge{display:inline-block;border-radius:20px;padding:2px 9px;font-size:11px;font-weight:600;white-space:nowrap;border:1px solid}
.badge-pos{background:#FEF2F2;color:#DC2626;border-color:#FECACA}
.badge-neg{background:#ECFDF5;color:#059669;border-color:#A7F3D0}
.badge-inc{background:#FFFBEB;color:#D97706;border-color:#FDE68A}
.badge-act{background:#EBF5FF;color:#1A56DB;border-color:#BFDBFE}
.badge-neu{background:#F3F4F6;color:#6B7280;border-color:#E5E7EB}
.btn{border-radius:8px;font-size:13px;font-weight:600;border:1px solid;padding:8px 18px}
.btn-primary{background:#1A56DB;color:#fff;border-color:#1A56DB}
.btn-outline{background:transparent;color:#1A56DB;border-color:#1A56DB}
.btn-ghost{background:transparent;color:#6B7280;border-color:#E5E7EB}
.btn-danger{background:#DC2626;color:#fff;border-color:#DC2626}
.btn-success{background:#ECFDF5;color:#059669;border-color:#A7F3D0}
.btn-sm{padding:5px 13px;font-size:12px}
.btn-lg{padding:11px 28px;font-size:14px}
.btn:disabled{opacity:.5;cursor:not-allowed}
.th{text-align:left;padding:8px 10px;font-size:10px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;border-bottom:2px solid #E5E7EB}
.td{padding:9px 10px;border-bottom:1px solid #E5E7EB;font-size:13px}
.input-field{width:100%;padding:9px 13px;border:1px solid #E5E7EB;border-radius:8px;font-size:13px;background:#fff}
.section-title{font-size:16px;font-weight:700;color:#111827;margin-bottom:4px}
.section-sub{font-size:12px;color:#6B7280;margin-bottom:16px}
.kpi-card{flex:1;background:#fff;border:1px solid #E5E7EB;border-radius:12px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.32);z-index:200;display:none}
.overlay.show{display:block}
.drawer{position:fixed;right:0;top:0;bottom:0;width:420px;background:#fff;padding:26px;overflow-y:auto;box-shadow:-4px 0 24px rgba(0,0,0,.12);z-index:201;transform:translateX(100%);transition:transform .2s ease}
.drawer.open{transform:translateX(0)}
.console-log{background:#0F172A;border-radius:10px;padding:18px;height:420px;overflow-y:auto;font-family:'DM Mono',monospace;font-size:11px}
.toast{position:fixed;bottom:22px;right:22px;z-index:9999;padding:12px 18px;border-radius:10px;font-size:13px;font-weight:600;color:#fff;box-shadow:0 4px 14px rgba(0,0,0,.22);display:flex;gap:12px;align-items:center;opacity:0;transition:opacity .2s}
.toast.show{opacity:1}
.toggle-track{width:38px;height:20px;border-radius:10px;border:none;position:relative;cursor:pointer;transition:background .15s}
.toggle-thumb{position:absolute;top:2px;width:16px;height:16px;border-radius:50%;background:#fff;transition:left .15s}
.progress-bar{height:4px;background:#F3F4F6;border-radius:2px;overflow:hidden}
.progress-fill{height:100%;border-radius:2px;transition:width .3s}
.conf-ring{transform:rotate(-90deg);transform-origin:50% 50%}
.expandable-row{background:#F8FAFF;border-bottom:1px solid #E5E7EB}
.tag{display:inline-block;border-radius:4px;padding:2px 7px;font-size:10px;font-family:'DM Mono',monospace;border:1px solid}
.tag-cbc{background:#EBF5FF;color:#1A56DB;border-color:#BFDBFE}
.tag-drop{background:#F3F4F6;color:#9CA3AF;border-color:#E5E7EB;text-decoration:line-through}
</style>
</head>
<body>

<!-- LAYOUT -->
<div style="display:flex;min-height:100vh">

  <!-- SIDEBAR -->
  <aside id="sidebar" style="width:240px;min-height:100vh;background:#0F172A;display:flex;flex-direction:column;position:fixed;left:0;top:0;z-index:100">
    <div style="padding:22px 20px 18px;border-bottom:1px solid #1E293B">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:36px;height:36px;border-radius:8px;background:#1A56DB;display:flex;align-items:center;justify-content:center;font-size:18px">🧬</div>
        <div>
          <p style="color:#F1F5F9;font-size:13px;font-weight:700">Leukemia</p>
          <p style="color:#64748B;font-size:10px;margin-top:1px">Harmoniser v2.0</p>
        </div>
      </div>
    </div>
    <nav style="flex:1;padding:10px 8px" id="nav-list"></nav>
    <div style="padding:14px;border-top:1px solid #1E293B" id="sys-status">
      <p style="font-size:9px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:.09em;margin-bottom:10px">System Status</p>
      <div id="status-rows"></div>
      <p id="status-counts" style="font-size:10px;color:#475569;margin-top:8px">0 datasets · 0 records</p>
    </div>
  </aside>

  <!-- MAIN -->
  <div style="margin-left:240px;flex:1;display:flex;flex-direction:column;min-height:100vh">
    <header style="background:#fff;border-bottom:1px solid #E5E7EB;padding:13px 30px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:50">
      <div style="display:flex;align-items:center;gap:8px;font-size:13px;color:#6B7280">
        <span>Leukemia Harmoniser</span><span>›</span>
        <span id="breadcrumb" style="color:#111827;font-weight:600">Overview</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px" id="header-badges"></div>
    </header>
    <main style="flex:1;padding:26px 30px" id="main-content"></main>
  </div>
</div>

<!-- DRAWER OVERLAY -->
<div class="overlay" id="overlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer"></div>

<!-- TOAST -->
<div class="toast" id="toast">
  <span id="toast-msg"></span>
  <button onclick="hideToast()" style="background:none;border:none;color:#fff;font-size:16px">✕</button>
</div>

<script>
// ── CONSTANTS ────────────────────────────────────────────────
const C = {
  pri:'#1A56DB', priL:'#EBF5FF', priBrd:'#BFDBFE',
  red:'#DC2626', redL:'#FEF2F2',
  amb:'#D97706', ambL:'#FFFBEB',
  grn:'#059669', grnL:'#ECFDF5',
  brd:'#E5E7EB', txt:'#111827', sub:'#6B7280', lite:'#9CA3AF',
  MODEL_COLORS:['#1A56DB','#7C3AED','#0D9488','#D97706'],
};

const NAV = [
  {id:'overview',  label:'Overview',          icon:'⬡'},
  {id:'intake',    label:'Data Intake',        icon:'↑'},
  {id:'harmonise', label:'Harmonisation',      icon:'⟳'},
  {id:'training',  label:'Model Training',     icon:'◈'},
  {id:'dashboard', label:'Analysis Dashboard', icon:'▦'},
  {id:'patients',  label:'Patient Records',    icon:'♦'},
  {id:'reports',   label:'Reports',            icon:'▤'},
];

// ── STATE ────────────────────────────────────────────────────
let STATE = {
  page: 'overview',
  status: null,
  datasets: [],
  overview: null,
  trainingJobId: null,
  trainingInterval: null,
  drawerData: null,
  drawerModelKeys: [],
};

// ── UTILS ────────────────────────────────────────────────────
const esc = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const fmt = v => v!=null ? Number(v).toLocaleString() : '—';
const fmtF = (v,d=1) => v!=null ? Number(v).toFixed(d) : '—';

function riskClass(r) {
  const l = (r||'').toLowerCase();
  if(l.includes('high')||l==='positive') return 'pos';
  if(l.includes('inconclusive'))         return 'inc';
  if(l.includes('low')||l==='negative')  return 'neg';
  return 'neu';
}
function riskColor(r) {
  const c = riskClass(r);
  return c==='pos'?C.red : c==='inc'?C.amb : C.grn;
}
function riskBg(r) {
  const c = riskClass(r);
  return c==='pos'?C.redL : c==='inc'?C.ambL : C.grnL;
}

function badge(label, type='neu', size='sm') {
  return `<span class="badge badge-${type}" style="${size==='md'?'padding:4px 13px;font-size:12px':''}">${esc(label)}</span>`;
}
function btn(label, cls='btn-primary', size='', extra='', disabled=false) {
  return `<button class="btn ${cls} ${size?'btn-'+size:''}" ${extra} ${disabled?'disabled':''}>${label}</button>`;
}
function card(inner, style='') {
  return `<div class="card" style="${style}">${inner}</div>`;
}
function sTitle(title, sub='') {
  return `<p class="section-title">${esc(title)}</p>${sub?`<p class="section-sub">${esc(sub)}</p>`:''}`;
}
function loader(msg='Loading...') {
  return `<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:56px;gap:14px">
    <div class="spin" style="width:32px;height:32px;border:3px solid #E5E7EB;border-top-color:#1A56DB;border-radius:50%"></div>
    <p style="font-size:13px;color:#6B7280">${esc(msg)}</p>
  </div>`;
}
function empty(icon, msg, sub='') {
  return card(`<div style="text-align:center;padding:48px 0">
    <div style="font-size:36px;margin-bottom:14px">${icon}</div>
    <p style="font-size:15px;font-weight:600;color:#111827;margin-bottom:6px">${esc(msg)}</p>
    ${sub?`<p style="font-size:13px;color:#6B7280">${esc(sub)}</p>`:''}
  </div>`);
}
function progressBar(pct, color=C.pri) {
  return `<div class="progress-bar"><div class="progress-fill" style="width:${Math.min(100,pct)}%;background:${color}"></div></div>`;
}
function confidenceRing(pct) {
  const r=31, circ=2*Math.PI*r;
  const color = pct>=80 ? C.grn : C.amb;
  return `<div style="position:relative;width:74px;height:74px;flex-shrink:0">
    <svg width="74" height="74" viewBox="0 0 74 74">
      <circle cx="37" cy="37" r="${r}" fill="none" stroke="#F3F4F6" stroke-width="7"/>
      <circle cx="37" cy="37" r="${r}" fill="none" stroke="${color}" stroke-width="7"
        stroke-dasharray="${circ*pct/100} ${circ}" stroke-linecap="round"
        class="conf-ring"/>
    </svg>
    <div class="mono" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;color:${color}">${pct}%</div>
  </div>`;
}

// ── TOAST ────────────────────────────────────────────────────
let _toastTimeout = null;
function showToast(msg, type='success') {
  const t = document.getElementById('toast');
  document.getElementById('toast-msg').textContent = msg;
  t.style.background = type==='error' ? C.red : C.grn;
  t.classList.add('show');
  if(_toastTimeout) clearTimeout(_toastTimeout);
  _toastTimeout = setTimeout(hideToast, 4000);
}
function hideToast() {
  document.getElementById('toast').classList.remove('show');
}

// ── DRAWER ───────────────────────────────────────────────────
function openDrawer(html) {
  document.getElementById('drawer').innerHTML = html;
  document.getElementById('drawer').classList.add('open');
  document.getElementById('overlay').classList.add('show');
}
function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('overlay').classList.remove('show');
}

// ── API ──────────────────────────────────────────────────────
async function api(path, opts={}) {
  const r = await fetch(path, opts);
  if(!r.ok) {
    const e = await r.json().catch(()=>({detail:'Request failed'}));
    throw new Error(e.detail||'Request failed');
  }
  return r.json();
}

// ── SIDEBAR RENDER ───────────────────────────────────────────
function renderSidebar() {
  document.getElementById('nav-list').innerHTML = NAV.map(n=>`
    <button class="nav-btn ${STATE.page===n.id?'active':''}" onclick="navigate('${n.id}')">
      <span style="font-size:13px;width:16px;text-align:center">${n.icon}</span>${n.label}
    </button>`).join('');
}

function renderStatus() {
  const s = STATE.status;
  const checks = [
    {label:'Models Loaded',   ok: (s?.models_loaded?.length||0)>0},
    {label:'Datasets Active', ok: (s?.datasets_loaded?.length||0)>0},
    {label:'API Ready',       ok: s?.status==='ready'},
  ];
  document.getElementById('status-rows').innerHTML = checks.map(c=>`
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
      <span style="font-size:11px;color:#64748B">${c.label}</span>
      <span style="width:7px;height:7px;border-radius:50%;background:${c.ok?C.grn:'#EF4444'};display:inline-block"></span>
    </div>`).join('');
  document.getElementById('status-counts').textContent =
    `${s?.datasets_loaded?.length||0} dataset(s) · ${(s?.total_records||0).toLocaleString()} records`;

  const hb = document.getElementById('header-badges');
  const ds = STATE.datasets.length;
  const ml = s?.models_loaded?.length||0;
  hb.innerHTML = `
    ${badge(`${ds} Dataset${ds!==1?'s':''} Active`,'neg','md')}
    ${badge(`${ml} Model${ml!==1?'s':''} Loaded`,'act','md')}
    <div style="width:32px;height:32px;border-radius:50%;background:#1A56DB;display:flex;align-items:center;justify-content:center;color:#fff;font-size:13px;font-weight:700">V</div>`;
}

// ── NAVIGATION ───────────────────────────────────────────────
const TITLES = {
  overview:'Overview', intake:'Data Intake', harmonise:'Harmonisation',
  training:'Model Training', dashboard:'Analysis Dashboard',
  patients:'Patient Records', reports:'Reports',
};

async function navigate(id) {
  STATE.page = id;
  document.getElementById('breadcrumb').textContent = TITLES[id];
  renderSidebar();
  const mc = document.getElementById('main-content');
  mc.innerHTML = loader();
  switch(id) {
    case 'overview':  await renderOverview();  break;
    case 'intake':    await renderIntake();    break;
    case 'harmonise': await renderHarmonise(); break;
    case 'training':  await renderTraining();  break;
    case 'dashboard': await renderDashboard(); break;
    case 'patients':  await renderPatients();  break;
    case 'reports':   await renderReports();   break;
  }
}

// ── REFRESH GLOBAL STATE ─────────────────────────────────────
async function refreshState() {
  try {
    [STATE.status, STATE.datasets, STATE.overview] = await Promise.all([
      api('/api/status'),
      api('/api/datasets'),
      api('/api/overview'),
    ]);
    renderStatus();
  } catch(e) { console.warn('State refresh failed:', e); }
}

// ════════════════════════════════════════════════════════════
// PAGE: OVERVIEW
// ════════════════════════════════════════════════════════════
async function renderOverview() {
  const mc = document.getElementById('main-content');
  let d;
  try { d = await api('/api/overview'); } catch(e) {
    mc.innerHTML = `<p style="color:red">Failed to load overview: ${e.message}</p>`; return;
  }
  if(!d.total_records) {
    mc.innerHTML = `<div class="fade">
      <h1 style="font-size:23px;font-weight:800;margin-bottom:22px">Command Overview</h1>
      ${empty('📋','No datasets loaded yet','Go to Data Intake to upload your first CSV dataset')}
    </div>`; return;
  }

  const pie = [
    {name:'Positive',    value:d.total_positive,    color:C.red},
    {name:'Inconclusive',value:d.total_inconclusive, color:C.amb},
    {name:'Negative',    value:d.total_negative,     color:C.grn},
  ];

  // Build model cols for per-dataset table
  const modelKeys = d.models.map(m=>m.key);

  mc.innerHTML = `<div class="fade">
    <h1 style="font-size:23px;font-weight:800;margin-bottom:4px">Command Overview</h1>
    <p style="font-size:13px;color:#6B7280;margin-bottom:22px">Real-time summary across all datasets and models</p>

    <div style="display:flex;gap:14px;margin-bottom:20px">
      ${kpiCard('Total Records', fmt(d.total_records), `${d.datasets.length} dataset(s)`, '📋', C.pri)}
      ${kpiCard('Leukemia Positive', fmt(d.total_positive), `${d.total_records>0?((d.total_positive/d.total_records)*100).toFixed(1):0}% rate`, '🔴', C.red)}
      ${kpiCard('Inconclusive', fmt(d.total_inconclusive), 'Further testing needed', '⚠️', C.amb)}
      ${kpiCard('Models Active', d.models.length, d.models.map(m=>m.display_name||m.key).join(' · '), '◈', C.grn)}
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
      ${card(`
        ${sTitle('Case Distribution','All datasets combined')}
        <div style="display:flex;align-items:center;gap:14px">
          <canvas id="pie-canvas" width="160" height="160"></canvas>
          <div style="flex:1">
            ${pie.map(p=>`
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                <div style="display:flex;align-items:center;gap:8px">
                  <div style="width:9px;height:9px;border-radius:2px;background:${p.color}"></div>
                  <span style="font-size:13px;color:#6B7280">${p.name}</span>
                </div>
                <span class="mono" style="font-size:13px;font-weight:700">${fmt(p.value)}</span>
              </div>`).join('')}
          </div>
        </div>`)}
      ${card(`
        ${sTitle('Positive Rate per Dataset','All loaded models')}
        <canvas id="bar-canvas" width="400" height="180"></canvas>`)}
    </div>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px">
      ${d.datasets.map((ds,i)=>{
        const col = C.MODEL_COLORS[i%4];
        return card(`
          <p style="font-size:13px;font-weight:700;margin-bottom:3px">${esc(ds.name.replace(/_/g,' '))}</p>
          <p style="font-size:12px;color:#6B7280;margin-bottom:12px">
            ${fmt(ds.rows)} rows · ${ds.confidence}% confidence
            ${ds.has_diagnosis?'<span style="color:#059669;font-weight:600"> · Training ready</span>':''}
          </p>
          ${d.models.map(m=>{
            const rk = m.key.toLowerCase().replace(/ /g,'_');
            const pos  = ds[`${rk}_positive`]||0;
            const rate = ds[`${rk}_positive_rate`]||0;
            return `<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #E5E7EB">
              <span style="font-size:12px;color:#6B7280">${esc(m.display_name||m.key)}</span>
              <div style="display:flex;gap:8px">
                <span class="mono" style="font-size:12px;font-weight:700;color:${col}">${fmt(pos)}</span>
                <span style="font-size:11px;color:#9CA3AF">(${(+rate).toFixed(1)}%)</span>
              </div>
            </div>`;
          }).join('')}`);
      }).join('')}
    </div>
  </div>`;

  // Draw pie chart on canvas
  drawPie('pie-canvas', pie);

  // Draw bar chart
  const barLabels = d.datasets.map(ds=>ds.name.replace(/_/g,' ').split(' ').slice(0,2).join(' '));
  const barSeries = d.models.map((m,i)=>({
    label: m.display_name||m.key,
    color: C.MODEL_COLORS[i%4],
    data: d.datasets.map(ds=>{
      const rk=m.key.toLowerCase().replace(/ /g,'_');
      return +(ds[`${rk}_positive_rate`]||0).toFixed(1);
    }),
  }));
  drawGroupedBar('bar-canvas', barLabels, barSeries, '%');
}

function kpiCard(label, value, sub, icon, color) {
  return `<div class="kpi-card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <div>
        <p style="font-size:10px;color:#6B7280;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">${label}</p>
        <p class="mono" style="font-size:28px;font-weight:800;color:#111827;line-height:1">${value}</p>
        ${sub?`<p style="font-size:12px;color:#6B7280;margin-top:5px">${esc(sub)}</p>`:''}
      </div>
      <div style="width:42px;height:42px;border-radius:10px;background:${color}1a;display:flex;align-items:center;justify-content:center;font-size:19px">${icon}</div>
    </div>
  </div>`;
}

// ── CANVAS CHARTS ────────────────────────────────────────────
function drawPie(id, data) {
  const canvas = document.getElementById(id);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx=80, cy=80, outerR=72, innerR=48;
  const total = data.reduce((s,d)=>s+d.value,0);
  if(!total) return;
  let angle = -Math.PI/2;
  data.forEach(d=>{
    const sweep = (d.value/total)*2*Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx,cy);
    ctx.arc(cx,cy,outerR,angle,angle+sweep);
    ctx.closePath();
    ctx.fillStyle = d.color;
    ctx.fill();
    angle += sweep;
  });
  // donut hole
  ctx.beginPath();
  ctx.arc(cx,cy,innerR,0,2*Math.PI);
  ctx.fillStyle = '#fff';
  ctx.fill();
}

function drawGroupedBar(id, labels, series, unit='') {
  const canvas = document.getElementById(id);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const pad={l:34,r:10,t:10,b:32};
  const chartW=W-pad.l-pad.r, chartH=H-pad.t-pad.b;
  const n=labels.length, s=series.length;
  if(!n||!s) return;
  const maxVal = Math.max(...series.flatMap(sr=>sr.data), 1);
  const groupW = chartW/n;
  const barW = Math.min(18, groupW/(s+1));

  // axes
  ctx.strokeStyle='#F3F4F6'; ctx.lineWidth=1;
  [0,.25,.5,.75,1].forEach(t=>{
    const y=pad.t+chartH*(1-t);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+chartW,y); ctx.stroke();
    ctx.fillStyle='#9CA3AF'; ctx.font='9px DM Sans,sans-serif'; ctx.textAlign='right';
    ctx.fillText((t*maxVal).toFixed(1)+unit, pad.l-4, y+3);
  });

  // bars
  series.forEach((sr,si)=>{
    sr.data.forEach((val,li)=>{
      const x = pad.l + li*groupW + (si-(s-1)/2)*barW*1.2 + groupW/2 - barW/2;
      const h = (val/maxVal)*chartH;
      const y = pad.t+chartH-h;
      ctx.fillStyle=sr.color;
      ctx.beginPath();
      const r=2;
      ctx.moveTo(x+r,y); ctx.lineTo(x+barW-r,y);
      ctx.quadraticCurveTo(x+barW,y,x+barW,y+r);
      ctx.lineTo(x+barW,y+h); ctx.lineTo(x,y+h);
      ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y);
      ctx.closePath(); ctx.fill();
    });
  });

  // x labels
  ctx.fillStyle='#9CA3AF'; ctx.font='9px DM Sans,sans-serif'; ctx.textAlign='center';
  labels.forEach((l,i)=>{
    ctx.fillText(l, pad.l+i*groupW+groupW/2, H-8);
  });

  // legend
  ctx.textAlign='left';
  series.forEach((sr,i)=>{
    const lx = pad.l+i*80, ly=8;
    ctx.fillStyle=sr.color;
    ctx.fillRect(lx,ly-6,8,8);
    ctx.fillStyle='#6B7280'; ctx.font='9px DM Sans,sans-serif';
    ctx.fillText(sr.label,lx+11,ly+1);
  });
}

function drawStackedBar(id, categories, series) {
  // series: [{label, color, data:[]}] — data sums to total per category
  const canvas = document.getElementById(id);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const pad={l:90,r:14,t:14,b:10};
  const chartW=W-pad.l-pad.r, chartH=H-pad.t-pad.b;
  const maxVal = Math.max(...categories.map((_,i)=>series.reduce((s,sr)=>s+sr.data[i],0)),1);
  const rowH = chartH/categories.length;
  const barH = Math.min(18,rowH*0.7);

  ctx.strokeStyle='#F3F4F6'; ctx.lineWidth=1;

  categories.forEach((cat,ci)=>{
    let x=pad.l;
    const y=pad.t+ci*rowH+(rowH-barH)/2;
    series.forEach(sr=>{
      const w=(sr.data[ci]/maxVal)*chartW;
      if(w<1) return;
      ctx.fillStyle=sr.color;
      ctx.fillRect(x,y,w,barH);
      x+=w;
    });
    ctx.fillStyle='#6B7280'; ctx.font='10px DM Sans,sans-serif'; ctx.textAlign='right';
    ctx.fillText(cat, pad.l-6, pad.t+ci*rowH+rowH/2+4);
  });

  // legend bottom
  const ly=H-4;
  ctx.textAlign='left';
  series.forEach((sr,i)=>{
    const lx=pad.l+i*110;
    ctx.fillStyle=sr.color; ctx.fillRect(lx,ly-7,8,8);
    ctx.fillStyle='#6B7280'; ctx.font='9px DM Sans,sans-serif';
    ctx.fillText(sr.label,lx+11,ly);
  });
}

// ════════════════════════════════════════════════════════════
// PAGE: DATA INTAKE
// ════════════════════════════════════════════════════════════
async function renderIntake() {
  const mc = document.getElementById('main-content');
  let datasets;
  try { datasets = await api('/api/datasets'); } catch(e) { datasets=[]; }

  mc.innerHTML = `<div class="fade">
    <h1 style="font-size:23px;font-weight:800;margin-bottom:4px">Data Intake</h1>
    <p style="font-size:13px;color:#6B7280;margin-bottom:22px">Upload CBC datasets · harmonisation runs automatically · click Run Predictions when ready</p>

    ${card(`
      <div id="drop-zone" style="border:2px dashed #E5E7EB;border-radius:10px;padding:36px 24px;text-align:center;background:#FAFAFA;transition:all .18s"
        ondragover="dzOver(event)" ondragleave="dzLeave()" ondrop="dzDrop(event)">
        <div id="dz-inner">
          <div style="font-size:34px;margin-bottom:12px">📁</div>
          <p style="font-size:16px;font-weight:700;margin-bottom:6px">Drop your CBC dataset here</p>
          <p style="font-size:13px;color:#6B7280;margin-bottom:18px">Any column naming convention · Auto-harmonised · CSV format</p>
          <label class="btn btn-primary" style="padding:11px 28px;font-size:13px;cursor:pointer">
            Browse Files
            <input type="file" accept=".csv" style="display:none" onchange="handleFileInput(this)">
          </label>
          <p style="font-size:11px;color:#9CA3AF;margin-top:10px">UTF-8, Latin-1, CP1252 supported</p>
        </div>
      </div>`)}

    <div id="upload-result"></div>

    ${card(`
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">
        <div>
          <p class="section-title">Dataset Registry</p>
          <p class="section-sub" style="margin-bottom:0">Loaded datasets — click row to expand column details</p>
        </div>
        ${badge(`${datasets.length} Active`,'act','md')}
      </div>
      <div id="datasets-table">
        ${renderDatasetsTable(datasets)}
      </div>`)}
  </div>`;
}

function renderDatasetsTable(datasets) {
  if(!datasets.length) return `<p style="text-align:center;padding:28px 0;color:#6B7280;font-size:13px">No datasets loaded. Upload a CSV above.</p>`;
  return `<table>
    <thead><tr>
      <th class="th">Dataset</th><th class="th">Rows</th><th class="th">Cols</th>
      <th class="th">Confidence</th><th class="th">Diagnosis</th><th class="th">Status</th><th class="th">Actions</th>
    </tr></thead>
    <tbody>
      ${datasets.map((d,i)=>{
        const col = C.MODEL_COLORS[i%4];
        return `
          <tr onclick="toggleExpand(${i})" style="cursor:pointer" id="ds-row-${i}">
            <td class="td">
              <div style="display:flex;align-items:center;gap:9px">
                <div style="width:3px;height:32px;border-radius:2px;background:${col};flex-shrink:0"></div>
                <div>
                  <p style="font-size:13px;font-weight:600">${esc(d.name.replace(/_/g,' '))}</p>
                  <p class="mono" style="font-size:10px;color:#9CA3AF">${esc(d.name)}</p>
                </div>
              </div>
            </td>
            <td class="td"><span class="mono" style="font-weight:600">${fmt(d.rows)}</span></td>
            <td class="td"><span class="mono">${d.cols}</span></td>
            <td class="td">
              <div style="display:flex;align-items:center;gap:6px">
                <div style="width:42px;height:4px;background:#F3F4F6;border-radius:2px;overflow:hidden">
                  <div style="width:${d.confidence}%;height:100%;background:${d.confidence>=80?C.grn:C.amb};border-radius:2px"></div>
                </div>
                <span class="mono" style="font-size:11px;color:#6B7280">${d.confidence}%</span>
              </div>
            </td>
            <td class="td">${badge(d.has_diagnosis?'Training Ready':'Prediction Only', d.has_diagnosis?'neg':'neu')}</td>
            <td class="td">${badge('Active','act')}</td>
            <td class="td">
              <button class="btn btn-primary btn-sm" id="pred-btn-${i}"
                onclick="event.stopPropagation();runPredictions('${esc(d.name)}',${i})">
                ▶ Run Predictions
              </button>
            </td>
          </tr>
          <tr id="ds-expand-${i}" style="display:none">
            <td colspan="7" class="expandable-row" style="padding:14px 16px">
              <p style="font-size:10px;font-weight:700;color:#1A56DB;margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em">
                CBC Columns Matched (${d.cbc_columns.length})
              </p>
              <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:12px">
                ${d.cbc_columns.map(c=>`<span class="tag tag-cbc">${esc(c)}</span>`).join('')}
              </div>
              <p style="font-size:10px;font-weight:700;color:#6B7280;margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em">
                Dropped (${d.dropped_columns.length})
              </p>
              <div style="display:flex;flex-wrap:wrap;gap:5px">
                ${d.dropped_columns.map(c=>`<span class="tag tag-drop">${esc(c)}</span>`).join('')}
              </div>
            </td>
          </tr>`;
      }).join('')}
    </tbody>
  </table>`;
}

function toggleExpand(i) {
  const row = document.getElementById(`ds-expand-${i}`);
  if(row) row.style.display = row.style.display==='none'?'':'none';
}

function dzOver(e) { e.preventDefault(); document.getElementById('drop-zone').style.borderColor=C.pri; document.getElementById('drop-zone').style.background=C.priL; }
function dzLeave() { document.getElementById('drop-zone').style.borderColor='#E5E7EB'; document.getElementById('drop-zone').style.background='#FAFAFA'; }
function dzDrop(e) { e.preventDefault(); dzLeave(); const f=e.dataTransfer.files[0]; if(f) uploadFile(f); }
function handleFileInput(input) { if(input.files[0]) uploadFile(input.files[0]); }

async function uploadFile(file) {
  if(!file.name.toLowerCase().endsWith('.csv')) { showToast('Only CSV files supported','error'); return; }
  const dz = document.getElementById('dz-inner');
  dz.innerHTML = `<div style="display:flex;flex-direction:column;align-items:center;gap:12px">
    <div class="spin" style="width:34px;height:34px;border:3px solid #E5E7EB;border-top-color:#1A56DB;border-radius:50%"></div>
    <p style="font-size:14px;font-weight:600">Processing dataset...</p>
    <p style="font-size:12px;color:#6B7280">Reading → Column detection → Harmonising → Cleaning</p>
  </div>`;
  try {
    const fd = new FormData(); fd.append('file', file);
    const data = await api('/api/upload', {method:'POST', body:fd});
    document.getElementById('upload-result').innerHTML = `
      <div class="card" style="margin:16px 0;border:1px solid #A7F3D0;background:#ECFDF5">
        <p style="font-size:14px;font-weight:700;color:#059669;margin-bottom:8px">✓ Upload complete: ${esc(data.name)}</p>
        <div style="display:flex;flex-wrap:wrap;gap:20px;font-size:13px">
          <span><b>${fmt(data.rows)}</b> rows · <b>${data.cols}</b> columns</span>
          <span>Confidence: <b style="color:#059669">${data.stats?.confidence}%</b></span>
          <span>Outliers cleaned: <b>${data.stats?.outliers_cleaned||0}</b></span>
          <span>Nulls filled: <b>${data.stats?.nulls_filled||0}</b></span>
          ${data.has_diagnosis?`<span style="color:#059669;font-weight:600">✓ Diagnosis column found — training ready</span>`:''}
        </div>
      </div>`;
    showToast(`✓ ${data.name} uploaded — click Run Predictions to analyse`);
    // reset drop zone
    dz.innerHTML = `
      <div style="font-size:34px;margin-bottom:12px">📁</div>
      <p style="font-size:16px;font-weight:700;margin-bottom:6px">Drop your CBC dataset here</p>
      <p style="font-size:13px;color:#6B7280;margin-bottom:18px">Any column naming convention · Auto-harmonised · CSV format</p>
      <label class="btn btn-primary" style="padding:11px 28px;font-size:13px;cursor:pointer">
        Browse Files<input type="file" accept=".csv" style="display:none" onchange="handleFileInput(this)">
      </label>
      <p style="font-size:11px;color:#9CA3AF;margin-top:10px">UTF-8, Latin-1, CP1252 supported</p>`;
    await refreshState();
    const datasets = await api('/api/datasets');
    document.getElementById('datasets-table').innerHTML = renderDatasetsTable(datasets);
  } catch(e) {
    showToast(`✗ ${e.message}`, 'error');
    dz.innerHTML = `
      <div style="font-size:34px;margin-bottom:12px">📁</div>
      <p style="font-size:16px;font-weight:700;margin-bottom:6px">Drop your CBC dataset here</p>
      <label class="btn btn-primary" style="padding:11px 28px;font-size:13px;cursor:pointer">
        Browse Files<input type="file" accept=".csv" style="display:none" onchange="handleFileInput(this)">
      </label>`;
  }
}

async function runPredictions(name, idx) {
  const btn = document.getElementById(`pred-btn-${idx}`);
  if(btn) { btn.disabled=true; btn.innerHTML=`<span class="spin">◈</span> Running...`; }
  try {
    await api(`/api/predict/${encodeURIComponent(name)}`, {method:'POST'});
    showToast(`✓ Predictions complete for ${name}`);
    await refreshState();
  } catch(e) { showToast(`✗ ${e.message}`,'error'); }
  if(btn) { btn.disabled=false; btn.innerHTML='▶ Run Predictions'; }
}

// ════════════════════════════════════════════════════════════
// PAGE: HARMONISATION
// ════════════════════════════════════════════════════════════
async function renderHarmonise() {
  const mc = document.getElementById('main-content');
  const datasets = STATE.datasets;
  if(!datasets.length) {
    mc.innerHTML = `<div class="fade"><h1 style="font-size:23px;font-weight:800;margin-bottom:22px">Harmonisation Engine</h1>${empty('⟳','No datasets loaded','Upload a dataset from Data Intake first')}</div>`;
    return;
  }
  mc.innerHTML = `<div class="fade">
    <h1 style="font-size:23px;font-weight:800;margin-bottom:4px">Harmonisation Engine</h1>
    <p style="font-size:13px;color:#6B7280;margin-bottom:22px">Column mapping, scale correction, outlier cleaning</p>
    ${card(hStepperHTML(),'margin-bottom:20px')}
    <div style="display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap" id="harm-tabs">
      ${datasets.map((d,i)=>`
        <button onclick="loadHarmStats('${esc(d.name)}',${i})"
          class="harm-tab" id="htab-${i}"
          style="padding:8px 16px;border-radius:8px;border:1px solid ${i===0?C.pri:'#E5E7EB'};
            background:${i===0?C.priL:'#fff'};color:${i===0?C.pri:'#6B7280'};font-size:12px;font-weight:600">
          ${esc(d.name.replace(/_/g,' ').split(' ').slice(0,3).join(' '))}
        </button>`).join('')}
    </div>
    <div id="harm-content">${loader()}</div>
  </div>`;
  loadHarmStats(datasets[0].name, 0);
}

function hStepperHTML() {
  const steps = ['Column Detection','Fuzzy Mapping','Scale Correction','Outlier Cleaning','Null Handling','Ready'];
  return `<div style="display:flex;align-items:center">
    ${steps.map((s,i)=>`
      <div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0">
        <div style="width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;
          background:${i<5?C.grn:C.pri};color:#fff;font-size:11px;font-weight:700">${i<5?'✓':'●'}</div>
        <p style="font-size:9px;color:${i<5?C.grn:C.pri};font-weight:600;margin-top:5px;text-align:center;white-space:nowrap">${s}</p>
      </div>
      ${i<steps.length-1?`<div style="flex:1;height:2px;background:${i<4?C.grn:'#E5E7EB'};margin:0 4px;margin-bottom:18px"></div>`:''}`
    ).join('')}
  </div>`;
}

async function loadHarmStats(name, idx) {
  // update tab styles
  document.querySelectorAll('.harm-tab').forEach((t,i)=>{
    const active = i===idx;
    t.style.borderColor   = active?C.pri:'#E5E7EB';
    t.style.background    = active?C.priL:'#fff';
    t.style.color         = active?C.pri:'#6B7280';
  });
  const hc = document.getElementById('harm-content');
  hc.innerHTML = loader();
  try {
    const s = await api(`/api/harmonisation/${encodeURIComponent(name)}`);
    hc.innerHTML = `<div style="display:grid;grid-template-columns:1.6fr 1fr;gap:16px">
      ${card(`
        ${sTitle('Mapping Table','How each column was mapped to a canonical CBC field')}
        <div style="max-height:420px;overflow-y:auto">
          <table>
            <thead style="position:sticky;top:0;background:#fff">
              <tr><th class="th">Original Column</th><th class="th">Maps To</th><th class="th">Match</th><th class="th">Confidence</th></tr>
            </thead>
            <tbody>
              ${(s.mapping_detail||[]).map(r=>`
                <tr style="background:${r.match==='Dropped'?'#FAFAFA':'transparent'}">
                  <td class="td"><span class="mono" style="font-size:11px;color:${r.match==='Dropped'?'#9CA3AF':'#111827'}">${esc(r.original)}</span></td>
                  <td class="td"><span class="mono" style="font-size:11px;font-weight:700;color:${r.match==='Dropped'?'#9CA3AF':C.pri}">${esc(r.mapped)}</span></td>
                  <td class="td">${badge(r.match, r.match==='Exact'?'act':r.match==='Fuzzy'?'inc':'neu')}</td>
                  <td class="td">
                    ${r.confidence>0?`
                      <div style="display:flex;align-items:center;gap:5px">
                        <div style="width:42px;height:4px;background:#F3F4F6;border-radius:2px;overflow:hidden">
                          <div style="width:${r.confidence}%;height:100%;background:${r.confidence>=90?C.grn:C.amb};border-radius:2px"></div>
                        </div>
                        <span class="mono" style="font-size:10px;color:#6B7280">${r.confidence}%</span>
                      </div>`:'<span style="color:#9CA3AF;font-size:11px">—</span>'}
                  </td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>`)}
      <div style="display:flex;flex-direction:column;gap:14px">
        ${card(`
          ${sTitle('Data Confidence')}
          <div style="display:flex;align-items:center;gap:18px">
            ${confidenceRing(s.confidence)}
            <div>
              <p style="font-size:13px;font-weight:600">${s.confidence>=80?'High Confidence':'Medium Confidence'}</p>
              <p style="font-size:12px;color:#6B7280;margin-top:2px">${s.matched}/${s.total} features matched</p>
              <p style="font-size:12px;color:#6B7280">${fmt(s.rows)} rows processed</p>
            </div>
          </div>`)}
        ${card(`
          ${sTitle('Cleaning Report')}
          ${[
            ['Outliers Cleaned',  fmt(s.outliers_cleaned||0), C.amb],
            ['Nulls Filled',      fmt(s.nulls_filled||0),     C.pri],
            ['Scale Conversions', (s.scale_conversions||[]).length, '#7C3AED'],
            ['Duplicate Rows',    s.duplicates||0,            '#6B7280'],
            ['Columns Dropped',   s.dropped||0,               '#6B7280'],
          ].map(([l,v,c])=>`
            <div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #E5E7EB">
              <span style="font-size:12px">${l}</span>
              <span class="mono" style="font-size:13px;font-weight:700;color:${c}">${v}</span>
            </div>`).join('')}
          ${(s.scale_conversions||[]).length?`
            <div style="margin-top:10px;padding:10px;background:#F8F9FF;border-radius:6px;border:1px solid ${C.priBrd}">
              ${s.scale_conversions.map(c=>`<p style="font-size:11px;color:${C.pri};margin-bottom:2px">↳ ${esc(c)}</p>`).join('')}
            </div>`:''}`)}
      </div>
    </div>`;
  } catch(e) { hc.innerHTML = `<p style="color:red">Failed: ${e.message}</p>`; }
}

// ════════════════════════════════════════════════════════════
// PAGE: MODEL TRAINING
// ════════════════════════════════════════════════════════════
async function renderTraining() {
  const mc = document.getElementById('main-content');
  let statusData;
  try { statusData = await api('/api/status'); } catch(e) { statusData={models_loaded:[]}; }
  const models   = statusData.models_loaded||[];
  const trainable = STATE.datasets.filter(d=>d.has_diagnosis);

  mc.innerHTML = `<div class="fade">
    <h1 style="font-size:23px;font-weight:800;margin-bottom:4px">Model Training</h1>
    <p style="font-size:13px;color:#6B7280;margin-bottom:22px">Train new models · saved alongside existing ones · real training on your labelled datasets</p>

    <div style="display:flex;gap:2px;margin-bottom:20px;background:#F1F5F9;padding:4px;border-radius:10px;width:fit-content">
      <button id="tab-saved" onclick="switchTrainTab('saved')"
        style="padding:8px 20px;border-radius:8px;border:none;background:#fff;color:#111827;font-size:13px;font-weight:600;box-shadow:0 1px 3px rgba(0,0,0,.09)">
        Saved Models
      </button>
      <button id="tab-train" onclick="switchTrainTab('train')"
        style="padding:8px 20px;border-radius:8px;border:none;background:transparent;color:#6B7280;font-size:13px;font-weight:400">
        Train New Model
      </button>
    </div>

    <div id="train-tab-saved">${renderSavedModels(models)}</div>
    <div id="train-tab-train" style="display:none">${renderTrainForm(trainable)}</div>
  </div>`;
}

function switchTrainTab(tab) {
  document.getElementById('train-tab-saved').style.display  = tab==='saved'?'block':'none';
  document.getElementById('train-tab-train').style.display  = tab==='train'?'block':'none';
  document.getElementById('tab-saved').style.background     = tab==='saved'?'#fff':'transparent';
  document.getElementById('tab-saved').style.color          = tab==='saved'?'#111827':'#6B7280';
  document.getElementById('tab-saved').style.fontWeight     = tab==='saved'?'600':'400';
  document.getElementById('tab-saved').style.boxShadow      = tab==='saved'?'0 1px 3px rgba(0,0,0,.09)':'none';
  document.getElementById('tab-train').style.background     = tab==='train'?'#fff':'transparent';
  document.getElementById('tab-train').style.color          = tab==='train'?'#111827':'#6B7280';
  document.getElementById('tab-train').style.fontWeight     = tab==='train'?'600':'400';
  document.getElementById('tab-train').style.boxShadow      = tab==='train'?'0 1px 3px rgba(0,0,0,.09)':'none';
}

function renderSavedModels(models) {
  if(!models.length) return empty('◈','No models loaded','Upload pkl files or train a new model');
  const cards = models.map((m,i)=>{
    const col=C.MODEL_COLORS[i%4], met=m.metrics||{};
    return `<div class="card">
      <div style="display:flex;justify-content:space-between;margin-bottom:14px">
        <div>
          <p style="font-size:15px;font-weight:700">${esc(m.display_name||m.key)}</p>
          <p style="font-size:11px;color:#6B7280;margin-top:2px">${esc(m.algorithm||m.key)}</p>
        </div>
        <div style="width:10px;height:10px;border-radius:50%;background:${C.grn};margin-top:4px"></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">
        ${[['Accuracy',met.accuracy!=null?`${met.accuracy}%`:'—'],
           ['ROC-AUC', met.roc_auc||'—'],
           ['F1 Score',met.f1!=null?`${met.f1}%`:'—'],
           ['Precision',met.precision!=null?`${met.precision}%`:'—']].map(([k,v])=>`
          <div style="background:#F9FAFB;border-radius:6px;padding:8px 10px">
            <p style="font-size:9px;color:#6B7280;font-weight:700;text-transform:uppercase">${k}</p>
            <p class="mono" style="font-size:14px;font-weight:700;margin-top:2px">${v}</p>
          </div>`).join('')}
      </div>
      <div style="display:flex;gap:6px">
        <div style="flex:1;padding:6px 10px;background:${C.grnL};border:1px solid ${C.grn}33;border-radius:8px;text-align:center;font-size:12px;font-weight:600;color:${C.grn}">✓ Loaded</div>
        <button class="btn btn-danger btn-sm" onclick="deleteModel('${esc(m.key)}')">Delete</button>
      </div>
    </div>`;
  }).join('');

  const comparison = `${card(`
    ${sTitle('Model Comparison','All loaded models head-to-head')}
    <table>
      <thead><tr>
        <th class="th">Model</th><th class="th">Algorithm</th>
        <th class="th">Accuracy</th><th class="th">ROC-AUC</th>
        <th class="th">F1</th><th class="th">Precision</th><th class="th">Recall</th>
      </tr></thead>
      <tbody>
        ${models.map((m,i)=>{
          const col=C.MODEL_COLORS[i%4], met=m.metrics||{};
          return `<tr style="background:${i%2===0?'#FAFAFA':'transparent'}">
            <td class="td">
              <div style="display:flex;align-items:center;gap:8px">
                <div style="width:9px;height:9px;border-radius:2px;background:${col}"></div>
                <span style="font-weight:600">${esc(m.display_name||m.key)}</span>
              </div>
            </td>
            <td class="td" style="color:#6B7280;font-size:12px">${esc(m.algorithm||m.key)}</td>
            ${[met.accuracy,met.roc_auc,met.f1,met.precision,met.recall].map((v,j)=>`
              <td class="td"><span class="mono" style="font-weight:600;color:${v!=null?C.grn:C.lite}">
                ${v!=null?`${v}${j===1?'':'%'}`:'—'}
              </span></td>`).join('')}
          </tr>`;
        }).join('')}
      </tbody>
    </table>`)}`;

  return `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:20px">${cards}</div>${comparison}`;
}

function renderTrainForm(trainable) {
  const dsOpts = trainable.length
    ? trainable.map(d=>`<option value="${esc(d.name)}">${esc(d.name.replace(/_/g,' '))}</option>`).join('')
    : `<option value="">— No labelled datasets —</option>`;
  return `<div style="display:grid;grid-template-columns:1fr 1.2fr;gap:20px">
    ${card(`
      ${sTitle('Training Setup','Configure parameters')}
      ${!trainable.length?`<div style="padding:14px;background:${C.ambL};border:1px solid #FDE68A;border-radius:8px;margin-bottom:16px;font-size:13px;color:${C.amb}">
        ⚠ No training-ready datasets. Upload a dataset with a diagnosis/label column first.</div>`:''}
      <div style="display:flex;flex-direction:column;gap:14px">
        ${formField('Training Dataset','select',dsOpts,'train-ds')}
        ${formField('Algorithm','select',
          ['XGBoost','Random Forest','Neural Network'].map(a=>`<option>${a}</option>`).join(''),
          'train-algo', 'onchange="updateTrainForm()"')}
        ${formField('Model Name (saved with this name)','input','','train-name','placeholder="e.g. XGBoost_v2"')}
        ${formField('Class Balance','select',
          ['SMOTE','Class Weight','None'].map(b=>`<option>${b}</option>`).join(''),
          'train-balance')}
        <div id="tree-params" style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          ${formField('N Estimators','input','','train-nest','type="number" value="200"')}
          ${formField('Max Depth','input','','train-depth','type="number" value="6"')}
        </div>
        <div id="lr-param">
          ${formField('Learning Rate','input','','train-lr','type="number" step="0.01" value="0.1"')}
        </div>
        <button class="btn btn-primary btn-lg" id="start-train-btn" onclick="startTraining()"
          style="margin-top:4px" ${!trainable.length?'disabled':''}>
          ◈  Start Training
        </button>
      </div>`)}
    ${card(`
      ${sTitle('Training Console','Live output')}
      <div class="console-log" id="train-console">
        <p style="color:#475569">Configure parameters and click Start Training.</p>
      </div>`)}
  </div>`;
}

function formField(label, type, inner, id, extra='') {
  const input = type==='select'
    ? `<select id="${id}" class="input-field" ${extra}>${inner}</select>`
    : `<input id="${id}" class="input-field" ${extra}>`;
  return `<div>
    <label style="font-size:11px;font-weight:700;color:#6B7280;display:block;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em">${label}</label>
    ${input}
  </div>`;
}

function updateTrainForm() {
  const algo = document.getElementById('train-algo')?.value;
  if(!algo) return;
  const tree = document.getElementById('tree-params');
  const lr   = document.getElementById('lr-param');
  if(tree) tree.style.display = algo==='Neural Network'?'none':'grid';
  if(lr)   lr.style.display   = algo==='XGBoost'?'block':'none';
  // suggest model name
  const names={'XGBoost':'XGBoost_v2','Random Forest':'RandomForest_v2','Neural Network':'NeuralNet_v2'};
  const nameEl = document.getElementById('train-name');
  if(nameEl && !nameEl.value) nameEl.value = names[algo]||'';
}

async function startTraining() {
  const dsName  = document.getElementById('train-ds')?.value;
  const algo    = document.getElementById('train-algo')?.value;
  const name    = document.getElementById('train-name')?.value?.trim();
  const balance = document.getElementById('train-balance')?.value;
  const nEst    = +(document.getElementById('train-nest')?.value||200);
  const maxD    = +(document.getElementById('train-depth')?.value||6);
  const lr      = +(document.getElementById('train-lr')?.value||0.1);

  if(!dsName) { showToast('Select a training dataset','error'); return; }
  if(!name)   { showToast('Enter a model name','error'); return; }

  const jobId = `job_${Date.now()}`;
  const btn   = document.getElementById('start-train-btn');
  if(btn) { btn.disabled=true; btn.innerHTML='<span class="spin">◈</span>  Training...'; }

  const con = document.getElementById('train-console');
  if(con) con.innerHTML = '';

  try {
    await api('/api/train', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({dataset_name:dsName, algorithm:algo, model_name:name,
        balance_method:balance, n_estimators:nEst, max_depth:maxD,
        learning_rate:lr, job_id:jobId}),
    });
  } catch(e) { showToast(`✗ ${e.message}`,'error'); if(btn){btn.disabled=false;btn.innerHTML='◈  Start Training';} return; }

  // Poll
  if(STATE.trainingInterval) clearInterval(STATE.trainingInterval);
  STATE.trainingInterval = setInterval(async()=>{
    try {
      const job = await api(`/api/train/status/${jobId}`);
      const con = document.getElementById('train-console');
      if(con) {
        con.innerHTML = job.log.map(line=>`
          <p style="font-size:11px;margin-bottom:4px;line-height:1.5;color:${
            line.startsWith('✓')?'#34D399':line.startsWith('✗')?'#F87171':'#94A3B8'}">${esc(line)}</p>
        `).join('');
        con.scrollTop = con.scrollHeight;
      }
      if(job.status==='complete'||job.status==='error') {
        clearInterval(STATE.trainingInterval);
        if(btn){btn.disabled=false;btn.innerHTML='◈  Start Training';}
        if(job.status==='complete') {
          showToast(`✓ "${job.model_key}" trained and saved`);
          await refreshState();
          // refresh saved models tab
          const statusData = await api('/api/status');
          document.getElementById('train-tab-saved').innerHTML = renderSavedModels(statusData.models_loaded||[]);
        } else {
          showToast('✗ Training failed — see console','error');
        }
      }
    } catch(e) {}
  }, 1000);
}

async function deleteModel(key) {
  try {
    await api(`/api/models/${encodeURIComponent(key)}`, {method:'DELETE'});
    showToast(`Deleted: ${key}`);
    await refreshState();
    const statusData = await api('/api/status');
    document.getElementById('train-tab-saved').innerHTML = renderSavedModels(statusData.models_loaded||[]);
  } catch(e) { showToast(`✗ ${e.message}`,'error'); }
}

// ════════════════════════════════════════════════════════════
// PAGE: ANALYSIS DASHBOARD
// ════════════════════════════════════════════════════════════
let DASH_STATE = {dsIdx:0, page:0, risk:'All', search:'', allModels:[]};
const PS = 20;

async function renderDashboard() {
  const mc = document.getElementById('main-content');
  const datasets = STATE.datasets;
  if(!datasets.length) {
    mc.innerHTML = `<div class="fade"><h1 style="font-size:23px;font-weight:800;margin-bottom:22px">Analysis Dashboard</h1>${empty('▦','No datasets loaded','Upload a dataset from Data Intake first')}</div>`;
    return;
  }
  DASH_STATE = {dsIdx:0, page:0, risk:'All', search:'', allModels:[]};
  mc.innerHTML = `<div class="fade">
    <h1 style="font-size:23px;font-weight:800;margin-bottom:4px">Analysis Dashboard</h1>
    <p style="font-size:13px;color:#6B7280;margin-bottom:20px">Prediction results and patient-level analysis</p>
    <div id="dash-kpis" style="display:flex;gap:12px;margin-bottom:18px">${loader()}</div>
    <div style="display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap">
      ${datasets.map((d,i)=>`
        <button onclick="dashSwitchDs(${i})" id="dash-tab-${i}"
          style="padding:9px 18px;border-radius:8px;border:1px solid ${i===0?C.pri:'#E5E7EB'};
            background:${i===0?C.priL:'#fff'};color:${i===0?C.pri:'#6B7280'};font-size:12px;font-weight:600">
          ${esc(d.name.replace(/_/g,' ').split(' ').slice(0,3).join(' '))}
        </button>`).join('')}
    </div>
    <div id="dash-charts" style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px">${loader()}</div>
    <div id="dash-table">${loader()}</div>
  </div>`;
  await loadDashData();
}

async function dashSwitchDs(i) {
  DASH_STATE.dsIdx=i; DASH_STATE.page=0;
  STATE.datasets.forEach((_,j)=>{
    const t=document.getElementById(`dash-tab-${j}`);
    if(!t)return;
    const active=j===i;
    t.style.borderColor=active?C.pri:'#E5E7EB';
    t.style.background=active?C.priL:'#fff';
    t.style.color=active?C.pri:'#6B7280';
  });
  document.getElementById('dash-kpis').innerHTML=loader();
  document.getElementById('dash-charts').innerHTML=loader();
  document.getElementById('dash-table').innerHTML=loader();
  await loadDashData();
}

async function loadDashData() {
  const ds = STATE.datasets[DASH_STATE.dsIdx];
  if(!ds) return;
  try {
    const data = await api(`/api/predictions/${encodeURIComponent(ds.name)}?page=${DASH_STATE.page}&page_size=${PS}&risk_filter=${encodeURIComponent(DASH_STATE.risk)}&search=${encodeURIComponent(DASH_STATE.search)}`);
    DASH_STATE.allModels = Object.keys(data.model_stats||{});

    // KPIs
    const mk0 = DASH_STATE.allModels[0];
    const s0  = data.model_stats?.[mk0]||{};
    document.getElementById('dash-kpis').innerHTML = !data.predicted
      ? `<div style="padding:16px;color:#D97706;font-weight:600;font-size:13px">⚠ Predictions not run yet for this dataset. Go to Data Intake → Run Predictions.</div>`
      : `${kpiCard('Total Analysed', fmt(ds.rows), 'This dataset', '📊', C.pri)}
         ${kpiCard('Leukemia Positive', fmt(s0.positive), `${s0.positive_rate||0}%`, '🔴', C.red)}
         ${kpiCard('Inconclusive', fmt(s0.inconclusive), 'Further testing', '⚠️', C.amb)}
         ${kpiCard('Healthy', fmt(s0.negative), `${(100-(s0.positive_rate||0)).toFixed(1)}%`, '✓', C.grn)}`;

    if(!data.predicted) {
      document.getElementById('dash-charts').innerHTML='';
      document.getElementById('dash-table').innerHTML='';
      return;
    }

    // Charts
    const allMK = DASH_STATE.allModels;
    document.getElementById('dash-charts').innerHTML = `
      ${card(`${sTitle('Risk Distribution','Risk levels across all loaded models')}<canvas id="stack-canvas" width="460" height="200"></canvas>`)}
      ${card(`${sTitle('Prediction Summary','Per model breakdown')}
        <div style="display:flex;flex-direction:column;gap:10px;margin-top:6px">
          ${allMK.slice(0,4).map((mk,i)=>{
            const s=data.model_stats[mk]||{};
            const col=C.MODEL_COLORS[i%4];
            return `<div style="padding:10px 12px;background:#F9FAFB;border-radius:8px;border:1px solid #E5E7EB">
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="font-size:12px;font-weight:600">${esc(s.display_name||mk)}</span>
                <span class="mono" style="font-size:12px;color:${col};font-weight:700">${s.positive_rate||0}% positive</span>
              </div>
              ${progressBar(s.positive_rate||0, col)}
              <div style="display:flex;gap:10px;margin-top:5px">
                <span style="font-size:10px;color:${C.red}">● ${fmt(s.high_risk)} high risk</span>
                <span style="font-size:10px;color:${C.amb}">● ${fmt(s.inconclusive)} inconclusive</span>
              </div>
            </div>`;
          }).join('')}
        </div>`)}`;

    // draw stacked bar
    const stackData = allMK.map(mk=>{
      const s=data.model_stats[mk]||{};
      return {name:s.display_name||mk, data:[s.high_risk||0,s.inconclusive||0,s.low_risk||0]};
    });
    setTimeout(()=>{
      drawStackedBar('stack-canvas',
        stackData.map(d=>d.name),
        [
          {label:'High Risk',  color:C.red, data:stackData.map(d=>d.data[0])},
          {label:'Inconclusive',color:C.amb, data:stackData.map(d=>d.data[1])},
          {label:'Low Risk',   color:C.grn, data:stackData.map(d=>d.data[2])},
        ]
      );
    },50);

    // Table
    renderDashTable(data);
    DASH_STATE._data = data;
  } catch(e) { console.error(e); }
}

function renderDashTable(data) {
  const allMK = DASH_STATE.allModels;
  const tc = document.getElementById('dash-table');
  if(!tc) return;
  tc.innerHTML = card(`
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
      <div>
        <p class="section-title">Patient Records</p>
        <p class="section-sub" style="margin-bottom:0">${fmt(data.total_rows)} records</p>
      </div>
      <div style="display:flex;gap:8px">
        <input id="dash-search" class="input-field" placeholder="Search row #..." style="width:140px"
          value="${esc(DASH_STATE.search)}" onchange="dashFilter()">
        <select id="dash-risk" class="input-field" style="width:auto" onchange="dashFilter()">
          ${['All','High Risk','Inconclusive','Low Risk'].map(o=>`<option ${DASH_STATE.risk===o?'selected':''}>${o}</option>`).join('')}
        </select>
      </div>
    </div>
    <div style="overflow-x:auto">
      <table style="min-width:860px">
        <thead><tr>
          <th class="th">#</th><th class="th">WBC</th><th class="th">RBC</th>
          <th class="th">HGB</th><th class="th">PLT</th>
          ${allMK.map(mk=>`<th class="th">${esc((data.model_stats[mk]?.display_name||mk).split(' ')[0])}</th>`).join('')}
          <th class="th">Consensus</th><th class="th">Risk</th><th class="th">Prob%</th>
        </tr></thead>
        <tbody>
          ${data.rows.map(p=>`
            <tr onclick="openPatientDrawer(${JSON.stringify(JSON.stringify(p))})" style="cursor:pointer;background:${riskBg(p.risk_level)}22">
              <td class="td"><span class="mono" style="font-size:11px;color:#6B7280">${p.row_index}</span></td>
              <td class="td"><span class="mono" style="font-size:11px">${fmt(p.WBC)}</span></td>
              <td class="td"><span class="mono" style="font-size:11px">${p.RBC||'—'}</span></td>
              <td class="td"><span class="mono" style="font-size:11px">${p.Hemoglobin||'—'}</span></td>
              <td class="td"><span class="mono" style="font-size:11px">${fmt(p.Platelets)}</span></td>
              ${allMK.map(mk=>`<td class="td">${badge(p[mk+'_prediction']||'—', riskClass(p[mk+'_prediction']))}</td>`).join('')}
              <td class="td">${badge(p.consensus, riskClass(p.consensus))}</td>
              <td class="td">${badge(p.risk_level, riskClass(p.risk_level))}</td>
              <td class="td"><span class="mono" style="font-size:11px;font-weight:600;color:${riskColor(p.risk_level)}">${fmtF(p.max_probability)}</span></td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:14px;padding-top:14px;border-top:1px solid #E5E7EB">
      <span style="font-size:12px;color:#6B7280">Page ${DASH_STATE.page+1} · ${fmt(data.total_rows)} total</span>
      <div style="display:flex;gap:6px">
        <button class="btn btn-ghost btn-sm" onclick="dashPage(-1)" ${DASH_STATE.page===0?'disabled':''}>← Prev</button>
        <button class="btn btn-ghost btn-sm" onclick="dashPage(1)"  ${(DASH_STATE.page+1)*PS>=data.total_rows?'disabled':''}>Next →</button>
      </div>
    </div>`);
}

function dashFilter() {
  DASH_STATE.search = document.getElementById('dash-search')?.value||'';
  DASH_STATE.risk   = document.getElementById('dash-risk')?.value||'All';
  DASH_STATE.page   = 0;
  loadDashData();
}
function dashPage(dir) { DASH_STATE.page = Math.max(0, DASH_STATE.page+dir); loadDashData(); }

function openPatientDrawer(jsonStr) {
  const p = JSON.parse(jsonStr);
  const allMK = DASH_STATE.allModels;
  const modelStats = DASH_STATE._data?.model_stats||{};
  openDrawer(`
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px">
      <div>
        <p class="mono" style="font-size:17px;font-weight:800">Row #${p.row_index}</p>
        <div style="margin-top:6px">${badge(p.risk_level, riskClass(p.risk_level),'md')}</div>
      </div>
      <button onclick="closeDrawer()" style="background:none;border:none;font-size:19px;color:#6B7280">✕</button>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:18px">
      ${[['WBC',fmt(p.WBC),'cells/µL'],['RBC',p.RBC,'mill/µL'],
         ['Hemoglobin',p.Hemoglobin,'g/dL'],['Hematocrit',p.Hematocrit,'%'],
         ['Platelets',fmt(p.Platelets),'cells/µL'],['Age',p.Age,'years']
        ].map(([k,v,u])=>`
        <div style="background:#F9FAFB;border-radius:8px;padding:10px 12px">
          <p style="font-size:9px;color:#6B7280;font-weight:700;text-transform:uppercase">${k}</p>
          <p class="mono" style="font-size:14px;font-weight:700;margin-top:2px">${v||'—'}</p>
          <p style="font-size:9px;color:#9CA3AF">${u}</p>
        </div>`).join('')}
    </div>
    <p style="font-size:10px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">Model Predictions</p>
    ${allMK.map(mk=>`
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #E5E7EB">
        <span style="font-size:13px;font-weight:600">${esc(modelStats[mk]?.display_name||mk)}</span>
        <div style="display:flex;align-items:center;gap:8px">
          <span class="mono" style="font-size:12px;color:#6B7280">${fmtF(p[mk+'_probability'])}%</span>
          ${badge(p[mk+'_prediction']||'—', riskClass(p[mk+'_prediction']))}
        </div>
      </div>`).join('')}
    <div style="margin-top:16px;padding:14px;background:${riskBg(p.risk_level)};border-radius:8px">
      <p style="font-size:12px;font-weight:700;color:${riskColor(p.risk_level)}">Risk Assessment</p>
      <p style="font-size:12px;color:#6B7280;margin-top:4px;line-height:1.55">
        ${p.risk_level==='High Risk'
          ?'CBC pattern strongly indicates leukemia. Immediate clinical review recommended.'
          :p.risk_level==='Inconclusive'
            ?'Borderline CBC values detected. Models disagree. Further diagnostic testing recommended.'
            :'CBC values within normal clinical ranges. No leukemia indicators detected.'}
      </p>
    </div>`);
}

// ════════════════════════════════════════════════════════════
// PAGE: PATIENT RECORDS
// ════════════════════════════════════════════════════════════
let PAT_STATE = {dsIdx:0, risk:'All', allModels:[], _data:null};

async function renderPatients() {
  const mc = document.getElementById('main-content');
  const datasets = STATE.datasets;
  if(!datasets.length) {
    mc.innerHTML=`<div class="fade"><h1 style="font-size:23px;font-weight:800;margin-bottom:22px">Patient Records</h1>${empty('♦','No datasets loaded','Upload a dataset from Data Intake first')}</div>`;
    return;
  }
  PAT_STATE={dsIdx:0,risk:'All',allModels:[],_data:null};
  mc.innerHTML=`<div class="fade">
    <h1 style="font-size:23px;font-weight:800;margin-bottom:4px">Patient Records</h1>
    <p style="font-size:13px;color:#6B7280;margin-bottom:20px">Browse predictions — click any card for full detail</p>
    ${card(`<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      ${datasets.map((d,i)=>`
        <button onclick="patSwitchDs(${i})" id="pat-tab-${i}"
          style="padding:8px 16px;border-radius:8px;border:1px solid ${i===0?C.pri:'#E5E7EB'};
            background:${i===0?C.priL:'#fff'};color:${i===0?C.pri:'#6B7280'};font-size:12px;font-weight:600">
          ${esc(d.name.replace(/_/g,' ').split(' ').slice(0,3).join(' '))}
        </button>`).join('')}
      <select id="pat-risk" class="input-field" style="width:auto;margin-left:auto" onchange="patFilter()">
        ${['All','High Risk','Inconclusive','Low Risk'].map(o=>`<option>${o}</option>`).join('')}
      </select>
      <span id="pat-count" style="font-size:13px;color:#6B7280">—</span>
    </div>`,'margin-bottom:18px')}
    <div id="pat-cards">${loader()}</div>
    <div id="pat-disagree" style="margin-top:22px"></div>
  </div>`;
  await loadPatData();
}

async function patSwitchDs(i) {
  PAT_STATE.dsIdx=i; PAT_STATE.risk='All';
  document.getElementById('pat-risk').value='All';
  STATE.datasets.forEach((_,j)=>{
    const t=document.getElementById(`pat-tab-${j}`);
    if(!t)return;
    const a=j===i;
    t.style.borderColor=a?C.pri:'#E5E7EB';
    t.style.background=a?C.priL:'#fff';
    t.style.color=a?C.pri:'#6B7280';
  });
  document.getElementById('pat-cards').innerHTML=loader();
  document.getElementById('pat-disagree').innerHTML='';
  await loadPatData();
}
function patFilter() {
  PAT_STATE.risk=document.getElementById('pat-risk')?.value||'All';
  loadPatData();
}

async function loadPatData() {
  const ds=STATE.datasets[PAT_STATE.dsIdx];
  if(!ds)return;
  try {
    const data=await api(`/api/predictions/${encodeURIComponent(ds.name)}?page=0&page_size=60&risk_filter=${encodeURIComponent(PAT_STATE.risk)}`);
    PAT_STATE.allModels=Object.keys(data.model_stats||{});
    PAT_STATE._data=data;
    const cnt=document.getElementById('pat-count');
    if(cnt) cnt.textContent=`${fmt(data.total_rows)} records`;

    const pc=document.getElementById('pat-cards');
    if(!data.predicted) {
      pc.innerHTML=empty('▶','Predictions not run yet','Go to Data Intake → Run Predictions on this dataset');
      return;
    }

    pc.innerHTML=`<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
      ${data.rows.slice(0,30).map(p=>`
        <div onclick="openPatDrawer(${JSON.stringify(JSON.stringify(p))})"
          style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;cursor:pointer;
            border-top:3px solid ${riskColor(p.risk_level)};box-shadow:0 1px 3px rgba(0,0,0,.05)">
          <div style="padding:13px 15px">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px">
              <span class="mono" style="font-size:11px;font-weight:700">Row #${p.row_index}</span>
              ${badge(p.consensus, riskClass(p.consensus))}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:9px">
              ${[['WBC',fmt(p.WBC)],['HGB',p.Hemoglobin]].map(([k,v])=>`
                <div style="background:#F9FAFB;border-radius:6px;padding:5px 8px">
                  <p style="font-size:9px;color:#6B7280;font-weight:700">${k}</p>
                  <p class="mono" style="font-size:12px;font-weight:700">${v||'—'}</p>
                </div>`).join('')}
            </div>
            <div style="display:flex;justify-content:space-between;padding-top:8px;border-top:1px solid #E5E7EB">
              <span style="font-size:11px;color:#6B7280">Risk: <b style="color:${riskColor(p.risk_level)}">${p.risk_level}</b></span>
              <span class="mono" style="font-size:11px;color:#6B7280">${fmtF(p.max_probability)}%</span>
            </div>
          </div>
        </div>`).join('')}
    </div>`;

    // Disagreements
    const disagree=data.rows.filter(p=>{
      const ps=PAT_STATE.allModels.map(mk=>p[mk+'_prediction']).filter(Boolean);
      return ps.length>1&&!ps.every(v=>v===ps[0]);
    });
    const pd=document.getElementById('pat-disagree');
    if(disagree.length&&pd) {
      pd.innerHTML=card(`
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
          <div style="width:8px;height:8px;border-radius:50%;background:${C.amb}"></div>
          <div>
            <p class="section-title">Model Disagreements — Clinical Review Required</p>
            <p class="section-sub" style="margin-bottom:0">${disagree.length} cases where models gave different predictions</p>
          </div>
        </div>
        <table>
          <thead><tr>
            <th class="th">Row #</th>
            ${PAT_STATE.allModels.map(mk=>`<th class="th">${esc((data.model_stats[mk]?.display_name||mk).split(' ')[0])}</th>`).join('')}
            <th class="th">Action</th>
          </tr></thead>
          <tbody>
            ${disagree.slice(0,10).map(p=>`
              <tr style="background:${C.ambL};cursor:pointer" onclick="openPatDrawer(${JSON.stringify(JSON.stringify(p))})">
                <td class="td"><span class="mono" style="font-weight:600;color:${C.pri}">#${p.row_index}</span></td>
                ${PAT_STATE.allModels.map(mk=>`<td class="td">${badge(p[mk+'_prediction']||'—',riskClass(p[mk+'_prediction']))}</td>`).join('')}
                <td class="td"><span style="font-size:12px;color:${C.amb};font-weight:600">⚠ Review Required</span></td>
              </tr>`).join('')}
          </tbody>
        </table>`);
    }
  } catch(e) { console.error(e); }
}

function openPatDrawer(jsonStr) {
  const p=JSON.parse(jsonStr);
  const allMK=PAT_STATE.allModels;
  const modelStats=PAT_STATE._data?.model_stats||{};
  openDrawer(`
    <div style="display:flex;justify-content:space-between;margin-bottom:18px">
      <p class="mono" style="font-size:17px;font-weight:800">Row #${p.row_index}</p>
      <button onclick="closeDrawer()" style="background:none;border:none;font-size:18px;color:#6B7280">✕</button>
    </div>
    <div style="margin-bottom:14px">${badge(p.risk_level,riskClass(p.risk_level),'md')}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px">
      ${[['WBC',fmt(p.WBC)],['RBC',p.RBC],['HGB',p.Hemoglobin],['PLT',fmt(p.Platelets)]].map(([k,v])=>`
        <div style="background:#F9FAFB;border-radius:8px;padding:10px 12px">
          <p style="font-size:9px;color:#6B7280;font-weight:700">${k}</p>
          <p class="mono" style="font-size:14px;font-weight:700">${v||'—'}</p>
        </div>`).join('')}
    </div>
    ${allMK.map(mk=>`
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #E5E7EB">
        <span style="font-size:13px;font-weight:600">${esc(modelStats[mk]?.display_name||mk)}</span>
        <div style="display:flex;align-items:center;gap:8px">
          <span class="mono" style="font-size:12px;color:#6B7280">${fmtF(p[mk+'_probability'])}%</span>
          ${badge(p[mk+'_prediction']||'—',riskClass(p[mk+'_prediction']))}
        </div>
      </div>`).join('')}
    <div style="margin-top:16px;padding:14px;background:${riskBg(p.risk_level)};border-radius:8px">
      <p style="font-size:12px;font-weight:700;color:${riskColor(p.risk_level)}">
        ${p.risk_level==='High Risk'?'⚠ Immediate clinical review recommended'
          :p.risk_level==='Inconclusive'?'⚠ Further diagnostic testing recommended'
          :'✓ No leukemia indicators detected'}
      </p>
    </div>`);
}

// ════════════════════════════════════════════════════════════
// PAGE: REPORTS
// ════════════════════════════════════════════════════════════
let REPORT_SECTIONS={exec:true,method:true,patients:true,stats:true};

async function renderReports() {
  const mc=document.getElementById('main-content');
  const ov=STATE.overview||{};
  const datasets=STATE.datasets;

  mc.innerHTML=`<div class="fade">
    <h1 style="font-size:23px;font-weight:800;margin-bottom:4px">Reports</h1>
    <p style="font-size:13px;color:#6B7280;margin-bottom:22px">Generate and download clinical PDF reports</p>
    <div style="display:grid;grid-template-columns:1fr 1.4fr;gap:20px;margin-bottom:22px">
      ${card(`
        ${sTitle('Report Builder','Configure your report')}
        <div style="display:flex;flex-direction:column;gap:14px">
          <div>
            <label style="font-size:11px;font-weight:700;color:#6B7280;display:block;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em">Dataset</label>
            <select id="rep-ds" class="input-field">
              <option value="All Datasets">All Datasets</option>
              ${datasets.map(d=>`<option value="${esc(d.name)}">${esc(d.name.replace(/_/g,' '))}</option>`).join('')}
            </select>
          </div>
          <div>
            <label style="font-size:11px;font-weight:700;color:#6B7280;display:block;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em">Risk Filter</label>
            <select id="rep-risk" class="input-field">
              ${['All Records','High Risk Only','High + Inconclusive'].map(o=>`<option>${o}</option>`).join('')}
            </select>
          </div>
          <div>
            <p style="font-size:11px;font-weight:700;color:#6B7280;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">Include Sections</p>
            ${[['exec','Executive Summary'],['method','Methodology'],['patients','High Risk Patient Detail'],['stats','Statistical Analysis']].map(([k,lbl])=>`
              <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #E5E7EB">
                <span style="font-size:13px">${lbl}</span>
                <button class="toggle-track" id="tog-${k}"
                  style="background:${REPORT_SECTIONS[k]?C.pri:'#D1D5DB'}"
                  onclick="toggleSection('${k}')">
                  <span class="toggle-thumb" style="left:${REPORT_SECTIONS[k]?'19px':'2px'}"></span>
                </button>
              </div>`).join('')}
          </div>
          <button class="btn btn-primary btn-lg" id="gen-report-btn"
            onclick="generateReport()" style="margin-top:4px"
            ${!datasets.length?'disabled':''}>
            ▤  Generate &amp; Download PDF
          </button>
          ${!datasets.length?`<p style="font-size:12px;color:#D97706;text-align:center">Upload and predict on a dataset first</p>`:''}
        </div>`)}
      <div id="report-preview">${renderReportPreview(ov,datasets)}</div>
    </div>
    <div id="report-history">${renderReportHistory()}</div>
  </div>`;
}

let REPORT_HISTORY = [];

function renderReportHistory() {
  if(!REPORT_HISTORY.length) return card(`
    ${sTitle('Report History','Downloaded reports this session')}
    <p style="text-align:center;padding:20px 0;font-size:13px;color:#9CA3AF">No reports generated yet this session.</p>`);
  return card(`
    ${sTitle('Report History','Downloaded reports this session')}
    <table>
      <thead><tr>
        <th class="th">Report Name</th><th class="th">Generated</th>
        <th class="th">Dataset</th><th class="th">Records</th><th class="th"></th>
      </tr></thead>
      <tbody>
        ${REPORT_HISTORY.map((r,i)=>`
          <tr style="background:${i%2===0?'#FAFAFA':'transparent'}">
            <td class="td" style="font-weight:600">${esc(r.name)}</td>
            <td class="td"><span class="mono" style="font-size:12px;color:#6B7280">${r.date}</span></td>
            <td class="td"><span style="font-size:12px;color:#6B7280">${esc(r.dataset)}</span></td>
            <td class="td"><span class="mono">${fmt(r.rows)}</span></td>
            <td class="td"><button class="btn btn-outline btn-sm" onclick="redownload(${i})">⬇ Re-download</button></td>
          </tr>`).join('')}
      </tbody>
    </table>`);
}

async function generateReport() {
  const dsName   = document.getElementById('rep-ds')?.value  || 'All Datasets';
  const riskFilter = document.getElementById('rep-risk')?.value || 'All Records';
  const btn = document.getElementById('gen-report-btn');
  if(btn){btn.disabled=true;btn.innerHTML='<span class="spin">◈</span>  Generating...';}

  try {
    const resp = await fetch('/api/report', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        dataset_name:  dsName,
        risk_filter:   riskFilter,
        sections:      {...REPORT_SECTIONS},
      }),
    });
    if(!resp.ok){
      const e=await resp.json().catch(()=>({detail:'PDF generation failed'}));
      throw new Error(e.detail||'PDF generation failed');
    }
    const blob  = await resp.blob();
    const url   = URL.createObjectURL(blob);
    const now   = new Date().toISOString().slice(0,16).replace('T',' ');
    const fname = `leukemia_report_${now.slice(0,10)}.pdf`;

    // Trigger browser download
    const a = document.createElement('a');
    a.href = url; a.download = fname;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    setTimeout(()=>URL.revokeObjectURL(url), 5000);

    // Add to history
    const ds = STATE.datasets.find(d=>d.name===dsName);
    REPORT_HISTORY.unshift({
      name: fname, date: now,
      dataset: dsName==='All Datasets'?'All Datasets':dsName.replace(/_/g,' '),
      rows: ds ? ds.rows : STATE.overview?.total_records||0,
      blob: blob,
    });
    document.getElementById('report-history').innerHTML = renderReportHistory();
    showToast('✓ PDF downloaded successfully');
  } catch(e){
    showToast(`✗ ${e.message}`,'error');
  }
  if(btn){btn.disabled=false;btn.innerHTML='▤  Generate &amp; Download PDF';}
}

async function redownload(idx) {
  const r = REPORT_HISTORY[idx];
  if(!r||!r.blob) return;
  const url = URL.createObjectURL(r.blob);
  const a   = document.createElement('a');
  a.href=url; a.download=r.name;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  setTimeout(()=>URL.revokeObjectURL(url),3000);
}

function renderReportPreview(ov, datasets) {
  return card(`
    ${sTitle('Report Preview','Live preview')}
    <div style="background:#F9FAFB;border-radius:8px;padding:20px;border:1px solid #E5E7EB">
      <div style="border-bottom:2px solid #1A56DB;padding-bottom:12px;margin-bottom:16px">
        <p style="font-size:15px;font-weight:800">Leukemia Harmoniser</p>
        <p style="font-size:12px;color:#6B7280">Clinical Analysis Report · ${new Date().toISOString().split('T')[0]}</p>
      </div>
      ${REPORT_SECTIONS.exec?`<div style="margin-bottom:14px">
        <p style="font-size:11px;font-weight:700;color:#1A56DB;margin-bottom:5px">EXECUTIVE SUMMARY</p>
        <p style="font-size:11px;color:#6B7280;line-height:1.6">Analysis of ${fmt(ov.total_records||0)} patient records across ${datasets.length} dataset(s) identified ${fmt(ov.total_positive||0)} leukemia-positive cases.</p>
      </div>`:''}
      ${REPORT_SECTIONS.method?`<div style="margin-bottom:14px">
        <p style="font-size:11px;font-weight:700;color:#1A56DB;margin-bottom:5px">METHODOLOGY</p>
        <p style="font-size:11px;color:#6B7280;line-height:1.6">Fuzzy column harmonisation applied. Outlier cleaning using clinical valid ranges. Scale auto-detection for 10⁹/L units.</p>
      </div>`:''}
      ${REPORT_SECTIONS.stats?`<div style="margin-bottom:10px">
        <p style="font-size:11px;font-weight:700;color:#1A56DB;margin-bottom:8px">STATISTICS</p>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
          ${[['Positive',ov.total_positive||0,C.red],['Inconclusive',ov.total_inconclusive||0,C.amb],['Negative',ov.total_negative||0,C.grn]].map(([k,v,c])=>`
            <div style="background:#fff;border-radius:6px;padding:8px;text-align:center;border:1px solid #E5E7EB">
              <p class="mono" style="font-size:15px;font-weight:800;color:${c}">${fmt(v)}</p>
              <p style="font-size:10px;color:#6B7280">${k}</p>
            </div>`).join('')}
        </div>
      </div>`:''}
      <p style="font-size:9px;color:#9CA3AF;text-align:center;margin-top:12px;padding-top:10px;border-top:1px solid #E5E7EB">CONFIDENTIAL — For clinical use only</p>
    </div>`);
}

function toggleSection(k) {
  REPORT_SECTIONS[k]=!REPORT_SECTIONS[k];
  const t=document.getElementById(`tog-${k}`);
  if(t){t.style.background=REPORT_SECTIONS[k]?C.pri:'#D1D5DB';}
  const thumb=document.querySelector(`#tog-${k} .toggle-thumb`);
  if(thumb) thumb.style.left=REPORT_SECTIONS[k]?'19px':'2px';
  document.getElementById('report-preview').innerHTML=renderReportPreview(STATE.overview||{}, STATE.datasets);
}

// ── BOOT ─────────────────────────────────────────────────────
async function boot() {
  renderSidebar();
  await refreshState();
  await navigate('overview');
  setInterval(refreshState, 15000);
}
boot();
</script>
</body>
</html>

"""



# ── REPORT GENERATION ─────────────────────────────────────────
class ReportRequest(BaseModel):
    dataset_name: str = "All Datasets"
    model_filter: str = "All Models"
    risk_filter:  str = "All Records"
    sections:     dict = {}

@app.post("/api/report")
def ep_generate_report(req: ReportRequest):
    """Generate a PDF report and return it as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    PRIMARY  = colors.HexColor("#1A56DB")
    RED      = colors.HexColor("#DC2626")
    AMBER    = colors.HexColor("#D97706")
    GREEN    = colors.HexColor("#059669")
    LITE_BG  = colors.HexColor("#F8FAFF")
    BORDER   = colors.HexColor("#E5E7EB")
    DARK     = colors.HexColor("#111827")
    GREY     = colors.HexColor("#6B7280")

    H1 = ParagraphStyle("H1", fontSize=20, fontName="Helvetica-Bold",
                         textColor=PRIMARY, spaceAfter=4)
    H2 = ParagraphStyle("H2", fontSize=13, fontName="Helvetica-Bold",
                         textColor=DARK, spaceAfter=6, spaceBefore=14)
    H3 = ParagraphStyle("H3", fontSize=10, fontName="Helvetica-Bold",
                         textColor=PRIMARY, spaceAfter=4, spaceBefore=8,
                         textTransform="uppercase")
    BODY = ParagraphStyle("BODY", fontSize=9, fontName="Helvetica",
                           textColor=GREY, spaceAfter=4, leading=14)
    SMALL = ParagraphStyle("SMALL", fontSize=8, fontName="Helvetica",
                            textColor=GREY, spaceAfter=2)
    CONF  = ParagraphStyle("CONF", fontSize=7, fontName="Helvetica",
                            textColor=GREY, alignment=TA_CENTER, spaceBefore=8)

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = req.sections or {
        "exec":True,"method":True,"patients":True,"stats":True
    }

    # ── Determine which datasets to include
    if req.dataset_name == "All Datasets":
        ds_names = list(RAW_DATASETS.keys())
    else:
        ds_names = [req.dataset_name] if req.dataset_name in RAW_DATASETS else list(RAW_DATASETS.keys())

    total_rows = sum(len(RAW_DATASETS[n]) for n in ds_names if n in RAW_DATASETS)
    total_pos  = 0
    total_inc  = 0
    model_keys = list(MODELS.keys())

    story = []

    # ── HEADER ───────────────────────────────────────────────
    story.append(Paragraph("Leukemia Harmoniser", H1))
    story.append(Paragraph(f"Clinical Analysis Report", H2))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceAfter=6))
    story.append(Paragraph(
        f"Generated: {now} &nbsp;·&nbsp; "
        f"Datasets: {len(ds_names)} &nbsp;·&nbsp; "
        f"Models: {len(model_keys)} &nbsp;·&nbsp; "
        f"Records: {total_rows:,}",
        SMALL
    ))
    story.append(Spacer(1, 12))

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────
    if sections.get("exec", True):
        story.append(Paragraph("Executive Summary", H3))
        # Gather positives
        for n in ds_names:
            preds = PREDICTIONS.get(n, {})
            mk0 = model_keys[0] if model_keys else None
            if mk0 and mk0 in preds:
                total_pos += preds[mk0].get("positive", 0)
                total_inc += preds[mk0].get("inconclusive", 0)

        pos_rate = round(total_pos/total_rows*100, 1) if total_rows else 0
        inc_rate = round(total_inc/total_rows*100, 1) if total_rows else 0
        neg = max(0, total_rows - total_pos - total_inc)

        story.append(Paragraph(
            f"This report covers a total of <b>{total_rows:,}</b> patient CBC records across "
            f"<b>{len(ds_names)}</b> dataset(s). Analysis using <b>{len(model_keys)}</b> trained "
            f"machine learning model(s) identified <b>{total_pos:,}</b> leukemia-positive cases "
            f"({pos_rate}%) and <b>{total_inc:,}</b> inconclusive cases ({inc_rate}%) requiring "
            f"further clinical evaluation. <b>{neg:,}</b> records showed no leukemia indicators.",
            BODY
        ))

        # Summary KPI table
        kpi_data = [
            ["Metric", "Count", "Rate"],
            ["Leukemia Positive", f"{total_pos:,}", f"{pos_rate}%"],
            ["Inconclusive",      f"{total_inc:,}", f"{inc_rate}%"],
            ["Negative / Healthy",f"{neg:,}",       f"{100-pos_rate-inc_rate:.1f}%"],
            ["Total Records",     f"{total_rows:,}", "100%"],
        ]
        kpi_table = Table(kpi_data, colWidths=[7*cm, 4*cm, 4*cm])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  PRIMARY),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LITE_BG]),
            ("TEXTCOLOR",    (0,1), (0,-1),  DARK),
            ("TEXTCOLOR",    (1,1), (1,1),   RED),
            ("TEXTCOLOR",    (1,2), (1,2),   AMBER),
            ("TEXTCOLOR",    (1,3), (1,3),   GREEN),
            ("GRID",         (0,0), (-1,-1), 0.5, BORDER),
            ("BOTTOMPADDING",(0,0), (-1,-1), 6),
            ("TOPPADDING",   (0,0), (-1,-1), 6),
            ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 10))

    # ── METHODOLOGY ───────────────────────────────────────────
    if sections.get("method", True):
        story.append(Paragraph("Methodology", H3))
        story.append(Paragraph(
            "Input CBC datasets were processed through the Leukemia Harmoniser pipeline: "
            "(1) Automatic column detection using fuzzy string matching against a canonical "
            "CBC field library covering hundreds of alias variants; "
            "(2) Scale normalisation to detect and convert 10⁹/L values to cells/µL; "
            "(3) Outlier cleaning using clinically validated reference ranges with median imputation; "
            "(4) Null value imputation using population-level defaults; "
            "(5) Multi-model ensemble prediction using XGBoost, Random Forest, and Neural Network classifiers; "
            "(6) Three-tier risk stratification: High Risk (≥70%), Inconclusive (30–70%), Low Risk (<30%).",
            BODY
        ))

    # ── DATASET BREAKDOWN ─────────────────────────────────────
    if sections.get("stats", True):
        story.append(Paragraph("Dataset & Model Statistics", H3))
        for n in ds_names:
            if n not in RAW_DATASETS:
                continue
            stats = HARMONIZE_STATS.get(n, {})
            preds = PREDICTIONS.get(n, {})
            story.append(Paragraph(f"<b>{n.replace('_',' ')}</b>", BODY))

            ds_data = [["Model", "Positive", "Rate", "High Risk", "Inconclusive", "Low Risk", "ROC-AUC"]]
            for mk in model_keys:
                ms = preds.get(mk, {})
                met = MODELS.get(mk, {}).get("metrics", {})
                ds_data.append([
                    MODELS[mk].get("display_name", mk) if mk in MODELS else mk,
                    f"{ms.get('positive',0):,}",
                    f"{ms.get('positive_rate',0):.1f}%",
                    f"{ms.get('high_risk',0):,}",
                    f"{ms.get('inconclusive',0):,}",
                    f"{ms.get('low_risk',0):,}",
                    str(met.get("roc_auc","—")),
                ])
            if len(ds_data) > 1:
                ds_table = Table(ds_data, colWidths=[3.5*cm,2*cm,2*cm,2.2*cm,2.5*cm,2*cm,2*cm])
                ds_table.setStyle(TableStyle([
                    ("BACKGROUND",   (0,0),(-1,0),   PRIMARY),
                    ("TEXTCOLOR",    (0,0),(-1,0),   colors.white),
                    ("FONTNAME",     (0,0),(-1,0),   "Helvetica-Bold"),
                    ("FONTNAME",     (0,1),(-1,-1),  "Helvetica"),
                    ("FONTSIZE",     (0,0),(-1,-1),  8),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LITE_BG]),
                    ("GRID",         (0,0),(-1,-1),  0.5, BORDER),
                    ("BOTTOMPADDING",(0,0),(-1,-1),  5),
                    ("TOPPADDING",   (0,0),(-1,-1),  5),
                    ("LEFTPADDING",  (0,0),(-1,-1),  6),
                    ("TEXTCOLOR",    (1,1),(-1,-1),  DARK),
                ]))
                story.append(ds_table)
            story.append(Spacer(1, 8))

    # ── PATIENT DETAIL ────────────────────────────────────────
    if sections.get("patients", True):
        story.append(Paragraph("Patient-Level Detail (High Risk Cases)", H3))
        for n in ds_names:
            preds = PREDICTIONS.get(n, {})
            rows  = preds.get("_rows", [])
            hr_rows = [r for r in rows if r.get("risk_level") == "High Risk"][:50]
            if not hr_rows:
                story.append(Paragraph(f"No high-risk cases in {n.replace('_',' ')}.", BODY))
                continue
            story.append(Paragraph(
                f"<b>{n.replace('_',' ')}</b> — showing {len(hr_rows)} high-risk record(s)", BODY
            ))
            mk0 = model_keys[0] if model_keys else None
            col_heads = ["Row #","WBC","RBC","HGB","PLT","Age","Risk","Prob%"]
            if mk0:
                col_heads.append(MODELS[mk0].get("display_name",mk0) if mk0 in MODELS else mk0)
            pat_data = [col_heads]
            for r in hr_rows:
                row_data = [
                    str(r.get("row_index","—")),
                    f"{r.get('WBC',0):,.0f}" if r.get("WBC") else "—",
                    str(r.get("RBC","—")),
                    str(r.get("Hemoglobin","—")),
                    f"{r.get('Platelets',0):,.0f}" if r.get("Platelets") else "—",
                    str(r.get("Age","—")),
                    r.get("risk_level","—"),
                    f"{r.get('max_probability',0):.1f}%",
                ]
                if mk0:
                    row_data.append(r.get(f"{mk0}_prediction","—"))
                pat_data.append(row_data)
            widths = [1.5*cm,2.2*cm,1.8*cm,1.8*cm,2.2*cm,1.5*cm,2.2*cm,1.8*cm]
            if mk0: widths.append(2.5*cm)
            pat_table = Table(pat_data, colWidths=widths, repeatRows=1)
            pat_table.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,0),  PRIMARY),
                ("TEXTCOLOR",    (0,0),(-1,0),  colors.white),
                ("FONTNAME",     (0,0),(-1,0),  "Helvetica-Bold"),
                ("FONTNAME",     (0,1),(-1,-1), "Helvetica"),
                ("FONTSIZE",     (0,0),(-1,-1), 7),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#FEF2F2"), colors.white]),
                ("GRID",         (0,0),(-1,-1), 0.4, BORDER),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                ("TOPPADDING",   (0,0),(-1,-1), 4),
                ("LEFTPADDING",  (0,0),(-1,-1), 5),
            ]))
            story.append(pat_table)
            if len([r for r in rows if r.get("risk_level")=="High Risk"]) > 50:
                story.append(Paragraph("(Showing first 50 high-risk records only)", SMALL))
            story.append(Spacer(1, 10))

    # ── FOOTER ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceBefore=16))
    story.append(Paragraph(
        f"CONFIDENTIAL — For clinical use only · Leukemia Harmoniser v2.0 · {now}",
        CONF
    ))

    doc.build(story)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="leukemia_report_{now[:10]}.pdf"'}
    )


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return FRONTEND_HTML

# ── SERVER LAUNCH ─────────────────────────────────────────────
print("\n  Starting server on port 8000...")

# Configure ngrok
ngrok.set_auth_token(NGROK_TOKEN)

# Start uvicorn in a background thread using nest_asyncio-safe approach
config = uvicorn.Config(
    app=app,
    host="0.0.0.0",
    port=8000,
    log_level="warning",
    loop="asyncio",
)
server = uvicorn.Server(config)

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

thread = threading.Thread(target=run_server, daemon=True)
thread.start()
time.sleep(3)   # wait for uvicorn to bind

# Open ngrok tunnel
tunnel = ngrok.connect(8000, bind_tls=True)
PUBLIC_URL = tunnel.public_url

print("\n" + "=" * 60)
print("  LEUKEMIA HARMONISER IS LIVE")
print("=" * 60)
print(f"\n  🌐  {PUBLIC_URL}")
print(f"\n  API:  {PUBLIC_URL}/api/status")
print(f"  Docs: {PUBLIC_URL}/docs")
print("\n  Keep this cell running.")
print("  All dataset uploads and predictions happen in the UI.")
print("=" * 60)