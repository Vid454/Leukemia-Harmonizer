import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

class CBCHarmonizer:
    CBC_SYNONYMS = {
        "WBC_k_ul": ["wbc", "whitebc", "wbc_count", "leukocytes_total", "leucocytes", "total_wbc", "wbc_abs"],
        "RBC_mil_ul": ["rbc", "rbc_count", "redbc", "red_blood_cells", "rbc_absolute", "rbc_total"],
        "HGB_g_dl": ["hgb", "hemoglobin", "hgb_g_dl", "haemoglobin", "hb"],
        "HCT_%": ["hct", "hematocrit", "hct_%", "haematocrit", "hct_pc"],
        "PLT_k_ul": ["plt", "platelets", "plt_cnt", "platelet_count", "thrombocytes", "plt_abs"],
        "Neutrophils_k_ul": ["neut", "neutrophils", "neutrophil_abs", "neutrophil_count", "neut_abs"],
        "Lymphocytes_k_ul": ["lymph", "lymphocytes", "lymph_abs", "lymphocyte_count", "lymph_total"],
        "Monocytes_k_ul": ["mono", "monocytes", "mono_abs", "monocyte_count"],
        "Eosinophils_k_ul": ["eos", "eosinophils", "eos_abs", "eosinophil_count"],
        "Basophils_k_ul": ["baso", "basophils", "baso_abs", "basophil_count"],
        "MCV_fL": ["mcv", "mcv_fl", "mean_corpuscular_volume", "mchc"],
        "RDW_%": ["rdw", "rdw_cv", "rdw_sd", "red_cell_distribution"],
        "PDW_%": ["pdw", "pdw_cv", "platelet_distribution_width", "pdw_sd"],
        "MPV_fL": ["mpv", "mpv_fl", "mean_platelet_volume", "mpv_fL"],
        "Blasts_%": ["blasts", "blast_cells", "blast_percentage", "blast_pct"],
        "IG_%": ["ig", "immature_granulocytes", "imm_gran", "ig_pct"],
        "NRBC_%": ["nrbc", "nucleated_rbc", "nrbc_per_100", "nrbc_pct"],
        "NLR": ["nlr", "neut_lymph_ratio", "neutrophil_lymphocyte"],
        "PLR": ["plr", "plt_lymph_ratio", "platelet_lymphocyte"],
        "MLR": ["mlr", "mono_lymph_ratio", "monocyte_lymphocyte"],
    }
    
    UNIT_PATTERNS = {
        r'(10\^9|/10.?9)/?l': 0.001,
        r'(k|thou)/?ul': 1.0,
        r'/mm3?': 0.001,
        r'x10.?9/l': 0.001,
        r'mil/?ul': 1000.0,
        r'%': 1.0,
        r'fl': 1.0,
        r'g/dl': 1.0,
    }
    
    STANDARD_COLS = list(CBC_SYNONYMS.keys())
    
    def __init__(self):
        self.imputer = IterativeImputer(random_state=42, max_iter=5, n_nearest_features=5)
    
    def _robust_map_column(self, col: str) -> Tuple[Optional[str], float]:
        raw = re.sub(r'[\(\)\[\]\s\-_]+', '', col.lower())
        candidates = list(self.CBC_SYNONYMS.keys())
        
        scores = []
        
        for scorer in [fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.ratio]:
            try:
                match, score, _ = process.extractOne(raw, candidates, scorer=scorer, score_cutoff=60)
                if match:
                    scores.append((match, score))
            except:
                continue
        
        if scores:
            return max(scores, key=lambda x: x[1])
        return None, 0.0
    
    def _safe_unit_convert(self, series: pd.Series, col_name: str) -> pd.Series:
        try:
            factor = 1.0
            col_lower = col_name.lower()
            for pattern, f in self.UNIT_PATTERNS.items():
                if re.search(pattern, col_lower):
                    factor = f
                    break
            return pd.to_numeric(series, errors='coerce') * factor
        except:
            return pd.to_numeric(series, errors='coerce')
    
    def _adaptive_outlier_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                if df[col].notna().sum() > 5:
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    df[col] = df[col].clip(lower, upper)
            except:
                continue
        return df
    
    def _multi_strategy_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        try:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        except:
            medians = df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(medians)
        return df
    
    def _safe_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ratio_pairs = {
            "NLR": ("Neutrophils_k_ul", "Lymphocytes_k_ul"),
            "PLR": ("PLT_k_ul", "Lymphocytes_k_ul"),
            "MLR": ("Monocytes_k_ul", "Lymphocytes_k_ul"),
        }
        
        for ratio_name, (num_col, den_col) in ratio_pairs.items():
            if num_col in df.columns and den_col in df.columns:
                mask = df[den_col] > 0
                df[ratio_name] = np.where(
                    mask, df[num_col] / df[den_col], np.nan
                )
        
        return df.fillna(0)
    
    def harmonize_batch(self, file_paths: List[Path]) -> Tuple[pd.DataFrame, Dict]:
        dfs = []
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                df["source_file"] = path.name
                dfs.append(df)
            except:
                continue
        
        if not dfs:
            return pd.DataFrame(), {}
        
        raw_df = pd.concat(dfs, ignore_index=True)
        
        mapping_stats = {}
        col_mapping = {}
        
        for col in raw_df.columns:
            match, score = self._robust_map_column(col)
            col_mapping[col] = match or col
            mapping_stats[col] = score
        
        df = raw_df.rename(columns=col_mapping)
        
        for old_col, new_col in col_mapping.items():
            if new_col in self.STANDARD_COLS and old_col != new_col:
                df[new_col] = self._safe_unit_convert(df[old_col], old_col)
        
        df = self._adaptive_outlier_clip(df)
        df = self._multi_strategy_impute(df)
        df = self._safe_ratio_features(df)
        
        report = {
            "input_files": len(file_paths),
            "input_rows": len(raw_df),
            "output_rows": len(df),
            "auto_matched_cols": sum(1 for s in mapping_stats.values() if s >= 70),
            "total_cols_mapped": len(mapping_stats),
            "match_rate": sum(1 for s in mapping_stats.values() if s >= 70) / len(mapping_stats),
            "standard_features_found": sum(1 for c in self.STANDARD_COLS if c in df.columns),
        }
        
        return df[self.STANDARD_COLS + ["source_file"]], report
    
    def harmonize_single(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        return self.harmonize_batch([df])
