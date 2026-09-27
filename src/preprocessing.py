import re
import pandas as pd
import numpy as np

def clean_series_text(series: pd.Series) -> pd.Series:
    """
    Vectorized text cleaner: lowercases, cleans special characters, normalizes whitespace.
    """
    s = series.fillna('').astype(str).str.lower()
    # Normalize '&' to ' and '
    s = s.str.replace('&', ' and ', regex=False)
    # Replace non-alphanumeric with space
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    # Collapse multiple whitespace
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ultra-fast vectorized preprocessing across business entity fields.
    """
    res = df.copy()
    
    name_col = [c for c in df.columns if 'name' in c.lower()][0]
    addr_col = [c for c in df.columns if 'address' in c.lower()][0]
    country_col = [c for c in df.columns if 'country' in c.lower()][0]
    
    # 1. Clean Name & Prefix signatures
    clean_name = clean_series_text(res[name_col])
    res['norm_name'] = clean_name
    res['name_prefix_4'] = clean_name.str.slice(0, 4)
    
    # 2. Clean Address
    clean_addr = clean_series_text(res[addr_col])
    res['norm_address'] = clean_addr
    
    # 3. Clean Country
    res['norm_country'] = res[country_col].fillna('').astype(str).str.upper().str.strip()
    res.loc[res['norm_country'].isin(['USA', 'UNITED STATES']), 'norm_country'] = 'US'
    res.loc[res['norm_country'].isin(['IND', 'IN']), 'norm_country'] = 'INDIA'
    res.loc[res['norm_country'].isin(['FRA', 'FR']), 'norm_country'] = 'FRANCE'
    
    return res
