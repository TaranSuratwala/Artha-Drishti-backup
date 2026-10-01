import sys
import os
import re

def patch_ml_predictor():
    pass

def patch_advanced_feature_engine():
    with open('AdvancedFeatureEngine.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Ensure text is imported from sqlalchemy
    if "from sqlalchemy import text" not in content:
        content = re.sub(r"(import pandas as pd\n)", r"\1from sqlalchemy import text\n", content)

    # A3. _fno_features, FII lag and new table
    if "shift(-1)" not in content:
        a3_patch = """fii_df = pd.read_sql(text("SELECT d AS date, fii_net AS fii_dii_net_flow FROM fii_dii ORDER BY d"), conn)
                if not fii_df.empty:
                    fii_df['date'] = pd.to_datetime(fii_df['date']).shift(-1)      # flow of day d usable from the NEXT session
                    fii_df = fii_df.dropna(subset=['date'])
                    df = df.merge(fii_df, on='date', how='left')"""
        
        # We replace the exact old block
        old_block = r"fii_df = pd\.read_sql\(\"SELECT date, fii_net_value as fii_dii_net_flow FROM fii_dii_flow\", conn\)\s*if not fii_df\.empty:\s*fii_df\['date'\] = pd\.to_datetime\(fii_df\['date'\]\)\s*df = df\.merge\(fii_df, on='date', how='left'\)"
        content = re.sub(old_block, a3_patch, content, flags=re.DOTALL)

    with open('AdvancedFeatureEngine.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_advanced_feature_engine()
    print("Patched successfully")
