#!/usr/bin/env python3
"""Quick test script to diagnose prediction issues"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from MLPredictor import UnifiedStockPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Test prediction
try:
    predictor = UnifiedStockPredictor()
    result = predictor.predict(ticker="RELIANCE.NS")
    
    print("\n✓ Prediction successful!")
    print(f"Signal: {result.get('recommendation', {}).get('signal', 'N/A')}")
    print(f"Direction Probability: {result.get('recommendation', {}).get('direction_probability', 'N/A')}")
    
    if 'error' in result:
        print(f"ERROR: {result['error']}")
    
except Exception as e:
    print(f"\n✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
