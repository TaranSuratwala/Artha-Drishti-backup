#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from MLPredictor import UnifiedStockPredictor
p = UnifiedStockPredictor()
r = p.predict(ticker='RELIANCE.NS')
import json
print(json.dumps(r.get('recommendation', {}), indent=2))
