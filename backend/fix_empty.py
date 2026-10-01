import re
import sys

def fix_mlpredictor():
    with open('MLPredictor.py', 'r', encoding='utf-8') as f:
        code = f.read()

    # Fix 1: Raw test returns logging (line 6646)
    pattern1 = (
        r'(logger\.info\(f\"   Raw test returns saved: \{len\(self\._test_raw_returns\):,\} samples, \")\s*'
        r'(f\"mean=\{np\.mean\(self\._test_raw_returns\):\.4f\}, std=\{np\.std\(self\._test_raw_returns\):\.4f\}, \")\s*'
        r'(f\"range=\[\{np\.min\(self\._test_raw_returns\):\.4f\}, \{np\.max\(self\._test_raw_returns\):\.4f\}\]\"\))'
    )
    replacement1 = '''if len(self._test_raw_returns) > 0:
            \\1
                        \\2
                        \\3
        else:
            logger.info("   Raw test returns saved: 0 samples (Test set is empty).")'''
    
    code = re.sub(pattern1, replacement1, code, count=1)

    # Fix 2: validation loader zero len
    code = re.sub(
        r'(\n\s*)(val_loss /= len\(val_loader\))',
        r'\1if len(val_loader) > 0:\1    \2',
        code, count=1
    )

    with open('MLPredictor.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Empty array fixes applied")

if __name__ == '__main__':
    fix_mlpredictor()
