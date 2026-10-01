import re
import sys

def fix_all_empty():
    with open('MLPredictor.py', 'r', encoding='utf-8') as f:
        code = f.read()

    # Guard around line 7865 (Raw returns stats)
    pattern2 = (
        r'(logger\.info\(f\"   Raw returns stats: mean=\{np\.mean\(_actual_returns\):\.6f\}, \"\s*)'
        r'(f\"std=\{np\.std\(_actual_returns\):\.6f\}, \"\s*)'
        r'(f\"range=\[\{np\.min\(_actual_returns\):\.4f\}, \{np\.max\(_actual_returns\):\.4f\}\]\"\))'
    )
    replacement2 = '''if len(_actual_returns) > 0:
                    \\1\\2\\3
                else:
                    logger.info("   Raw returns stats: 0 samples.")'''
    code = re.sub(pattern2, replacement2, code, count=1)
    
    # Fix around line 8757: actual_returns logging in CWCB backtest
    pattern3 = (
        r'(logger\.info\(f\"   Actual returns: mean=\{np\.mean\(actual_returns\):\.4f\}, \"\s*)'
        r'(f\"std=\{np\.std\(actual_returns\):\.4f\}, \"\s*)'
        r'(f\"range=\[\{np\.min\(actual_returns\):\.4f\}, \{np\.max\(actual_returns\):\.4f\}\]\"\))'
    )
    replacement3 = '''if len(actual_returns) > 0:
                \\1\\2\\3
            else:
                logger.info("   Actual returns: 0 samples.")'''
    code = re.sub(pattern3, replacement3, code, count=1)

    with open('MLPredictor.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Additional empty array fixes applied.")

if __name__ == '__main__':
    fix_all_empty()
