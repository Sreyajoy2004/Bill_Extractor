import json
from pathlib import Path

print('TESSERACT PERFORMANCE ANALYSIS\n')
print('=' * 70)

# Load Tesseract results
with open('outputs/tesseract_results/tesseract_results.json') as f:
    results = json.load(f)

# Load ground truth
gt_dir = Path('data/ground_truth')

# Track stats
total = 0
vendor_correct = 0
date_correct = 0
amount_correct = 0
vendor_extracted = 0
date_extracted = 0
amount_extracted = 0

print(f"{'Image':<15} {'Vendor Match':<15} {'Date Match':<15} {'Amount Match':<15}")
print('-' * 70)

for r in results:
    gt_path = gt_dir / r['image_name'].replace('.jpg', '.json')
    if not gt_path.exists():
        continue
    
    with open(gt_path) as f:
        gt = json.load(f)
    
    pred = r['extracted_data']
    total += 1
    
    # Check vendor
    gt_vendor = gt['vendor_name'].lower()
    pred_vendor = pred.get('vendor_name')
    if pred_vendor is not None:
        vendor_extracted += 1
        pred_vendor = pred_vendor.lower()
        if gt_vendor in pred_vendor or pred_vendor in gt_vendor:
            vendor_correct += 1
            vendor_status = '[OK]'
        else:
            vendor_status = '[FAIL]'
    else:
        vendor_status = '[FAIL]'
    
    # Check date
    gt_date = gt['date']
    pred_date = pred.get('date')
    if pred_date is not None:
        date_extracted += 1
        if gt_date == pred_date:
            date_correct += 1
            date_status = '[OK]'
        else:
            date_status = '[FAIL]'
    else:
        date_status = '[FAIL]'
    
    # Check amount
    gt_amount = gt['total_amount']
    pred_amount = pred.get('total_amount')
    if pred_amount is not None:
        amount_extracted += 1
        if gt_amount == pred_amount:
            amount_correct += 1
            amount_status = '[OK]'
        else:
            amount_status = '[FAIL]'
    else:
        amount_status = '[FAIL]'
    
    print(f"{r['image_name']:<15} {vendor_status:<15} {date_status:<15} {amount_status:<15}")

print('=' * 70)

print('\nACCURACY SUMMARY:')
if total > 0:
    print(f'   Vendor Name: {vendor_correct}/{total} ({vendor_correct/total*100:.1f}%)')
    print(f'   Date: {date_correct}/{total} ({date_correct/total*100:.1f}%)')
    print(f'   Amount: {amount_correct}/{total} ({amount_correct/total*100:.1f}%)')
else:
    print('   No data to analyze')

print('\nEXTRACTION RATE:')
if total > 0:
    print(f'   Vendor extracted: {vendor_extracted}/{total} ({vendor_extracted/total*100:.1f}%)')
    print(f'   Date extracted: {date_extracted}/{total} ({date_extracted/total*100:.1f}%)')
    print(f'   Amount extracted: {amount_extracted}/{total} ({amount_extracted/total*100:.1f}%)')
