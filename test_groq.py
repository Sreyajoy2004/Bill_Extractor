from src.extractors.tesseract_extractor import TesseractExtractor
from src.extractors.llm_extractor import BillExtractor

print('Testing Tesseract + Groq text extraction on bill_001.jpg...')
print('=' * 60)

tesseract = TesseractExtractor()
result = tesseract.extract_bill('data/images/bill_001.jpg')

if not result['success']:
    print(f"Tesseract failed: {result['error']}")
    exit(1)

raw_text = result.get('raw_text', '')
print(f"Tesseract extracted {len(raw_text)} chars")
print(f"Raw text preview: {raw_text[:200]}...")
print()

models_to_test = [
    'groq-llama-versatile',
    'groq-llama-instant',
    'groq-llama-scout'
]

working_model = None

for model_key in models_to_test:
    print(f"Testing Groq model: {model_key}")
    try:
        extractor = BillExtractor(model_key)
        llm_result = extractor.extract_from_text(raw_text)
        
        if llm_result['success']:
            print(f"  [OK] WORKS!")
            print(f"  Vendor: {llm_result['extracted_data'].get('vendor_name', 'N/A')}")
            print(f"  Amount: {llm_result['extracted_data'].get('total_amount', 'N/A')}")
            print(f"  Date: {llm_result['extracted_data'].get('date', 'N/A')}")
            working_model = model_key
            break
        else:
            print(f"  [FAIL] Failed: {llm_result.get('error', 'Unknown')[:80]}")
    except Exception as e:
        print(f"  [FAIL] Error: {str(e)[:80]}")

print()
if working_model:
    print(f"RECOMMENDED: Use '{working_model}' for Groq text extraction")
else:
    print("No working Groq models found.")
