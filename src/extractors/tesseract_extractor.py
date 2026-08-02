import os
import re
import json
from pathlib import Path
from PIL import Image
import pytesseract
import cv2
import numpy as np

class TesseractExtractor:
    """Extract bill information using Tesseract OCR"""
    
    def __init__(self, tesseract_path=None):
        """Initialize Tesseract OCR"""
        # For Windows, specify the path to tesseract.exe
        if os.name == 'nt':  # Windows
            if tesseract_path is None:
                tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Supported languages
        self.languages = 'eng'
        
        # Patterns for extracting fields from text
        self.patterns = {
            'vendor_name': r'(?:Vendor|Shop|Store|Merchant|From|Bill from|Bill of|M\/s|M\/S)[\s:]*([A-Za-z0-9\s\.\&]+)',
            'invoice_number': r'(?:Invoice|Bill|Receipt|Inv|Ref)[\s:]*[#NO:]*\s*([A-Za-z0-9\-]+)',
            'date': r'(?:Date|Dated)[\s:]*([0-9]{1,2}[/\-][0-9]{1,2}[/\-][0-9]{2,4})',
            'amount': r'(?:Total|Amount|Grand Total|Net)[\s:]*[₹$€£]?\s*([0-9,]+\.?[0-9]*)',
            'gst': r'(?:GST|Tax|VAT)[\s:]*[₹$€£]?\s*([0-9,]+\.?[0-9]*)'
        }
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        # Read image
        img = cv2.imread(str(image_path))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to enhance contrast
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def extract_text(self, image_path):
        """Extract raw text from image using Tesseract"""
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Save temporary image
            temp_path = "temp_processed.jpg"
            cv2.imwrite(temp_path, processed_img)
            
            # Extract text with Tesseract
            text = pytesseract.image_to_string(
                temp_path,
                lang=self.languages,
                config='--psm 6 --oem 3'
            )
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Tesseract OCR error: {str(e)}")
    
    def parse_bill_text(self, text):
        """Parse extracted text to extract bill fields"""
        data = {
            'vendor_name': None,
            'invoice_number': None,
            'date': None,
            'total_amount': None,
            'currency': None,
            'gst_amount': None,
            'gst_rate': None
        }
        
        # Clean text
        text = text.replace('\n', ' ').replace('\r', '')
        
        # Extract vendor name
        vendor_match = re.search(self.patterns['vendor_name'], text, re.IGNORECASE)
        if vendor_match:
            data['vendor_name'] = vendor_match.group(1).strip()
        
        # Extract invoice number
        invoice_match = re.search(self.patterns['invoice_number'], text, re.IGNORECASE)
        if invoice_match:
            data['invoice_number'] = invoice_match.group(1).strip()
        
        # Extract date
        date_match = re.search(self.patterns['date'], text, re.IGNORECASE)
        if date_match:
            data['date'] = date_match.group(1).strip()
        
        # Extract total amount
        amount_match = re.search(self.patterns['amount'], text, re.IGNORECASE)
        if amount_match:
            amount_str = amount_match.group(1).replace(',', '')
            try:
                data['total_amount'] = float(amount_str)
            except:
                pass
        
        # Extract GST
        gst_match = re.search(self.patterns['gst'], text, re.IGNORECASE)
        if gst_match:
            gst_str = gst_match.group(1).replace(',', '')
            try:
                data['gst_amount'] = float(gst_str)
            except:
                pass
        
        # Detect currency
        if '₹' in text or 'INR' in text:
            data['currency'] = 'INR'
        elif '$' in text or 'USD' in text:
            data['currency'] = 'USD'
        elif '€' in text or 'EUR' in text:
            data['currency'] = 'EUR'
        else:
            data['currency'] = 'INR'
        
        return data
    
    def extract_bill(self, image_path):
        """Complete extraction pipeline"""
        result = {
            'image_path': str(image_path),
            'model': 'tesseract-ocr',
            'success': False,
            'extracted_data': {},
            'raw_text': '',
            'error': None
        }
        
        try:
            # Extract raw text
            raw_text = self.extract_text(image_path)
            result['raw_text'] = raw_text
            
            if not raw_text or len(raw_text) < 10:
                result['error'] = 'No text extracted from image'
                return result
            
            # Parse text
            parsed_data = self.parse_bill_text(raw_text)
            result['extracted_data'] = parsed_data
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result


def extract_batch(images_dir="data/images", output_dir="outputs/tesseract_results"):
    """Extract all bills in a directory"""
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Extracting from {len(image_files)} bills using Tesseract OCR...")
    print()
    
    extractor = TesseractExtractor()
    results = []
    
    for i, img in enumerate(image_files, 1):
        print(f"   [{i}/{len(image_files)}] Processing: {img.name}")
        result = extractor.extract_bill(img)
        result['image_name'] = img.name
        results.append(result)
        
        if result['success']:
            data = result['extracted_data']
            print(f"   [OK] Vendor: {data.get('vendor_name', 'N/A')}")
            print(f"   Amount: {data.get('total_amount', 'N/A')}")
            print(f"   Date: {data.get('date', 'N/A')}")
        else:
            print(f"   [FAIL] Failed: {result['error'][:50]}")
        print()
    
    # Save results
    output_file = output_path / "tesseract_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"   Success rate: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Run extraction on all bills
    extract_batch()