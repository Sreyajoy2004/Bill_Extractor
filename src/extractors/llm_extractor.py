import os
import json
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import time

from PIL import Image
import google.generativeai as genai
from openai import OpenAI
import anthropic
from groq import Groq

from config import Config

class BillExtractor:
    """Extract bill information using various LLM models"""
    
    def __init__(self, model_key: str = "gemini-flash"):
        """Initialize the extractor with a specific model"""
        self.model_key = model_key
        self.model_config = Config.get_model_config(model_key)
        self.provider = self.model_config["provider"]
        self.model_name = self.model_config["name"]
        
        # Initialize the appropriate client
        self.client = None
        self._init_client()
        
        # Track usage for cost calculation
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
        
    def _init_client(self):
        """Initialize the API client based on provider"""
        if self.provider == "google":
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.client = genai.GenerativeModel(self.model_name)
        elif self.provider == "openai":
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        elif self.provider == "groq":
            self.client = Groq(api_key=Config.GROQ_API_KEY)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _get_text_extraction_prompt(self) -> str:
        """Get the prompt for text-based bill extraction from OCR output"""
        return """You are an AI assistant that extracts structured information from OCR text of handwritten bills/receipts.
Given the following OCR text from a handwritten Indian bill, extract the specified fields and return ONLY a valid JSON object.

Fields to extract:
1. vendor_name: The shop/vendor name (string)
2. invoice_number: The bill/invoice number if present, else null
3. date: The date on the bill in DD/MM/YYYY format
4. total_amount: The total amount as a number (without currency symbol)
5. currency: The currency code (e.g., INR, USD)
6. gst_amount: The GST/tax amount if present, else null
7. gst_rate: The GST rate percentage if present, else null

OCR Text:
---
{ocr_text}
---

Return ONLY valid JSON. Do not include markdown formatting, explanations, or any text outside the JSON object.
"""
    
    def _get_extraction_prompt(self) -> str:
        """Get the prompt for bill extraction"""
        return """You are an AI assistant that extracts information from handwritten bills/receipts.
Extract the following fields from the bill image and return them as a JSON object:

1. vendor_name: The shop/vendor name
2. invoice_number: The bill/invoice number (if present, else null)
3. date: The date on the bill (format as DD/MM/YYYY)
4. total_amount: The total amount (as a number, without currency symbol)
5. currency: The currency (e.g., INR, USD, etc.)
6. gst_amount: The GST/tax amount (if present, else null)
7. gst_rate: The GST rate percentage (if present, else null)

Return ONLY a valid JSON object with these fields. Do not include any other text or explanation.
Example:
{
  "vendor_name": "Krishna General Store",
  "invoice_number": "INV-2024-001",
  "date": "15/01/2024",
  "total_amount": 1250.50,
  "currency": "INR",
  "gst_amount": 75.03,
  "gst_rate": 6.0
}
"""
    
    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract bill information from an image"""
        
        result = {
            "image_path": image_path,
            "model": self.model_key,
            "success": False,
            "extracted_data": {},
            "error": None,
            "usage": {}
        }
        
        try:
            # Read the image
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Get extraction prompt
            prompt = self._get_extraction_prompt()
            
            # Call the appropriate API
            if self.provider == "google":
                response = self._call_gemini(image_data, prompt)
            elif self.provider == "openai":
                response = self._call_openai(image_data, prompt)
            elif self.provider == "anthropic":
                response = self._call_claude(image_data, prompt)
            elif self.provider == "groq":
                response = self._call_groq(image_data, prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Parse response
            result["extracted_data"] = self._parse_response(response)
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            print(f"   Error: {str(e)[:100]}...")
        
        return result
    
    def extract_from_text(self, ocr_text: str) -> Dict[str, Any]:
        """Extract bill information from OCR text using text-only LLM"""
        
        result = {
            "image_path": None,
            "model": self.model_key,
            "success": False,
            "extracted_data": {},
            "error": None,
            "usage": {}
        }
        
        try:
            if not ocr_text or len(ocr_text.strip()) < 10:
                result["error"] = "OCR text is too short or empty"
                return result
            
            prompt = self._get_text_extraction_prompt().format(ocr_text=ocr_text)
            
            if self.provider == "groq":
                response = self._call_groq_text(prompt)
            elif self.provider == "google":
                response = self._call_gemini_text(prompt)
            elif self.provider == "openai":
                response = self._call_openai_text(prompt)
            elif self.provider == "anthropic":
                response = self._call_claude_text(prompt)
            else:
                raise ValueError(f"Unsupported provider for text extraction: {self.provider}")
            
            result["extracted_data"] = self._parse_response(response)
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            print(f"   Error: {str(e)[:100]}...")
        
        return result
    
    def _call_gemini(self, image_data: bytes, prompt: str) -> str:
        """Call Google Gemini API"""
        try:
            from PIL import Image as PILImage
            import io
            
            # Convert bytes to PIL Image
            image = PILImage.open(io.BytesIO(image_data))
            
            # Generate content with image and prompt
            response = self.client.generate_content([prompt, image])
            
            # Track usage (approximate)
            self.input_tokens += len(prompt.split()) + len(image_data) // 1000
            if response.text:
                self.output_tokens += len(response.text.split())
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _call_openai(self, image_data: bytes, prompt: str) -> str:
        """Call OpenAI GPT-4 Vision API"""
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Track usage
            usage = response.usage
            self.input_tokens += usage.prompt_tokens
            self.output_tokens += usage.completion_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _call_claude(self, image_data: bytes, prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Track usage (approximate for Claude)
            self.input_tokens += len(prompt.split()) + len(image_data) // 1000
            self.output_tokens += len(response.content[0].text.split())
            
            return response.content[0].text
            
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")
    
    def _call_groq(self, image_data: bytes, prompt: str) -> str:
        """Call Groq API with vision model"""
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Call Groq with vision
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=500
            )
            
            # Track usage (approximate)
            self.input_tokens += len(prompt.split()) + len(image_data) // 1000
            self.output_tokens += len(response.choices[0].message.content.split())
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    def _call_groq_text(self, prompt: str) -> str:
        """Call Groq API with text-only model"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=500
            )
            
            self.input_tokens += len(prompt.split())
            self.output_tokens += len(response.choices[0].message.content.split())
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Groq text API error: {str(e)}")
    
    def _call_gemini_text(self, prompt: str) -> str:
        """Call Google Gemini API with text-only input"""
        try:
            response = self.client.generate_content(prompt)
            self.input_tokens += len(prompt.split())
            if response.text:
                self.output_tokens += len(response.text.split())
            return response.text
        except Exception as e:
            raise Exception(f"Gemini text API error: {str(e)}")
    
    def _call_openai_text(self, prompt: str) -> str:
        """Call OpenAI API with text-only input"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI text API error: {str(e)}")
    
    def _call_claude_text(self, prompt: str) -> str:
        """Call Anthropic Claude API with text-only input"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            self.input_tokens += len(prompt.split())
            self.output_tokens += len(response.content[0].text.split())
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Claude text API error: {str(e)}")
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format"""
        try:
            # Try to extract JSON from the response
            import re
            
            # Find JSON-like content between curly braces
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the whole response
                data = json.loads(response_text)
            
            # Ensure all fields exist
            expected_fields = [
                "vendor_name", "invoice_number", "date", 
                "total_amount", "currency", "gst_amount", "gst_rate"
            ]
            
            for field in expected_fields:
                if field not in data:
                    data[field] = None
            
            # Clean up amounts
            if data.get("total_amount"):
                try:
                    data["total_amount"] = float(data["total_amount"])
                except:
                    data["total_amount"] = None
            
            if data.get("gst_amount"):
                try:
                    data["gst_amount"] = float(data["gst_amount"])
                except:
                    data["gst_amount"] = None
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to parse response: {str(e)}\nResponse: {response_text[:200]}")
    
    def calculate_cost(self) -> float:
        """Calculate the cost based on token usage"""
        cost_per_1k = self.model_config["cost_per_1k_tokens"]
        total_tokens = self.input_tokens + self.output_tokens
        self.total_cost = (total_tokens / 1000) * cost_per_1k
        return self.total_cost


def extract_from_all_images(model_key: str = "gemini-flash", 
                            images_dir: str = "data/images") -> List[Dict[str, Any]]:
    """Extract information from all images in a directory"""
    
    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return []
    
    print(f"Extracting from {len(image_files)} images using {model_key}...")
    
    extractor = BillExtractor(model_key)
    results = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"   Processing {i}/{len(image_files)}: {image_file.name}")
        
        result = extractor.extract_from_image(str(image_file))
        result["image_name"] = image_file.name
        results.append(result)
        
        # Print success/failure for this image
        if result["success"]:
            print(f"   [OK] Extracted {len(result['extracted_data'])} fields")
        else:
            print(f"   [FAIL] Failed: {result['error'][:50] if result['error'] else 'Unknown error'}")
        
        # Small delay to avoid rate limits
        time.sleep(1)
    
    # Calculate total cost
    total_cost = extractor.calculate_cost()
    
    success_count = sum(1 for r in results if r['success'])
    print(f"\nExtraction complete!")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Success rate: {success_count}/{len(results)}")
    
    return results


if __name__ == "__main__":
    # Test extraction on one image
    test_model = "gemini-flash"
    print(f"Testing extraction with {test_model}...")
    
    results = extract_from_all_images(test_model)
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"extraction_results_{test_model}_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Show a sample successful extraction
    successful = [r for r in results if r['success']]
    if successful:
        print("\nSample extracted data:")
        sample = successful[0]
        print(f"   Image: {sample['image_name']}")
        print(f"   Extracted: {json.dumps(sample['extracted_data'], indent=2)}")