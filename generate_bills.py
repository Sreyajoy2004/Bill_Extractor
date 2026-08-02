import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
from pathlib import Path

class SyntheticBillGenerator:
    """Generate realistic handwritten-style bills with ground truth data"""
    
    def __init__(self, output_dir="data/images", ground_truth_dir="data/ground_truth"):
        self.output_dir = Path(output_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample data for variety
        self.vendors = [
            "Krishna General Store", "Sai Mart", "Om Traders", "Rajesh Grocery",
            "Priya Supermarket", "Vinayak Stores", "Laxmi General Store",
            "Ganesh Trading Co", "Durga Kirana", "Santosh Bakery",
            "Annapurna Grocery", "Shri Ram Stores", "Mohan & Sons",
            "Ravi Traders", "Suresh Provision Store", "Nandini Dairy",
            "Kumar Dry Fruits", "Gupta General Store", "Chandru Vegetables",
            "Muthu Textiles", "Kalyan Jewellers", "Sai Electronics",
            "Vinoth Mobiles", "Kannan Footwear", "Selvi Bakery"
        ]
        
        self.street_names = [
            "MG Road", "Park Street", "Main Bazaar", "Gandhi Nagar",
            "Nehru Street", "Lal Bagh Road", "Commercial Street",
            "Market Road", "Station Road", "College Road"
        ]
        
        self.cities = [
            "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
            "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
        ]
        
        self.currency = "INR"
        
        # Common grocery items
        self.items = [
            ("Rice", 50, 100),
            ("Wheat Flour", 30, 60),
            ("Sugar", 40, 80),
            ("Salt", 10, 30),
            ("Cooking Oil", 100, 200),
            ("Tea", 50, 150),
            ("Coffee", 100, 300),
            ("Milk", 50, 80),
            ("Eggs", 5, 10),
            ("Bread", 20, 40),
            ("Butter", 50, 120),
            ("Cheese", 100, 200),
            ("Yogurt", 30, 60),
            ("Tamarind", 40, 80),
            ("Ginger", 20, 50),
            ("Garlic", 20, 50),
            ("Onions", 20, 40),
            ("Tomatoes", 20, 40),
            ("Potatoes", 15, 30),
            ("Green Chillies", 10, 30),
            ("Coriander", 5, 15),
            ("Mint", 5, 15),
            ("Lemon", 5, 15),
            ("Coconut", 20, 50),
            ("Bananas", 30, 60),
            ("Apples", 80, 150),
            ("Oranges", 50, 100),
            ("Grapes", 60, 120),
            ("Watermelon", 30, 60),
            ("Mango", 80, 200)
        ]
        
    def generate_bill_data(self, bill_num):
        """Generate random bill data with ground truth"""
        
        # Generate random date (last 30 days)
        date_offset = random.randint(1, 30)
        bill_date = datetime.now() - timedelta(days=date_offset)
        date_str = bill_date.strftime("%d/%m/%Y")
        
        # Random vendor
        vendor = random.choice(self.vendors)
        
        # Random address
        address = f"{random.randint(1, 100)}, {random.choice(self.street_names)}, {random.choice(self.cities)}"
        
        # Generate items (2-6 items)
        num_items = random.randint(2, 6)
        selected_items = random.sample(self.items, num_items)
        
        item_list = []
        subtotal = 0
        
        for item_name, min_price, max_price in selected_items:
            quantity = random.randint(1, 5)
            price_per_unit = random.randint(min_price, max_price)
            total_price = quantity * price_per_unit
            
            item_list.append({
                "name": item_name,
                "quantity": quantity,
                "price_per_unit": price_per_unit,
                "total": total_price
            })
            subtotal += total_price
        
        # Calculate taxes (GST)
        gst_rate = random.choice([0, 5, 12, 18, 28])
        gst_amount = round(subtotal * (gst_rate / 100), 2)
        total_amount = round(subtotal + gst_amount, 2)
        
        # Generate invoice number
        invoice_num = f"INV-{random.randint(1000, 9999)}-{random.randint(100, 999)}"
        
        # Random payment method
        payment_method = random.choice(["Cash", "UPI", "Card", "Credit", "Bank Transfer"])
        
        return {
            "bill_number": bill_num,
            "vendor_name": vendor,
            "address": address,
            "date": date_str,
            "invoice_number": invoice_num,
            "items": item_list,
            "subtotal": subtotal,
            "gst_rate": gst_rate,
            "gst_amount": gst_amount,
            "total_amount": total_amount,
            "currency": self.currency,
            "payment_method": payment_method,
            "handwritten": True
        }
    
    def create_handwritten_bill_image(self, bill_data, image_path):
        """Create a realistic handwritten bill image"""
        
        # Create a white background with slight off-white tint
        width = 800
        height = 1100
        image = Image.new('RGB', (width, height), color=(252, 250, 245))
        draw = ImageDraw.Draw(image)
        
        # Simulate handwriting with slight randomness
        y_position = 40
        x_margin = 50
        
        # Add header - Vendor Name (handwritten style)
        draw.text((x_margin + random.randint(-5, 5), y_position), 
                  bill_data["vendor_name"].upper(), 
                  fill=(0, 0, 0))
        y_position += 50
        
        # Add address (smaller, messy)
        draw.text((x_margin + random.randint(-3, 3), y_position), 
                  bill_data["address"], 
                  fill=(80, 80, 80))
        y_position += 35
        
        # Add invoice number and date (with slight rotation effect)
        draw.text((x_margin, y_position), 
                  f"Invoice: {bill_data['invoice_number']}", 
                  fill=(0, 0, 0))
        y_position += 30
        
        draw.text((x_margin, y_position), 
                  f"Date: {bill_data['date']}", 
                  fill=(0, 0, 0))
        y_position += 50
        
        # Add separator line (hand-drawn style)
        for i in range(3):
            draw.line([(x_margin, y_position + i*2), (width - x_margin, y_position + i*2)], 
                      fill=(100, 100, 100), width=1)
        y_position += 30
        
        # Add items (handwritten-style)
        draw.text((x_margin, y_position), "Items:", fill=(0, 0, 0))
        y_position += 35
        
        for item in bill_data["items"]:
            # Each item with slight position randomness
            x_offset = random.randint(-5, 5)
            item_text = f"{item['quantity']} x {item['name']} @ ₹{item['price_per_unit']} = ₹{item['total']}"
            draw.text((x_margin + x_offset, y_position + random.randint(-3, 3)), 
                      item_text, 
                      fill=(0, 0, 0))
            y_position += 30
        
        y_position += 20
        
        # Add separator line
        for i in range(2):
            draw.line([(x_margin, y_position + i*2), (width - x_margin, y_position + i*2)], 
                      fill=(100, 100, 100), width=1)
        y_position += 30
        
        # Add totals
        draw.text((x_margin, y_position), 
                  f"Subtotal: ₹{bill_data['subtotal']:.2f}", 
                  fill=(0, 0, 0))
        y_position += 30
        
        if bill_data['gst_rate'] > 0:
            draw.text((x_margin, y_position), 
                      f"GST ({bill_data['gst_rate']}%): ₹{bill_data['gst_amount']:.2f}", 
                      fill=(0, 0, 0))
            y_position += 30
        
        # Total (highlighted, larger)
        draw.text((x_margin + random.randint(-8, 8), y_position), 
                  f"Total: ₹{bill_data['total_amount']:.2f}", 
                  fill=(0, 0, 0))
        y_position += 50
        
        # Add payment method
        draw.text((x_margin, y_position), 
                  f"Payment: {bill_data['payment_method']}", 
                  fill=(80, 80, 80))
        
        # Add some random "handwritten" marks/scratches for realism
        for _ in range(random.randint(5, 15)):
            x1 = random.randint(x_margin, width - x_margin)
            y1 = random.randint(50, height - 50)
            x2 = x1 + random.randint(-20, 20)
            y2 = y1 + random.randint(-20, 20)
            draw.line([(x1, y1), (x2, y2)], fill=(200, 200, 200), width=1)
        
        # Save the image
        image.save(image_path, quality=85)
    
    def generate_dataset(self, num_bills=15):
        """Generate multiple bills with ground truth"""
        
        print(f"🔄 Generating {num_bills} synthetic handwritten bills...")
        
        for i in range(1, num_bills + 1):
            # Generate bill data
            bill_data = self.generate_bill_data(i)
            
            # Save image
            image_filename = f"bill_{i:03d}.jpg"
            image_path = self.output_dir / image_filename
            self.create_handwritten_bill_image(bill_data, image_path)
            
            # Save ground truth
            ground_truth_filename = f"bill_{i:03d}.json"
            ground_truth_path = self.ground_truth_dir / ground_truth_filename
            
            # Extract fields needed for evaluation
            ground_truth = {
                "image_name": image_filename,
                "vendor_name": bill_data["vendor_name"],
                "invoice_number": bill_data["invoice_number"],
                "date": bill_data["date"],
                "total_amount": bill_data["total_amount"],
                "currency": bill_data["currency"],
                "gst_amount": bill_data["gst_amount"],
                "gst_rate": bill_data["gst_rate"],
                "items_count": len(bill_data["items"]),
                "payment_method": bill_data["payment_method"],
                "source": "synthetic_handwritten"
            }
            
            with open(ground_truth_path, 'w') as f:
                json.dump(ground_truth, f, indent=2)
            
            print(f"✅ Generated bill_{i:03d}.jpg with ground truth")
        
        print(f"\n🎉 Successfully generated {num_bills} bills!")
        print(f"   Images saved to: {self.output_dir}")
        print(f"   Ground truth saved to: {self.ground_truth_dir}")
        
        return num_bills

if __name__ == "__main__":
    # Generate 15 bills
    generator = SyntheticBillGenerator()
    generator.generate_dataset(15)
    
    print("\n📋 Next Steps:")
    print("1. Check the images in data/images/ folder")
    print("2. Check ground truth JSON files in data/ground_truth/")
    print("3. Run the evaluation pipeline when ready")