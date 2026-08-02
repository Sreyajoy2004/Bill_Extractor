import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fuzzywuzzy import fuzz


class BillEvaluator:
    """Evaluate bill extraction results against ground truth"""
    
    def __init__(self, ground_truth_dir: str = "data/ground_truth"):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.fields = [
            "vendor_name", "invoice_number", "date",
            "total_amount", "currency", "gst_amount", "gst_rate"
        ]
    
    def load_ground_truth(self, image_name: str) -> Optional[Dict[str, Any]]:
        gt_path = self.ground_truth_dir / image_name.replace('.jpg', '.json').replace('.png', '.json')
        if not gt_path.exists():
            return None
        with open(gt_path) as f:
            return json.load(f)
    
    def compare_field(self, gt_value, pred_value, field: str) -> Dict[str, Any]:
        result = {
            "field": field,
            "ground_truth": gt_value,
            "predicted": pred_value,
            "exact_match": False,
            "fuzzy_score": 0.0,
            "correct": False
        }
        
        if gt_value is None and pred_value is None:
            result["exact_match"] = True
            result["fuzzy_score"] = 100.0
            result["correct"] = True
            return result
        
        if gt_value is None or pred_value is None:
            result["correct"] = False
            return result
        
        if field in ["total_amount", "gst_amount"]:
            try:
                gt_float = float(gt_value)
                pred_float = float(pred_value)
                result["exact_match"] = abs(gt_float - pred_float) < 0.02
                result["correct"] = result["exact_match"]
                result["fuzzy_score"] = 100.0 if result["exact_match"] else 0.0
            except Exception:
                result["correct"] = False
        elif field == "gst_rate":
            try:
                gt_float = float(gt_value)
                pred_float = float(pred_value)
                result["exact_match"] = abs(gt_float - pred_float) < 0.1
                result["correct"] = result["exact_match"]
                result["fuzzy_score"] = 100.0 if result["exact_match"] else 0.0
            except Exception:
                result["correct"] = False
        elif field == "date":
            result["exact_match"] = str(gt_value).strip() == str(pred_value).strip()
            result["correct"] = result["exact_match"]
            result["fuzzy_score"] = 100.0 if result["exact_match"] else 0.0
        elif field == "currency":
            result["exact_match"] = str(gt_value).upper().strip() == str(pred_value).upper().strip()
            result["correct"] = result["exact_match"]
            result["fuzzy_score"] = 100.0 if result["exact_match"] else 0.0
        elif field == "invoice_number":
            result["exact_match"] = str(gt_value).upper().strip() == str(pred_value).upper().strip()
            result["correct"] = result["exact_match"]
            result["fuzzy_score"] = 100.0 if result["exact_match"] else 0.0
        else:
            gt_str = str(gt_value).lower().strip()
            pred_str = str(pred_value).lower().strip()
            score = fuzz.ratio(gt_str, pred_str)
            result["fuzzy_score"] = score
            result["exact_match"] = gt_str == pred_str
            result["correct"] = score >= 80
        
        return result
    
    def evaluate_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        image_name = prediction.get("image_name", prediction.get("image_path", "unknown"))
        gt = self.load_ground_truth(image_name)
        
        if not gt:
            return {
                "image_name": image_name,
                "error": "Ground truth not found",
                "field_results": {},
                "accuracy": 0.0,
                "fields_correct": 0,
                "fields_total": len(self.fields)
            }
        
        pred_data = prediction.get("extracted_data", {})
        field_results = {}
        fields_correct = 0
        
        for field in self.fields:
            gt_value = gt.get(field)
            pred_value = pred_data.get(field)
            comparison = self.compare_field(gt_value, pred_value, field)
            field_results[field] = comparison
            if comparison["correct"]:
                fields_correct += 1
        
        accuracy = fields_correct / len(self.fields) if self.fields else 0.0
        
        return {
            "image_name": image_name,
            "error": None,
            "field_results": field_results,
            "accuracy": accuracy,
            "fields_correct": fields_correct,
            "fields_total": len(self.fields),
            "success": prediction.get("success", False)
        }
    
    def evaluate_batch(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        total_correct = 0
        total_fields = 0
        field_correct_counts = {f: 0 for f in self.fields}
        success_count = 0
        
        for pred in predictions:
            result = self.evaluate_prediction(pred)
            results.append(result)
            total_correct += result["fields_correct"]
            total_fields += result["fields_total"]
            if result["success"]:
                success_count += 1
            for field in self.fields:
                if result["field_results"].get(field, {}).get("correct", False):
                    field_correct_counts[field] += 1
        
        overall_accuracy = total_correct / total_fields if total_fields > 0 else 0.0
        field_accuracies = {
            field: (field_correct_counts[field] / len(predictions)) if len(predictions) > 0 else 0.0
            for field in self.fields
        }
        
        return {
            "total_images": len(predictions),
            "successful_extractions": success_count,
            "overall_accuracy": overall_accuracy,
            "field_accuracies": field_accuracies,
            "field_correct_counts": field_correct_counts,
            "total_correct": total_correct,
            "total_fields": total_fields,
            "per_image_results": results
        }
    
    def print_report(self, evaluation: Dict[str, Any]):
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)
        
        print(f"\nSUMMARY")
        print(f"   Total images: {evaluation['total_images']}")
        print(f"   Successful extractions: {evaluation['successful_extractions']}")
        print(f"   Overall field accuracy: {evaluation['overall_accuracy']*100:.1f}%")
        print(f"   Fields correct: {evaluation['total_correct']}/{evaluation['total_fields']}")
        
        print(f"\nPER-FIELD ACCURACY")
        print(f"   {'Field':<20} {'Accuracy':<15} {'Correct':<10}")
        print(f"   {'-'*50}")
        for field in self.fields:
            acc = evaluation['field_accuracies'][field]
            correct = evaluation['field_correct_counts'][field]
            total = evaluation['total_images']
            status = "[OK]" if acc >= 0.8 else "[WARN]" if acc >= 0.5 else "[FAIL]"
            print(f"   {field:<20} {acc*100:>6.1f}%       {correct:>3}/{total:<3} {status}")
        
        print(f"\nPER-IMAGE RESULTS")
        print(f"   {'Image':<15} {'Accuracy':<12} {'Fields':<10}")
        print(f"   {'-'*45}")
        for img_result in evaluation['per_image_results']:
            name = img_result['image_name']
            acc = img_result['accuracy']
            correct = img_result['fields_correct']
            total = img_result['fields_total']
            status = "[OK]" if acc >= 0.8 else "[WARN]" if acc >= 0.5 else "[FAIL]"
            print(f"   {name:<15} {acc*100:>6.1f}%      {correct:>2}/{total:<2} {status}")
    
    def save_results(self, evaluation: Dict[str, Any], output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
