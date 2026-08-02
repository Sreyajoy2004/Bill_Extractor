import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from config import Config
from src.extractors.tesseract_extractor import TesseractExtractor
from src.extractors.llm_extractor import BillExtractor
from src.evaluation.evaluator import BillEvaluator


def run_single_model(model_key: str,
                     images: List[Path],
                     tesseract: TesseractExtractor,
                     output_path: Path,
                     timestamp: str) -> Dict[str, Any]:
    """Run extraction pipeline for a single model on all images"""
    print(f"\n{'='*70}")
    print(f"Running model: {model_key}")
    print(f"{'='*70}")
    
    llm = BillExtractor(model_key=model_key)
    evaluator = BillEvaluator(ground_truth_dir=str(Config.GROUND_TRUTH_DIR))
    
    results = []
    
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}", end=" ")
        
        tesseract_result = tesseract.extract_bill(img_path)
        raw_text = tesseract_result.get("raw_text", "")
        
        if not tesseract_result["success"] or not raw_text:
            print("[FAIL] Tesseract failed")
            result = {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "success": False,
                "error": "Tesseract OCR failed",
                "tesseract_raw_text": "",
                "extracted_data": {},
                "stage": "tesseract_failed"
            }
            results.append(result)
            continue
        
        llm_result = llm.extract_from_text(raw_text)
        llm_result["image_name"] = img_path.name
        llm_result["image_path"] = str(img_path)
        llm_result["tesseract_raw_text"] = raw_text
        llm_result["stage"] = "completed"
        
        results.append(llm_result)
        
        if llm_result["success"]:
            print("[OK]")
        else:
            print(f"[FAIL] {llm_result.get('error', 'Unknown')[:40]}")
        
        time.sleep(0.3)
    
    extraction_output = output_path / f"extraction_results_{model_key}_{timestamp}.json"
    with open(extraction_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    evaluation = evaluator.evaluate_batch(results)
    
    eval_output = output_path / f"evaluation_{model_key}_{timestamp}.json"
    evaluator.save_results(evaluation, str(eval_output))
    
    return {
        "model_key": model_key,
        "model_name": llm.model_name,
        "provider": llm.provider,
        "results": results,
        "evaluation": evaluation,
        "extraction_file": str(extraction_output),
        "evaluation_file": str(eval_output)
    }


def generate_comparison_report(all_results: List[Dict[str, Any]], 
                               output_path: Path,
                               timestamp: str) -> str:
    """Generate a comprehensive comparison report across all models"""
    
    if not all_results:
        return "No results to compare"
    
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("MULTI-MODEL COMPARISON REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Models compared: {len(all_results)}")
    report_lines.append("")
    
    fields = [
        "vendor_name", "invoice_number", "date",
        "total_amount", "currency", "gst_amount", "gst_rate"
    ]
    
    report_lines.append("PER-FIELD ACCURACY COMPARISON")
    report_lines.append("-" * 90)
    header = f"{'Model':<25} {'Provider':<12}"
    for field in fields:
        header += f" {field:<15}"
    header += f" {'Overall':<10}"
    report_lines.append(header)
    report_lines.append("-" * 90)
    
    best_model = None
    best_accuracy = -1
    
    for res in all_results:
        eval_data = res["evaluation"]
        model_name = res["model_name"]
        provider = res["provider"]
        overall = eval_data["overall_accuracy"]
        
        if overall > best_accuracy:
            best_accuracy = overall
            best_model = res["model_key"]
        
        row = f"{model_name:<25} {provider:<12}"
        for field in fields:
            acc = eval_data["field_accuracies"].get(field, 0.0)
            row += f" {acc*100:>6.1f}%      "
        row += f" {overall*100:>6.1f}%"
        report_lines.append(row)
    
    report_lines.append("-" * 90)
    report_lines.append("")
    
    report_lines.append("COST ANALYSIS")
    report_lines.append("-" * 90)
    for res in all_results:
        from config import Config
        model_config = Config.MODELS.get(res["model_key"], {})
        cost_per_1k = model_config.get("cost_per_1k_tokens", 0.0)
        
        total_input = sum(
            r.get("usage", {}).get("input_tokens", 0) 
            for r in res["results"]
        )
        total_output = sum(
            r.get("usage", {}).get("output_tokens", 0) 
            for r in res["results"]
        )
        total_tokens = total_input + total_output
        estimated_cost = (total_tokens / 1000) * cost_per_1k
        
        report_lines.append(f"  {res['model_name']:<25}")
        report_lines.append(f"    Input tokens:  {total_input}")
        report_lines.append(f"    Output tokens: {total_output}")
        report_lines.append(f"    Total tokens:  {total_tokens}")
        report_lines.append(f"    Cost per 1k:   ${cost_per_1k:.6f}")
        report_lines.append(f"    Est. total:    ${estimated_cost:.4f}")
        report_lines.append("")
    
    report_lines.append("SUCCESS RATES")
    report_lines.append("-" * 90)
    for res in all_results:
        eval_data = res["evaluation"]
        success_rate = (eval_data["successful_extractions"] / eval_data["total_images"] * 100) if eval_data["total_images"] > 0 else 0
        report_lines.append(f"  {res['model_name']:<25} {eval_data['successful_extractions']}/{eval_data['total_images']} ({success_rate:.1f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("RECOMMENDATION")
    report_lines.append("=" * 90)
    
    if best_model:
        best_res = next(r for r in all_results if r["model_key"] == best_model)
        best_eval = best_res["evaluation"]
        report_lines.append(f"Best performing model: {best_res['model_name']} ({best_res['provider']})")
        report_lines.append(f"Overall accuracy: {best_eval['overall_accuracy']*100:.1f}%")
        report_lines.append(f"Successful extractions: {best_eval['successful_extractions']}/{best_eval['total_images']}")
        report_lines.append("")
        
        report_lines.append("Strengths:")
        for field in fields:
            acc = best_eval["field_accuracies"].get(field, 0.0)
            if acc >= 0.8:
                report_lines.append(f"  - {field}: {acc*100:.1f}%")
        
        report_lines.append("")
        report_lines.append("Weaknesses:")
        for field in fields:
            acc = best_eval["field_accuracies"].get(field, 0.0)
            if acc < 0.5:
                report_lines.append(f"  - {field}: {acc*100:.1f}%")
    
    report_lines.append("")
    report_lines.append("=" * 90)
    
    report_text = "\n".join(report_lines)
    
    report_file = output_path / f"comparison_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nComparison report saved to: {report_file}")
    
    return report_text


def run_comparison(models: Optional[List[str]] = None,
                   images_dir: str = "data/images",
                   output_dir: str = "outputs",
                   tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    image_files = sorted(list(images_path.glob("*.jpg")) + 
                         list(images_path.glob("*.jpeg")) + 
                         list(images_path.glob("*.png")))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    if models is None:
        models = [
            "groq-llama-versatile",
            "groq-llama-instant",
            "claude-haiku",
            "gpt-4o",
            "gemini-flash"
        ]
    
    print("=" * 90)
    print("MULTI-MODEL BILL EXTRACTION COMPARISON")
    print("=" * 90)
    print(f"Models: {', '.join(models)}")
    print(f"Images: {len(image_files)}")
    print(f"Output: {output_dir}")
    print("=" * 90)
    
    tesseract = TesseractExtractor(tesseract_path=tesseract_path)
    
    all_results = []
    
    for model_key in models:
        if model_key not in Config.MODELS:
            print(f"\n[WARN] Model '{model_key}' not found in config, skipping...")
            continue
        
        result = run_single_model(
            model_key=model_key,
            images=image_files,
            tesseract=tesseract,
            output_path=output_path,
            timestamp=timestamp
        )
        all_results.append(result)
    
    print("\n\n")
    report_text = generate_comparison_report(all_results, output_path, timestamp)
    print(report_text)
    
    comparison_data = {
        "timestamp": timestamp,
        "models_compared": len(all_results),
        "models": [r["model_key"] for r in all_results],
        "results": all_results
    }
    
    comparison_file = output_path / f"comparison_data_{timestamp}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nDetailed comparison data: {comparison_file}")
    
    return comparison_data


def run_pipeline(model_key: str = "groq-llama-versatile",
                 images_dir: str = "data/images",
                 output_dir: str = "outputs",
                 tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    image_files = sorted(list(images_path.glob("*.jpg")) + 
                         list(images_path.glob("*.jpeg")) + 
                         list(images_path.glob("*.png")))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print("Bill Extraction Pipeline")
    print(f"   Model: {model_key}")
    print(f"   Images: {len(image_files)}")
    print(f"   Output: {output_dir}")
    print("=" * 60)
    
    result = run_single_model(
        model_key=model_key,
        images=image_files,
        tesseract=TesseractExtractor(tesseract_path=tesseract_path),
        output_path=output_path,
        timestamp=timestamp
    )
    
    print(f"\nExtraction results: {result['extraction_file']}")
    print(f"Evaluation report: {result['evaluation_file']}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        models = None
        if len(sys.argv) > 2:
            models = sys.argv[2].split(",")
        run_comparison(models=models)
    else:
        model = sys.argv[1] if len(sys.argv) > 1 else "groq-llama-versatile"
        run_pipeline(model_key=model)
