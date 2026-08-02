import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from config import Config
from src.extractors.tesseract_extractor import TesseractExtractor
from src.extractors.llm_extractor import BillExtractor
from src.evaluation.evaluator import BillEvaluator


OUTPUTS_DIR = Path("outputs")
COMPARISON_DIR = OUTPUTS_DIR / "comparison"
GROQ_DIR = OUTPUTS_DIR / "groq_results"
OPENAI_DIR = OUTPUTS_DIR / "openai_results"
ANTHROPIC_DIR = OUTPUTS_DIR / "anthropic_results"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
ZOHO_DIR = OUTPUTS_DIR / "zoho"

for d in [COMPARISON_DIR, GROQ_DIR, OPENAI_DIR, ANTHROPIC_DIR, EVAL_DIR, ZOHO_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_output_dir(model_key: str) -> Path:
    provider = Config.MODELS[model_key]["provider"]
    if provider == "groq":
        return GROQ_DIR
    elif provider == "openai":
        return OPENAI_DIR
    elif provider == "anthropic":
        return ANTHROPIC_DIR
    return OUTPUTS_DIR


def run_model(model_key: str, images: List[Path], tesseract: TesseractExtractor, timestamp: str) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"Running: {model_key}")
    print(f"{'='*70}")
    
    llm = BillExtractor(model_key=model_key)
    evaluator = BillEvaluator(ground_truth_dir=str(Config.GROUND_TRUTH_DIR))
    results = []
    
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}", end=" ")
        tesseract_result = tesseract.extract_bill(img_path)
        raw_text = tesseract_result.get("raw_text", "")
        
        if not tesseract_result["success"] or not raw_text:
            print("[FAIL] Tesseract")
            results.append({
                "image_name": img_path.name,
                "image_path": str(img_path),
                "success": False,
                "error": "Tesseract OCR failed",
                "tesseract_raw_text": "",
                "extracted_data": {},
                "stage": "tesseract_failed"
            })
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
    
    out_dir = get_output_dir(model_key)
    extraction_file = out_dir / f"extraction_results_{model_key}_{timestamp}.json"
    with open(extraction_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    evaluation = evaluator.evaluate_batch(results)
    eval_file = EVAL_DIR / f"evaluation_{model_key}_{timestamp}.json"
    evaluator.save_results(evaluation, str(eval_file))
    
    return {
        "model_key": model_key,
        "model_name": llm.model_name,
        "provider": llm.provider,
        "results": results,
        "evaluation": evaluation,
        "extraction_file": str(extraction_file),
        "evaluation_file": str(eval_file),
        "input_tokens": llm.input_tokens,
        "output_tokens": llm.output_tokens
    }


def generate_bar_chart(field: str, data: Dict[str, float], width: int = 40) -> str:
    lines = [f"\n{field.replace('_', ' ').title()}:"]
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    max_val = max(max(v for v in data.values()), 0.001)
    
    for model, acc in sorted_items:
        filled = int((acc / max_val) * width)
        bar = "#" * filled
        pct = f"{acc*100:.1f}%" if acc > 0 else "0.0%"
        lines.append(f"  {model:<25} {bar:<{width}} {pct}")
    
    return "\n".join(lines)


def generate_comparison_report(all_results: List[Dict[str, Any]], timestamp: str) -> str:
    fields = ["vendor_name", "invoice_number", "date", "total_amount", "currency", "gst_amount", "gst_rate"]
    
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("MODEL PERFORMANCE COMPARISON")
    report_lines.append("=" * 90)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Models compared: {len(all_results)}")
    report_lines.append("")
    
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
    
    chart_data = {res["model_name"]: res["evaluation"]["overall_accuracy"] for res in all_results}
    report_lines.append(generate_bar_chart("Overall Accuracy", chart_data))
    
    for field in fields:
        chart_data = {res["model_name"]: res["evaluation"]["field_accuracies"].get(field, 0.0) for res in all_results}
        report_lines.append(generate_bar_chart(field, chart_data))
    
    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("COST vs ACCURACY ANALYSIS")
    report_lines.append("=" * 90)
    report_lines.append(f"{'Model':<25} {'Cost/Bill':<12} {'Accuracy':<12} {'Value Score'}")
    report_lines.append("-" * 90)
    
    scored_models = []
    for res in all_results:
        model_config = Config.MODELS.get(res["model_key"], {})
        cost_per_1k = model_config.get("cost_per_1k_tokens", 0.0)
        total_tokens = res.get("input_tokens", 0) + res.get("output_tokens", 0)
        cost_per_bill = (total_tokens / 1000) * cost_per_1k if total_tokens > 0 else 0.0
        accuracy = res["evaluation"]["overall_accuracy"]
        value_score = (accuracy * 100) / (cost_per_bill * 1000 + 0.001)
        scored_models.append((res, cost_per_bill, accuracy, value_score))
    
    scored_models.sort(key=lambda x: x[3], reverse=True)
    
    for res, cost, acc, score in scored_models:
        stars = "*" * min(5, max(1, int(score / 20)))
        stars = stars.ljust(5, ".")
        cost_str = f"${cost:.6f}" if cost > 0 else "$0.000000"
        report_lines.append(f"{res['model_name']:<25} {cost_str:<12} {acc*100:>6.1f}%     {stars}")
    
    report_lines.append("-" * 90)
    report_lines.append("")
    
    report_lines.append("RECOMMENDATION")
    report_lines.append("-" * 90)
    if scored_models:
        best_value = scored_models[0][0]
        best_val_score = scored_models[0][3]
        best_val_cost = scored_models[0][1]
        best_val_acc = scored_models[0][2]
        report_lines.append(f"Best value: {best_value['model_name']} ({best_value['provider']})")
        report_lines.append(f"  Accuracy: {best_val_acc*100:.1f}%, Cost: ${best_val_cost:.6f}/bill")
    
    if best_model:
        best_res = next(r for r in all_results if r["model_key"] == best_model)
        report_lines.append(f"Highest accuracy: {best_res['model_name']} ({best_res['provider']})")
        report_lines.append(f"  Accuracy: {best_res['evaluation']['overall_accuracy']*100:.1f}%")
    
    free_models = [r for r in all_results if Config.MODELS.get(r["model_key"], {}).get("cost_per_1k_tokens", 0) == 0]
    if free_models:
        best_free = max(free_models, key=lambda x: x["evaluation"]["overall_accuracy"])
        report_lines.append(f"Best free option: {best_free['model_name']} ({best_free['provider']})")
        report_lines.append(f"  Accuracy: {best_free['evaluation']['overall_accuracy']*100:.1f}%")
    
    report_lines.append("")
    report_lines.append("=" * 90)
    
    report_text = "\n".join(report_lines)
    report_file = COMPARISON_DIR / f"comparison_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text


def generate_per_image_report(all_results: List[Dict[str, Any]], timestamp: str) -> Dict[str, Any]:
    from src.evaluation.evaluator import BillEvaluator
    evaluator = BillEvaluator(ground_truth_dir=str(Config.GROUND_TRUTH_DIR))
    images = {}
    for res in all_results:
        model_name = res["model_name"]
        for result in res["results"]:
            img_name = result.get("image_name", result.get("image_path", "unknown"))
            if img_name not in images:
                gt = evaluator.load_ground_truth(img_name)
                gt_data = {}
                if gt:
                    gt_data = {
                        "vendor_name": gt.get("vendor_name"),
                        "invoice_number": gt.get("invoice_number"),
                        "date": gt.get("date"),
                        "total_amount": gt.get("total_amount"),
                        "currency": gt.get("currency"),
                        "gst_amount": gt.get("gst_amount"),
                        "gst_rate": gt.get("gst_rate")
                    }
                images[img_name] = {"ground_truth": gt_data, "models": {}}
            
            if result.get("success") and result.get("extracted_data"):
                pred = result["extracted_data"]
                images[img_name]["models"][model_name] = {
                    "vendor_name": pred.get("vendor_name"),
                    "invoice_number": pred.get("invoice_number"),
                    "date": pred.get("date"),
                    "total_amount": pred.get("total_amount"),
                    "currency": pred.get("currency"),
                    "gst_amount": pred.get("gst_amount"),
                    "gst_rate": pred.get("gst_rate")
                }
    
    detail_file = COMPARISON_DIR / f"per_image_details_{timestamp}.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(images, f, indent=2, ensure_ascii=False)
    
    return images


def push_to_zoho(all_results: List[Dict[str, Any]], timestamp: str) -> Dict[str, Any]:
    try:
        from src.integrations.zoho import ZohoBooksClient
    except ImportError:
        return {"status": "skipped", "reason": "Zoho integration not available"}
    
    zoho = ZohoBooksClient()
    pushed = []
    
    for res in all_results:
        if res["provider"] != "groq":
            continue
        for result in res["results"]:
            if not result.get("success"):
                continue
            data = result.get("extracted_data", {})
            vendor = data.get("vendor_name")
            amount = data.get("total_amount")
            if vendor and amount:
                expense = zoho.create_expense(
                    vendor_name=vendor,
                    amount=float(amount),
                    currency=data.get("currency", "INR"),
                    date=data.get("date"),
                    reference_number=data.get("invoice_number")
                )
                if expense.get("expense_id") and not expense.get("error"):
                    pushed.append({
                        "image": result.get("image_name"),
                        "expense_id": expense.get("expense_id"),
                        "vendor": vendor,
                        "amount": amount
                    })
                else:
                    pushed.append({
                        "image": result.get("image_name"),
                        "error": expense.get("error", "Unknown error"),
                        "status": expense.get("status", "failed")
                    })
    
    zoho_file = ZOHO_DIR / f"zoho_expenses_{timestamp}.json"
    with open(zoho_file, 'w') as f:
        json.dump(pushed, f, indent=2)
    
    return {"status": "completed", "pushed_count": len([p for p in pushed if "expense_id" in p]), "details": pushed}


def generate_final_report(all_results: List[Dict[str, Any]], 
                          zoho_result: Dict[str, Any],
                          timestamp: str) -> str:
    
    scored_models = []
    for res in all_results:
        model_config = Config.MODELS.get(res["model_key"], {})
        cost_per_1k = model_config.get("cost_per_1k_tokens", 0.0)
        total_tokens = res.get("input_tokens", 0) + res.get("output_tokens", 0)
        cost_per_bill = (total_tokens / 1000) * cost_per_1k if total_tokens > 0 else 0.0
        accuracy = res["evaluation"]["overall_accuracy"]
        value_score = (accuracy * 100) / (cost_per_bill * 1000 + 0.001)
        scored_models.append((res, cost_per_bill, accuracy, value_score))
    
    scored_models.sort(key=lambda x: x[3], reverse=True)
    
    best_value = scored_models[0][0] if scored_models else None
    best_accuracy = max(all_results, key=lambda x: x["evaluation"]["overall_accuracy"]) if all_results else None
    free_models = [r for r in all_results if Config.MODELS.get(r["model_key"], {}).get("cost_per_1k_tokens", 0) == 0]
    best_free = max(free_models, key=lambda x: x["evaluation"]["overall_accuracy"]) if free_models else None
    
    report = f"""# Final Recommendation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- Models tested: {len(all_results)}
- Total images: {all_results[0]['evaluation']['total_images'] if all_results else 0}
- Successful extractions: {sum(r['evaluation']['successful_extractions'] for r in all_results)}

## Best Overall Model

"""
    
    if best_value:
        bv = best_value
        report += f"**{bv['model_name']}** ({bv['provider']})\n"
        report += f"- Accuracy: {bv['evaluation']['overall_accuracy']*100:.1f}%\n"
        report += f"- Successful: {bv['evaluation']['successful_extractions']}/{bv['evaluation']['total_images']}\n\n"
    
    report += "## Per-Field Accuracy (Best Model)\n\n"
    if best_value:
        for field in ["vendor_name", "invoice_number", "date", "total_amount", "currency", "gst_amount", "gst_rate"]:
            acc = best_value["evaluation"]["field_accuracies"].get(field, 0.0)
            report += f"- {field}: {acc*100:.1f}%\n"
    
    report += "\n## Cost vs Accuracy\n\n"
    report += "| Model | Cost/Bill | Accuracy | Value Score |\n"
    report += "|-------|-----------|----------|-------------|\n"
    for res, cost, acc, score in scored_models:
        report += f"| {res['model_name']} | ${cost:.6f} | {acc*100:.1f}% | {score:.1f} |\n"
    
    report += "\n## Recommendations\n\n"
    if best_value:
        report += f"- **Best value**: {best_value['model_name']}\n"
    if best_accuracy and best_accuracy != best_value:
        report += f"- **Highest accuracy**: {best_accuracy['model_name']}\n"
    if best_free and best_free != best_value:
        report += f"- **Best free option**: {best_free['model_name']}\n"
    
    report += "\n## Zoho Books Integration\n\n"
    if zoho_result.get("status") == "completed":
        report += f"- Pushed {zoho_result.get('pushed_count', 0)} expenses to Zoho Books\n"
    else:
        report += f"- Status: {zoho_result.get('status', 'unknown')}\n"
        if zoho_result.get("reason"):
            report += f"- Reason: {zoho_result['reason']}\n"
    
    report_file = OUTPUTS_DIR / "final_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def run_comparison(models: Optional[List[str]] = None,
                   images_dir: str = "data/images",
                   tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    
    images_path = Path(images_dir)
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
            "openai-gpt-4o-mini",
            "gpt-4o",
            "claude-haiku",
            "claude-sonnet",
            "claude-opus"
        ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 90)
    print("MULTI-MODEL BILL EXTRACTION COMPARISON")
    print("=" * 90)
    print(f"Models: {', '.join(models)}")
    print(f"Images: {len(image_files)}")
    print(f"Output: {OUTPUTS_DIR}")
    print("=" * 90)
    
    tesseract = TesseractExtractor(tesseract_path=tesseract_path)
    all_results = []
    
    for model_key in models:
        if model_key not in Config.MODELS:
            print(f"\n[WARN] Model '{model_key}' not found in config, skipping...")
            continue
        try:
            result = run_model(model_key, image_files, tesseract, timestamp)
            all_results.append(result)
        except Exception as e:
            print(f"\n[FAIL] Model '{model_key}' failed: {str(e)[:100]}")
    
    print("\n\n")
    
    comparison_text = generate_comparison_report(all_results, timestamp)
    print(comparison_text)
    
    per_image = generate_per_image_report(all_results, timestamp)
    
    zoho_result = push_to_zoho(all_results, timestamp)
    
    final_report = generate_final_report(all_results, zoho_result, timestamp)
    
    comparison_data = {
        "timestamp": timestamp,
        "models_compared": len(all_results),
        "models": [r["model_key"] for r in all_results],
        "results": all_results
    }
    comparison_file = COMPARISON_DIR / f"model_comparison_{timestamp}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison data: {comparison_file}")
    print(f"Per-image details: {COMPARISON_DIR / f'per_image_details_{timestamp}.json'}")
    print(f"Final report: {OUTPUTS_DIR / 'final_report.md'}")
    
    return comparison_data


if __name__ == "__main__":
    import sys
    models = None
    if len(sys.argv) > 1:
        models = sys.argv[1].split(",")
    run_comparison(models=models)
