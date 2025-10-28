import cv2
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class DetectionValidator:
    def __init__(self, ground_truth_path=None):
        """Initialize the validator with ground truth data"""
        if ground_truth_path is None:
            # Get the project root directory (parent of src)
            project_root = Path(__file__).resolve().parent.parent
            ground_truth_path = project_root / "datasets" / "symbols" / "annotations.json"
            
        self.ground_truth_path = Path(ground_truth_path)
        self.load_ground_truth()
        
        # Create results directory relative to project root
        self.results_dir = project_root / "results" / "validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_ground_truth(self):
        """Load ground truth annotations"""
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")
            
        with open(self.ground_truth_path) as f:
            self.ground_truth = json.load(f)
            
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
        
    def validate_detection(self, detected_symbols, category):
        """Validate detection results for a specific category"""
        ground_truth_symbols = self.ground_truth['symbols'].get(category, [])
        
        if not ground_truth_symbols or not detected_symbols:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'average_iou': 0.0,
                'total_detected': len(detected_symbols),
                'total_ground_truth': len(ground_truth_symbols)
            }
            
        true_positives = 0
        total_iou = 0
        matched_gt = set()
        
        # Match detected symbols to ground truth
        for det in detected_symbols:
            best_iou = 0
            best_gt_idx = None
            
            # Get detection bbox and area
            det_bbox = det.get('bbox', None)
            det_area = det.get('area', None)
            
            if not det_bbox:
                continue
                
            for idx, gt in enumerate(ground_truth_symbols):
                if idx in matched_gt:
                    continue
                
                gt_bbox = gt.get('bbox', None)
                gt_area = gt.get('area', None)
                
                if not gt_bbox:
                    continue
                    
                # Calculate IoU
                iou = self.calculate_iou(det_bbox, gt_bbox)
                
                # Area similarity score (if available)
                area_score = 1.0
                if det_area and gt_area:
                    area_ratio = min(det_area, gt_area) / max(det_area, gt_area)
                    area_score = 0.5 + (0.5 * area_ratio)  # Scale between 0.5 and 1.0
                
                # Adjust IoU by area score
                iou *= area_score
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # Use category-specific IoU thresholds
            iou_threshold = {
                'terrain': 0.2,      # More lenient for numerous terrain features
                'settlements': 0.3,   # Medium threshold for settlements
                'boundaries': 0.25,   # Slightly lenient for boundaries
                'transport': 0.25,    # Slightly lenient for transport features
                'water': 0.3         # Medium threshold for water features
            }.get(category, 0.3)
            
            if best_iou > iou_threshold:
                true_positives += 1
                total_iou += best_iou
                matched_gt.add(best_gt_idx)
        
        # Calculate metrics
        precision = true_positives / len(detected_symbols) if detected_symbols else 0
        recall = true_positives / len(ground_truth_symbols) if ground_truth_symbols else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        average_iou = total_iou / true_positives if true_positives > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'average_iou': average_iou,
            'total_detected': len(detected_symbols),
            'total_ground_truth': len(ground_truth_symbols)
        }
        
    def generate_validation_report(self, all_detections):
        """Generate comprehensive validation report"""
        print("\n📊 Generating Validation Report...")
        report_data = {}
        
        # Validate each category
        for category in self.ground_truth['metadata']['categories']:
            detected = all_detections.get(category, [])
            metrics = self.validate_detection(detected, category)
            report_data[category] = metrics
            
        # Create detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"validation_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
            
        # Generate visualizations
        self.create_validation_visualizations(report_data, timestamp)
        
        return report_data, report_path
        
    def create_validation_visualizations(self, report_data, timestamp):
        """Create visualization of validation results"""
        # Prepare data for plotting
        categories = list(report_data.keys())
        metrics = ['precision', 'recall', 'f1_score', 'average_iou']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detection Validation Results', fontsize=16)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [report_data[cat][metric] * 100 for cat in categories]
            
            # Create bar plot
            bars = ax.bar(categories, values)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Percentage (%)')
            ax.set_ylim(0, 100)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom')
                
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"validation_results_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Create summary table
        summary_data = []
        for category in categories:
            row = {
                'Category': category,
                'Detection Rate': f"{report_data[category]['recall']*100:.1f}%",
                'Accuracy': f"{report_data[category]['precision']*100:.1f}%",
                'F1 Score': f"{report_data[category]['f1_score']*100:.1f}%",
                'Detected/Total': f"{report_data[category]['total_detected']}/{report_data[category]['total_ground_truth']}"
            }
            summary_data.append(row)
            
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.results_dir / f"validation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Get the report path
        report_path = self.results_dir / f"validation_report_{timestamp}.json"
        
        print(f"\n✅ Validation results saved to:")
        print(f"  📊 Metrics: {report_path}")
        print(f"  📈 Visualization: {plot_path}")
        print(f"  📑 Summary: {summary_path}")

def main():
    print("🔍 Starting Detection Validation")
    print("=" * 50)
    
    try:
        # Initialize validator
        validator = DetectionValidator()
        
        # Load latest detection results
        results_path = "cadastral_output.json"
        if not os.path.exists(results_path):
            print(f"❌ No detection results found at: {results_path}")
            return
            
        with open(results_path) as f:
            detection_results = json.load(f)
            
        # Generate validation report
        report_data, report_path = validator.generate_validation_report(detection_results)
        
        # Print summary
        print("\n📋 Validation Summary:")
        print("-" * 30)
        for category, metrics in report_data.items():
            print(f"\n{category.upper()}:")
            print(f"  • Precision: {metrics['precision']*100:.1f}%")
            print(f"  • Recall: {metrics['recall']*100:.1f}%")
            print(f"  • F1 Score: {metrics['f1_score']*100:.1f}%")
            print(f"  • Average IoU: {metrics['average_iou']*100:.1f}%")
            
    except Exception as e:
        print(f"❌ Error during validation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 