import cv2
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import precision_recall_fscore_support
from comprehensive_detector import ComprehensiveMapDetector
import matplotlib.pyplot as plt
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self):
        self.detector = ComprehensiveMapDetector()
        self.metrics = {
            'text_detection': {
                'precision': [],
                'recall': [],
                'f1_score': []
            },
            'symbol_detection': {
                'precision': [],
                'recall': [],
                'f1_score': []
            },
            'boundary_detection': {
                'precision': [],
                'recall': [],
                'f1_score': []
            }
        }
    
    def load_ground_truth(self, annotations_file):
        """Load ground truth annotations"""
        with open(annotations_file, 'r', encoding='utf-8') as f:
            raw_annotations = json.load(f)
        
        # Convert to our expected format
        processed_annotations = {}
        image_name = raw_annotations['image_name']
        processed_annotations[image_name] = {
            'text': [],  # We'll add text detection later
            'symbols': [],
            'boundaries': []
        }
        
        # Process symbols
        for category in ['settlements', 'terrain', 'boundaries']:
            if category in raw_annotations['symbols']:
                for symbol in raw_annotations['symbols'][category]:
                    bbox = symbol['bbox']
                    processed_annotations[image_name]['symbols'].append({
                        'location': [bbox[0], bbox[1]],
                        'size': [bbox[2], bbox[3]],
                        'type': category
                    })
        
        return processed_annotations
    
    def calculate_text_accuracy(self, detected_text, ground_truth_text):
        """Calculate accuracy for text detection"""
        # Convert detections to sets for comparison
        detected_set = set((t['text'].lower(), t['type']) for t in detected_text)
        truth_set = set((t['text'].lower(), t['type']) for t in ground_truth_text)
        
        # Calculate metrics
        true_positives = len(detected_set.intersection(truth_set))
        false_positives = len(detected_set - truth_set)
        false_negatives = len(truth_set - detected_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def calculate_symbol_accuracy(self, detected_symbols, ground_truth_symbols, iou_threshold=0.5):
        """Calculate accuracy for symbol detection using IoU"""
        matches = []
        
        for gt_symbol in ground_truth_symbols:
            gt_box = self.get_symbol_box(gt_symbol)
            best_iou = 0
            best_match = None
            
            for det_symbol in detected_symbols:
                det_box = self.get_symbol_box(det_symbol)
                iou = self.calculate_iou(gt_box, det_box)
                
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = det_symbol
            
            if best_match is not None:
                matches.append((gt_symbol, best_match))
        
        precision = len(matches) / len(detected_symbols) if detected_symbols else 0
        recall = len(matches) / len(ground_truth_symbols) if ground_truth_symbols else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def calculate_boundary_accuracy(self, detected_boundaries, ground_truth_boundaries, distance_threshold=10):
        """Calculate accuracy for boundary detection"""
        matches = []
        
        for gt_boundary in ground_truth_boundaries:
            gt_coords = gt_boundary['coordinates']
            best_distance = float('inf')
            best_match = None
            
            for det_boundary in detected_boundaries:
                det_coords = det_boundary['coordinates']
                distance = self.calculate_boundary_distance(gt_coords, det_coords)
                
                if distance < distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = det_boundary
            
            if best_match is not None:
                matches.append((gt_boundary, best_match))
        
        precision = len(matches) / len(detected_boundaries) if detected_boundaries else 0
        recall = len(matches) / len(ground_truth_boundaries) if ground_truth_boundaries else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def get_symbol_box(self, symbol):
        """Convert symbol location and size to box format"""
        loc = symbol['location']
        size = symbol['size']
        return [loc[0], loc[1], loc[0] + size[0], loc[1] + size[1]]
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def calculate_boundary_distance(self, coords1, coords2):
        """Calculate average distance between two boundaries"""
        dist1 = np.sqrt((coords1[0][0] - coords2[0][0])**2 + (coords1[0][1] - coords2[0][1])**2)
        dist2 = np.sqrt((coords1[1][0] - coords2[1][0])**2 + (coords1[1][1] - coords2[1][1])**2)
        return (dist1 + dist2) / 2
    
    def evaluate_model(self, test_images, ground_truth_file):
        """Evaluate model performance on test images"""
        ground_truth = self.load_ground_truth(ground_truth_file)
        results = []
        
        for image_path in tqdm(test_images, desc="Evaluating model"):
            # Get model predictions
            predictions = self.detector.process_image(image_path)
            if predictions and image_path in ground_truth:
                gt = ground_truth[image_path]
                
                # Calculate accuracies
                text_precision, text_recall, text_f1 = self.calculate_text_accuracy(
                    predictions['text'], gt['text']
                )
                symbol_precision, symbol_recall, symbol_f1 = self.calculate_symbol_accuracy(
                    predictions['symbols'], gt['symbols']
                )
                boundary_precision, boundary_recall, boundary_f1 = self.calculate_boundary_accuracy(
                    predictions['boundaries'], gt['boundaries']
                )
                
                # Store metrics
                self.metrics['text_detection']['precision'].append(text_precision)
                self.metrics['text_detection']['recall'].append(text_recall)
                self.metrics['text_detection']['f1_score'].append(text_f1)
                
                self.metrics['symbol_detection']['precision'].append(symbol_precision)
                self.metrics['symbol_detection']['recall'].append(symbol_recall)
                self.metrics['symbol_detection']['f1_score'].append(symbol_f1)
                
                self.metrics['boundary_detection']['precision'].append(boundary_precision)
                self.metrics['boundary_detection']['recall'].append(boundary_recall)
                self.metrics['boundary_detection']['f1_score'].append(boundary_f1)
                
                results.append({
                    'image_path': image_path,
                    'text_metrics': {'precision': text_precision, 'recall': text_recall, 'f1': text_f1},
                    'symbol_metrics': {'precision': symbol_precision, 'recall': symbol_recall, 'f1': symbol_f1},
                    'boundary_metrics': {'precision': boundary_precision, 'recall': boundary_recall, 'f1': boundary_f1}
                })
        
        return results
    
    def plot_metrics(self, save_path='output/metrics'):
        """Plot evaluation metrics"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Plot metrics for each detection type
        for det_type in self.metrics:
            plt.figure(figsize=(10, 6))
            x = range(len(self.metrics[det_type]['precision']))
            
            plt.plot(x, self.metrics[det_type]['precision'], label='Precision')
            plt.plot(x, self.metrics[det_type]['recall'], label='Recall')
            plt.plot(x, self.metrics[det_type]['f1_score'], label='F1 Score')
            
            plt.title(f'{det_type.replace("_", " ").title()} Metrics')
            plt.xlabel('Image Number')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(save_path / f'{det_type}_metrics.png')
            plt.close()
    
    def save_results(self, results, output_file='output/evaluation_results.json'):
        """Save evaluation results to file"""
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Calculate average metrics
        avg_metrics = {
            'text_detection': {
                'avg_precision': np.mean(self.metrics['text_detection']['precision']),
                'avg_recall': np.mean(self.metrics['text_detection']['recall']),
                'avg_f1': np.mean(self.metrics['text_detection']['f1_score'])
            },
            'symbol_detection': {
                'avg_precision': np.mean(self.metrics['symbol_detection']['precision']),
                'avg_recall': np.mean(self.metrics['symbol_detection']['recall']),
                'avg_f1': np.mean(self.metrics['symbol_detection']['f1_score'])
            },
            'boundary_detection': {
                'avg_precision': np.mean(self.metrics['boundary_detection']['precision']),
                'avg_recall': np.mean(self.metrics['boundary_detection']['recall']),
                'avg_f1': np.mean(self.metrics['boundary_detection']['f1_score'])
            }
        }
        
        # Save detailed results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'average_metrics': avg_metrics,
                'detailed_results': results
            }, f, indent=2)

def main():
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Get test images
    image_dir = Path('datasets/images')
    test_images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    print(f"Found {len(test_images)} images to evaluate")
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluator.evaluate_model(
        test_images,
        'datasets/symbols/annotations.json'
    )
    
    if not results:
        print("No results generated. This could be because:")
        print("1. No matching images found between test set and annotations")
        print("2. Error in processing images")
        print("3. No valid predictions generated")
        return
    
    # Plot and save results
    evaluator.plot_metrics()
    evaluator.save_results(results)
    
    # Print summary
    print("\nEvaluation Complete!")
    print(f"\nProcessed {len(results)} images successfully")
    print("\nAverage Metrics:")
    for det_type in evaluator.metrics:
        if evaluator.metrics[det_type]['precision']:  # Only print if we have results
            print(f"\n{det_type.replace('_', ' ').title()}:")
            print(f"Precision: {np.mean(evaluator.metrics[det_type]['precision']):.3f}")
            print(f"Recall: {np.mean(evaluator.metrics[det_type]['recall']):.3f}")
            print(f"F1 Score: {np.mean(evaluator.metrics[det_type]['f1_score']):.3f}")

if __name__ == "__main__":
    main() 