import cv2
import numpy as np
import pandas as pd
from detect import CadastralMapExtractor
from preprocess import ImagePreprocessor
import os
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class DetectionEvaluator:
    def __init__(self, config_path: str = 'config/detection_params.json'):
        self.extractor = CadastralMapExtractor()
        self.preprocessor = ImagePreprocessor(config_path)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def evaluate_image(self, image_path: str) -> Dict:
        """Evaluate detection accuracy for a single image"""
        try:
            # Process image
            characters, numbers, all_results, symbols = self.extractor.extract_map_text(image_path)
            
            # Load image for preprocessing
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Enhance image
            enhanced = self.preprocessor.enhance_image(image)
            
            # Prepare detection results
            text_detections = []
            for result in all_results:
                if isinstance(result, dict) and 'confidence' in result:
                    text_detections.append({'confidence': float(result['confidence'])})
                else:
                    text_detections.append({'confidence': 0.8})  # Default confidence
            
            symbol_detections = []
            for symbol in symbols:
                if isinstance(symbol, dict):
                    symbol_detections.append({'confidence': float(symbol.get('confidence', 0.8))})
                else:
                    symbol_detections.append({'confidence': 0.8})  # Default confidence
            
            # Get validation metrics
            validation = self.preprocessor.validate_detection(enhanced, {
                'text_detections': text_detections,
                'symbol_detections': symbol_detections
            })
            
            # Calculate detailed metrics
            metrics = {
                'text_count': len(characters) + len(numbers),
                'symbol_count': len(symbols),
                'red_accuracy': float(validation.get('red_accuracy', 0)),
                'blue_accuracy': float(validation.get('blue_accuracy', 0)),
                'green_accuracy': float(validation.get('green_accuracy', 0)),
                'black_accuracy': float(validation.get('black_accuracy', 0)),
                'text_confidence': float(validation.get('text_confidence', 0)),
                'symbol_confidence': float(validation.get('symbol_confidence', 0)),
                'overall_confidence': float(validation.get('overall_confidence', 0))
            }
            
            return metrics
        except Exception as e:
            print(f"Error evaluating {image_path}: {str(e)}")
            # Return default metrics on error
            return {
                'text_count': 0,
                'symbol_count': 0,
                'red_accuracy': 0.0,
                'blue_accuracy': 0.0,
                'green_accuracy': 0.0,
                'black_accuracy': 0.0,
                'text_confidence': 0.0,
                'symbol_confidence': 0.0,
                'overall_confidence': 0.0
            }
    
    def evaluate_dataset(self, image_dir: str) -> Tuple[pd.DataFrame, Dict]:
        """Evaluate detection accuracy across multiple images"""
        results = []
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                print(f"\nEvaluating {filename}...")
                
                metrics = self.evaluate_image(image_path)
                metrics['filename'] = filename
                results.append(metrics)
        
        if not results:
            print("No images were processed successfully.")
            return pd.DataFrame(), {}
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate aggregate statistics
        numeric_cols = ['text_count', 'symbol_count', 'red_accuracy', 'blue_accuracy', 
                       'green_accuracy', 'black_accuracy', 'text_confidence', 
                       'symbol_confidence', 'overall_confidence']
        
        stats = {
            'mean': df[numeric_cols].mean().to_dict(),
            'std': df[numeric_cols].std().to_dict(),
            'min': df[numeric_cols].min().to_dict(),
            'max': df[numeric_cols].max().to_dict()
        }
        
        # Save results
        df.to_csv('output/detection_accuracy.csv', index=False)
        
        try:
            # Create visualization
            self._plot_accuracy_metrics(df)
        except Exception as e:
            print(f"Warning: Could not create visualization: {str(e)}")
        
        return df, stats
    
    def _plot_accuracy_metrics(self, df: pd.DataFrame):
        """Create visualization of accuracy metrics"""
        plt.figure(figsize=(15, 10))
        
        # Color accuracy box plot
        color_metrics = ['red_accuracy', 'blue_accuracy', 'green_accuracy', 'black_accuracy']
        plt.subplot(2, 1, 1)
        df[color_metrics].boxplot()
        plt.title('Color Detection Accuracy Distribution')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        
        # Overall metrics
        plt.subplot(2, 1, 2)
        confidence_metrics = ['text_confidence', 'symbol_confidence', 'overall_confidence']
        df[confidence_metrics].mean().plot(kind='bar')
        plt.title('Average Confidence Scores')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('output/accuracy_analysis.png')
        plt.close()

def main():
    print("Starting Detection System Evaluation...")
    
    evaluator = DetectionEvaluator()
    df, stats = evaluator.evaluate_dataset('datasets/raw_maps')
    
    if not stats:
        print("No results to display.")
        return
    
    print("\nEvaluation Results:")
    print("="*50)
    print("\nAverage Metrics:")
    for metric, value in stats['mean'].items():
        print(f"{metric}: {value:.2f}")
    
    print("\nStandard Deviations:")
    for metric, value in stats['std'].items():
        print(f"{metric}: ±{value:.2f}")
    
    print("\nDetailed results saved to: output/detection_accuracy.csv")
    print("Visualizations saved to: output/accuracy_analysis.png")

if __name__ == '__main__':
    main() 