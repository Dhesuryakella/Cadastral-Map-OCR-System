import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from src.detect import CadastralMapExtractor

class TrainingDataExtractor:
    def __init__(self, images_dir="datasets/images", output_dir="datasets/training_data"):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.extractor = CadastralMapExtractor()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        (self.output_dir / "symbols").mkdir(exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)
        
    def extract_training_data(self):
        """Extract training data from all maps"""
        print("🔍 Extracting training data from topographical maps...")
        
        all_text_data = []
        all_symbols_data = []
        all_features_data = []
        
        # Process each map
        for img_path in self.images_dir.glob("*.jpg"):
            print(f"\nProcessing: {img_path.name}")
            try:
                # Extract data using our existing extractor
                characters, numbers, all_results, symbols = self.extractor.extract_map_text(str(img_path))
                
                # Save text data
                text_data = {
                    "map_name": img_path.name,
                    "characters": characters,
                    "numbers": numbers
                }
                all_text_data.append(text_data)
                
                # Save symbol data
                symbols_data = {
                    "map_name": img_path.name,
                    "symbols": symbols
                }
                all_symbols_data.append(symbols_data)
                
                # Extract and save features
                img = cv2.imread(str(img_path))
                features = self._extract_map_features(img)
                features["map_name"] = img_path.name
                all_features_data.append(features)
                
                print(f"✅ Successfully processed {img_path.name}")
                
            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {str(e)}")
                continue
        
        # Save all extracted data
        self._save_training_data(all_text_data, all_symbols_data, all_features_data)
        
    def _extract_map_features(self, img):
        """Extract additional features from map"""
        features = {
            "color_distribution": self._analyze_color_distribution(img),
            "line_features": self._analyze_line_features(img),
            "texture_features": self._analyze_texture_features(img)
        }
        return features
    
    def _analyze_color_distribution(self, img):
        """Analyze color distribution for different map features"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Color ranges for different features
        color_ranges = {
            "red": [(0, 50, 50), (10, 255, 255)],     # Transport
            "blue": [(100, 50, 50), (130, 255, 255)], # Water
            "green": [(40, 40, 40), (80, 255, 255)],  # Boundaries
            "black": [(0, 0, 0), (180, 50, 100)]      # Settlements
        }
        
        distributions = {}
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            distribution = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
            distributions[color] = float(distribution)
            
        return distributions
    
    def _analyze_line_features(self, img):
        """Analyze line features in the map"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return {"total_lines": 0, "avg_length": 0}
            
        # Calculate line statistics
        lengths = []
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            lengths.append(length)
            angles.append(angle)
            
        return {
            "total_lines": len(lines),
            "avg_length": float(np.mean(lengths)) if lengths else 0,
            "angle_distribution": self._analyze_angle_distribution(angles)
        }
    
    def _analyze_angle_distribution(self, angles):
        """Analyze distribution of line angles"""
        if not angles:
            return {}
            
        # Group angles into 30-degree bins
        bins = {}
        for angle in angles:
            bin_key = int(angle // 30) * 30
            bins[bin_key] = bins.get(bin_key, 0) + 1
            
        # Convert to percentages
        total = sum(bins.values())
        return {f"{k}_{k+30}": v/total for k, v in bins.items()}
    
    def _analyze_texture_features(self, img):
        """Extract texture features using GLCM"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic texture statistics
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        entropy = float(cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().std())
        
        return {
            "mean_intensity": mean,
            "std_intensity": std,
            "entropy": entropy
        }
    
    def _save_training_data(self, text_data, symbols_data, features_data):
        """Save extracted training data"""
        print("\n💾 Saving training data...")
        
        # Save text data
        text_file = self.output_dir / "text" / "text_annotations.json"
        with open(text_file, 'w') as f:
            json.dump(text_data, f, indent=2)
        print(f"✅ Text data saved to {text_file}")
        
        # Save symbols data
        symbols_file = self.output_dir / "symbols" / "symbols_annotations.json"
        with open(symbols_file, 'w') as f:
            json.dump(symbols_data, f, indent=2)
        print(f"✅ Symbols data saved to {symbols_file}")
        
        # Save features data
        features_file = self.output_dir / "features" / "features_data.json"
        with open(features_file, 'w') as f:
            json.dump(features_data, f, indent=2)
        print(f"✅ Features data saved to {features_file}")
        
        # Create summary DataFrame
        summary = []
        for td, sd, fd in zip(text_data, symbols_data, features_data):
            summary.append({
                "map_name": td["map_name"],
                "num_characters": len(td["characters"]),
                "num_numbers": len(td["numbers"]),
                "num_symbols": sum(len(symbols) for symbols in sd["symbols"].values()),
                "color_distribution": fd["color_distribution"]
            })
            
        summary_df = pd.DataFrame(summary)
        summary_file = self.output_dir / "training_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary saved to {summary_file}")

def main():
    print("🎯 Topographical Map Training Data Extractor")
    print("="*50)
    
    extractor = TrainingDataExtractor()
    extractor.extract_training_data()
    
    print("\n✨ Training data extraction complete!")
    print("You can now use this data to train/fine-tune the model.")

if __name__ == "__main__":
    main() 