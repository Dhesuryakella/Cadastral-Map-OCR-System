import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from tqdm import tqdm

class MapFeatureExtractor:
    def __init__(self):
        self.symbol_categories = ['water', 'terrain', 'transport', 'settlements', 'boundaries']
        self.load_symbol_templates()
        
    def load_symbol_templates(self):
        """Load symbol templates for each category"""
        self.templates = {}
        symbols_dir = Path("datasets/symbols")
        
        for category in self.symbol_categories:
            self.templates[category] = []
            category_dir = symbols_dir / category
            if category_dir.exists():
                for symbol_path in category_dir.glob("*.png"):
                    template = cv2.imread(str(symbol_path))
                    if template is not None:
                        self.templates[category].append({
                            'name': symbol_path.stem,
                            'template': template
                        })
                print(f"Loaded {len(self.templates[category])} {category} templates")
                
    def extract_features(self, image):
        """Extract features from an image"""
        features = {
            'color_features': self._extract_color_features(image),
            'texture_features': self._extract_texture_features(image),
            'symbol_features': self._extract_symbol_features(image),
            'line_features': self._extract_line_features(image)
        }
        return features
        
    def _extract_color_features(self, image):
        """Extract color-based features"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = {}
        
        # Color distribution
        for category in self.symbol_categories:
            mask = self._get_category_mask(hsv, category)
            features[f'{category}_ratio'] = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
            
        return features
        
    def _extract_texture_features(self, image):
        """Extract texture-based features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = {}
        
        # Basic statistics
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        
        # Gradient features
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_mean'] = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        return features
        
    def _extract_symbol_features(self, image):
        """Extract symbol-based features using template matching"""
        features = {}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for category in self.symbol_categories:
            matches = []
            for template_info in self.templates[category]:
                template = template_info['template']
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # Multi-scale template matching
                for scale in [0.5, 1.0, 2.0]:
                    resized = cv2.resize(template_gray, None, fx=scale, fy=scale)
                    if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
                        continue
                        
                    result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                    matches.extend(result[result > 0.8])
                    
            features[f'{category}_matches'] = len(matches)
            if matches:
                features[f'{category}_max_conf'] = max(matches)
                features[f'{category}_mean_conf'] = np.mean(matches)
            else:
                features[f'{category}_max_conf'] = 0
                features[f'{category}_mean_conf'] = 0
                
        return features
        
    def _extract_line_features(self, image):
        """Extract line-based features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        features = {'total_lines': 0, 'avg_length': 0}
        
        if lines is not None:
            features['total_lines'] = len(lines)
            lengths = []
            angles = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                lengths.append(length)
                angles.append(angle)
                
            features['avg_length'] = np.mean(lengths) if lengths else 0
            features['std_length'] = np.std(lengths) if lengths else 0
            
            # Angle distribution
            angle_bins = [0, 30, 60, 90, 120, 150, 180]
            hist, _ = np.histogram(angles, bins=angle_bins)
            for i, count in enumerate(hist):
                features[f'angle_bin_{i}'] = count / len(angles) if angles else 0
                
        return features
        
    def _get_category_mask(self, hsv, category):
        """Get binary mask for a category based on color ranges"""
        if category == 'water':
            return cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        elif category == 'terrain':
            return cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        elif category == 'transport':
            return cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 50]))
        elif category == 'settlements':
            return cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 100]))
        elif category == 'boundaries':
            return cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

class MapDataset(Dataset):
    def __init__(self, images_dir, feature_extractor):
        self.images_dir = Path(images_dir)
        self.feature_extractor = feature_extractor
        self.image_files = list(self.images_dir.glob("*.jpg"))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        features = self.feature_extractor.extract_features(img)
        return {
            'path': str(img_path),
            'features': features
        }

def train_model():
    """Train the map analysis model"""
    print("\n🎯 Training Map Analysis Model")
    print("=" * 50)
    
    # Initialize feature extractor
    feature_extractor = MapFeatureExtractor()
    
    # Create dataset
    dataset = MapDataset("datasets/images", feature_extractor)
    print(f"\n📚 Loaded {len(dataset)} training images")
    
    # Process images and collect features
    all_features = []
    paths = []
    
    for i in tqdm(range(len(dataset)), desc="Processing images"):
        try:
            data = dataset[i]
            features = data['features']
            
            # Flatten features dictionary
            flat_features = {}
            for category, values in features.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        flat_features[f"{category}_{k}"] = v
                else:
                    flat_features[category] = values
                    
            all_features.append(flat_features)
            paths.append(data['path'])
            
        except Exception as e:
            print(f"\nError processing image {i}: {str(e)}")
            continue
            
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['image_path'] = paths
    
    # Save features
    output_dir = Path("datasets/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_df.to_csv(output_dir / "extracted_features.csv", index=False)
    print(f"\n💾 Saved features to {output_dir / 'extracted_features.csv'}")
    
    # Generate training analysis
    generate_training_analysis(features_df, output_dir)
    print(f"\n📊 Generated training analysis in {output_dir}")
    
def generate_training_analysis(features_df, output_dir):
    """Generate comprehensive training data analysis"""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Feature statistics
    stats = features_df.describe()
    stats.to_csv(output_dir / "feature_statistics.csv")
    
    # 2. Color distribution analysis
    color_features = [col for col in features_df.columns if 'ratio' in col]
    plt.figure(figsize=(12, 6))
    features_df[color_features].boxplot()
    plt.title("Color Distribution Across Maps")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(viz_dir / "color_distributions.png")
    plt.close()
    
    # 3. Symbol detection analysis
    symbol_features = [col for col in features_df.columns if 'matches' in col]
    plt.figure(figsize=(12, 6))
    features_df[symbol_features].boxplot()
    plt.title("Symbol Detection Results")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(viz_dir / "symbol_detections.png")
    plt.close()
    
    # 4. Line feature analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(features_df['line_features_total_lines'], 
                features_df['line_features_avg_length'])
    plt.xlabel("Total Lines")
    plt.ylabel("Average Line Length")
    plt.title("Line Feature Analysis")
    plt.savefig(viz_dir / "line_features.png")
    plt.close()
    
    # 5. Generate summary report
    report = []
    report.append("# Map Analysis Training Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nTotal maps processed: {len(features_df)}")
    
    report.append("\n## Color Distribution Summary")
    for feature in color_features:
        mean = features_df[feature].mean()
        std = features_df[feature].std()
        report.append(f"- {feature}: {mean:.2%} ± {std:.2%}")
        
    report.append("\n## Symbol Detection Summary")
    for feature in symbol_features:
        mean = features_df[feature].mean()
        std = features_df[feature].std()
        report.append(f"- {feature}: {mean:.1f} ± {std:.1f}")
        
    report.append("\n## Line Feature Summary")
    report.append(f"- Average total lines per map: {features_df['line_features_total_lines'].mean():.1f}")
    report.append(f"- Average line length: {features_df['line_features_avg_length'].mean():.1f}")
    
    with open(output_dir / "training_report.md", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    train_model() 