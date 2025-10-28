import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
import json

class SymbolTrainer:
    def __init__(self, dataset_path="datasets/images"):
        self.dataset_path = Path(dataset_path)
        self.color_ranges = defaultdict(list)
        self.line_patterns = []
        self.terrain_patterns = []
        
    def train(self):
        """Train on all images in the dataset"""
        print("\n🎯 Starting symbol training...")
        print("=" * 50)
        
        # Load and process each image
        for img_path in self.dataset_path.glob("*.jpg"):
            print(f"\nProcessing: {img_path.name}")
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Could not load {img_path}")
                    continue
                    
                # Extract color ranges and patterns
                self._extract_features(img)
                print(f"✓ Features extracted from {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
                
        # Process collected data
        self._process_collected_data()
        
        # Save trained data
        self._save_trained_data()
        
    def _extract_features(self, img):
        """Extract features from a single image"""
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract water features (blue regions)
        self._extract_water_features(hsv)
        
        # Extract terrain features (brown/green regions)
        self._extract_terrain_features(hsv)
        
        # Extract transport features (lines)
        self._extract_transport_features(img)
        
    def _extract_water_features(self, hsv):
        """Extract water features using color clustering"""
        # Create mask for blue-ish pixels
        blue_mask = cv2.inRange(hsv, np.array([90, 20, 20]), np.array([140, 255, 255]))
        
        if cv2.countNonZero(blue_mask) > 100:
            # Get blue pixels
            blue_pixels = hsv[blue_mask > 0]
            
            if len(blue_pixels) > 0:
                # Cluster the colors
                kmeans = KMeans(n_clusters=min(3, len(blue_pixels)), random_state=42)
                kmeans.fit(blue_pixels)
                
                # Store the cluster centers
                for center in kmeans.cluster_centers_:
                    self.color_ranges['water'].append({
                        'center': center.tolist(),
                        'range': 15  # Tolerance range
                    })
                    
    def _extract_terrain_features(self, hsv):
        """Extract terrain features using color clustering"""
        # Create masks for terrain colors
        brown_mask = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([30, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 255]))
        
        # Process brown colors
        if cv2.countNonZero(brown_mask) > 100:
            brown_pixels = hsv[brown_mask > 0]
            if len(brown_pixels) > 0:
                kmeans = KMeans(n_clusters=min(3, len(brown_pixels)), random_state=42)
                kmeans.fit(brown_pixels)
                for center in kmeans.cluster_centers_:
                    self.color_ranges['terrain_brown'].append({
                        'center': center.tolist(),
                        'range': 20
                    })
                    
        # Process green colors
        if cv2.countNonZero(green_mask) > 100:
            green_pixels = hsv[green_mask > 0]
            if len(green_pixels) > 0:
                kmeans = KMeans(n_clusters=min(3, len(green_pixels)), random_state=42)
                kmeans.fit(green_pixels)
                for center in kmeans.cluster_centers_:
                    self.color_ranges['terrain_green'].append({
                        'center': center.tolist(),
                        'range': 20
                    })
                    
    def _extract_transport_features(self, img):
        """Extract transport features using line detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Detect edges
        edges = cv2.Canny(gray, 30, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=20)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 50:
                    self.line_patterns.append({
                        'angle': angle,
                        'length': length
                    })
                    
    def _process_collected_data(self):
        """Process and consolidate collected data"""
        print("\nProcessing collected data...")
        
        # Process water colors
        water_ranges = self._consolidate_color_ranges(self.color_ranges['water'])
        
        # Process terrain colors
        terrain_brown = self._consolidate_color_ranges(self.color_ranges['terrain_brown'])
        terrain_green = self._consolidate_color_ranges(self.color_ranges['terrain_green'])
        
        # Process line patterns
        transport_patterns = self._analyze_line_patterns()
        
        # Store processed data
        self.processed_data = {
            'water': water_ranges,
            'terrain': {
                'brown': terrain_brown,
                'green': terrain_green
            },
            'transport': transport_patterns
        }
        
    def _consolidate_color_ranges(self, color_data):
        """Consolidate similar color ranges"""
        if not color_data:
            return []
            
        # Convert to numpy array for easier processing
        centers = np.array([c['center'] for c in color_data])
        
        # Cluster similar colors
        if len(centers) > 1:
            kmeans = KMeans(n_clusters=min(3, len(centers)), random_state=42)
            kmeans.fit(centers)
            
            consolidated = []
            for center in kmeans.cluster_centers_:
                consolidated.append({
                    'lower': (center - [15, 50, 50]).clip(0, 255).tolist(),
                    'upper': (center + [15, 50, 50]).clip(0, 255).tolist()
                })
            return consolidated
        else:
            return [{
                'lower': (centers[0] - [15, 50, 50]).clip(0, 255).tolist(),
                'upper': (centers[0] + [15, 50, 50]).clip(0, 255).tolist()
            }]
            
    def _analyze_line_patterns(self):
        """Analyze and categorize line patterns"""
        if not self.line_patterns:
            return {}
            
        angles = [p['angle'] for p in self.line_patterns]
        lengths = [p['length'] for p in self.line_patterns]
        
        return {
            'angles': {
                'horizontal': [0, 10],
                'vertical': [80, 100],
                'diagonal': [35, 55]
            },
            'length': {
                'min': float(np.percentile(lengths, 10)),
                'max': float(np.percentile(lengths, 90))
            }
        }
        
    def _save_trained_data(self):
        """Save trained data to JSON file"""
        output_path = Path("datasets/symbols/trained_symbols.json")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.processed_data, f, indent=4)
            print(f"\n✅ Trained data saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving trained data: {str(e)}")
            
    def get_trained_parameters(self):
        """Return trained parameters for symbol detection"""
        return self.processed_data

if __name__ == "__main__":
    # Create and train the symbol detector
    trainer = SymbolTrainer()
    trainer.train()
    
    # Print summary of trained parameters
    print("\n📊 Training Summary")
    print("=" * 50)
    
    trained_data = trainer.get_trained_parameters()
    
    print("\nWater Detection:")
    print(f"Number of color ranges: {len(trained_data['water'])}")
    
    print("\nTerrain Detection:")
    print(f"Brown ranges: {len(trained_data['terrain']['brown'])}")
    print(f"Green ranges: {len(trained_data['terrain']['green'])}")
    
    print("\nTransport Detection:")
    print("Angle ranges:")
    for angle_type, range_values in trained_data['transport']['angles'].items():
        print(f"- {angle_type}: {range_values[0]}° to {range_values[1]}°")
    print(f"Length range: {trained_data['transport']['length']['min']:.1f} to {trained_data['transport']['length']['max']:.1f} pixels") 