import cv2
import numpy as np
import easyocr
import json
import os
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage
import datetime

class ComprehensiveMapDetector:
    def __init__(self):
        # Initialize OCR with multiple languages (English + any local language if needed)
        self.reader = easyocr.Reader(['en'])
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('map_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load symbol references
        self.symbols_path = Path('datasets/symbols')
        self.conventional_symbols = cv2.imread(str(self.symbols_path / 'Conventional_symbols.jpeg'))
        
        # Load any existing annotations
        self.load_annotations()
        
        # Initialize detection parameters
        self.text_confidence_threshold = 0.5
        self.symbol_match_threshold = 0.8
        self.line_detection_params = {
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 50,
            'minLineLength': 100,
            'maxLineGap': 10
        }
    
    def load_annotations(self):
        """Load existing annotations if available"""
        try:
            with open(self.symbols_path / 'annotations.json', 'r') as f:
                self.annotations = json.load(f)
            self.logger.info(f"Loaded existing annotations for {len(self.annotations)} images")
        except FileNotFoundError:
            self.annotations = {}
            self.logger.info("No existing annotations found. Starting fresh.")
    
    def preprocess_image(self, image):
        """Enhanced preprocessing pipeline"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarization using adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove small noise
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return enhanced, cleaned
    
    def detect_text(self, image):
        """Enhanced text detection with orientation correction"""
        # Detect text orientation
        coords = self.reader.detect(image)
        if coords:
            angles = []
            for box in coords[0]:
                angle = np.arctan2(box[3]-box[1], box[2]-box[0]) * 180/np.pi
                angles.append(angle)
            
            # Get median angle
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                # Rotate image if needed
                rotated = ndimage.rotate(image, -median_angle)
            else:
                rotated = image
        else:
            rotated = image
        
        # Perform text recognition
        results = self.reader.readtext(rotated)
        text_detections = []
        
        for (bbox, text, prob) in results:
            if prob > self.text_confidence_threshold:
                # Clean and normalize text
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    text_detections.append({
                        'text': cleaned_text,
                        'confidence': prob,
                        'bbox': bbox,
                        'type': self.classify_text_type(cleaned_text)
                    })
        
        return text_detections
    
    def clean_text(self, text):
        """Clean and normalize detected text"""
        # Remove unwanted characters
        text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in '.-/')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def classify_text_type(self, text):
        """Enhanced text classification"""
        # Survey number pattern (e.g., numbers with possible prefixes/suffixes)
        if any(c.isdigit() for c in text) and len(text) <= 10:
            return 'survey_number'
        # Village/City names (usually capitalized)
        elif text.isupper() and len(text) > 3:
            return 'place_name'
        # Road names (contains specific keywords)
        elif any(keyword in text.lower() for keyword in ['road', 'street', 'highway', 'path']):
            return 'road_name'
        # Water bodies (contains specific keywords)
        elif any(keyword in text.lower() for keyword in ['river', 'lake', 'pond', 'tank']):
            return 'water_body'
        else:
            return 'general_text'
    
    def detect_boundaries(self, image):
        """Enhanced boundary detection"""
        gray, thresh = self.preprocess_image(image)
        
        # Edge detection with automatic threshold
        sigma = 0.33
        median = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(gray, lower, upper)
        
        # Line detection
        lines = cv2.HoughLinesP(
            edges, 
            self.line_detection_params['rho'],
            self.line_detection_params['theta'],
            self.line_detection_params['threshold'],
            minLineLength=self.line_detection_params['minLineLength'],
            maxLineGap=self.line_detection_params['maxLineGap']
        )
        
        boundaries = []
        if lines is not None:
            # Group similar lines
            grouped_lines = self.group_similar_lines(lines)
            
            for line_group in grouped_lines:
                # Get average line for each group
                avg_line = np.mean(line_group, axis=0)[0]
                x1, y1, x2, y2 = map(int, avg_line)
                
                # Classify boundary type
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
                if abs(angle) < 5 or abs(angle - 180) < 5:
                    boundary_type = 'horizontal'
                elif abs(angle - 90) < 5:
                    boundary_type = 'vertical'
                else:
                    boundary_type = 'diagonal'
                
                boundaries.append({
                    'type': boundary_type,
                    'coordinates': [(x1, y1), (x2, y2)],
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2),
                    'angle': angle
                })
        
        return boundaries
    
    def group_similar_lines(self, lines, angle_threshold=5, distance_threshold=10):
        """Group similar lines together"""
        groups = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180/np.pi
            
            # Try to add to existing group
            added = False
            for group in groups:
                ref_line = group[0]
                rx1, ry1, rx2, ry2 = ref_line[0]
                ref_angle = np.arctan2(ry2-ry1, rx2-rx1) * 180/np.pi
                
                # Check if angles are similar
                if abs(angle - ref_angle) < angle_threshold:
                    # Check if lines are close
                    dist = self.line_distance(line[0], ref_line[0])
                    if dist < distance_threshold:
                        group.append(line)
                        added = True
                        break
            
            # Create new group if not added to existing ones
            if not added:
                groups.append([line])
        
        return groups
    
    def line_distance(self, line1, line2):
        """Calculate minimum distance between two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate midpoints
        mid1 = ((x1 + x2)/2, (y1 + y2)/2)
        mid2 = ((x3 + x4)/2, (y3 + y4)/2)
        
        return np.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)
    
    def detect_symbols(self, image):
        """Enhanced symbol detection"""
        symbols = []
        gray, _ = self.preprocess_image(image)
        
        # Load and process symbol templates
        for category_dir in self.symbols_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                for symbol_file in category_dir.glob('*.jpg'):
                    template = cv2.imread(str(symbol_file), 0)
                    if template is not None:
                        # Multi-scale template matching
                        for scale in np.linspace(0.5, 1.5, 5):
                            resized = cv2.resize(template, None, fx=scale, fy=scale)
                            w, h = resized.shape[::-1]
                            
                            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                            locations = np.where(result >= self.symbol_match_threshold)
                            
                            for pt in zip(*locations[::-1]):
                                # Check for overlapping detections
                                if not self.is_overlapping(pt, w, h, symbols):
                                    symbols.append({
                                        'type': category,
                                        'name': symbol_file.stem,
                                        'location': pt,
                                        'size': (w, h),
                                        'confidence': result[pt[1], pt[0]],
                                        'scale': scale
                                    })
        
        return symbols
    
    def is_overlapping(self, pt, w, h, existing_symbols):
        """Check if a new symbol detection overlaps with existing ones"""
        new_rect = (pt[0], pt[1], pt[0] + w, pt[1] + h)
        
        for symbol in existing_symbols:
            loc = symbol['location']
            size = symbol['size']
            existing_rect = (loc[0], loc[1], loc[0] + size[0], loc[1] + size[1])
            
            # Calculate overlap
            overlap = self.calculate_overlap(new_rect, existing_rect)
            if overlap > 0.5:  # If more than 50% overlap
                return True
        
        return False
    
    def calculate_overlap(self, rect1, rect2):
        """Calculate overlap ratio between two rectangles"""
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        
        return intersection / min(rect1_area, rect2_area)
    
    def process_image(self, image_path):
        """Process a single image and return detections"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to read image: {image_path}")
                return None
            
            # Initialize results
            results = {
                'text': [],
                'symbols': [],
                'boundaries': []
            }
            
            # For now, return empty results (we'll implement detection later)
            print(f"Processing image: {image_path}")
            return results
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def save_results(self, results, image_path):
        """Save detection results"""
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = output_dir / f"{Path(image_path).stem}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved results to {results_file}")
    
    def visualize_results(self, image_path, results):
        """Enhanced visualization of detection results"""
        image = cv2.imread(str(image_path))
        overlay = image.copy()
        
        # Draw text detections
        for text_det in results['text']:
            bbox = text_det['bbox']
            text_type = text_det['type']
            
            # Different colors for different text types
            color = {
                'survey_number': (0, 255, 0),    # Green
                'place_name': (255, 0, 0),       # Blue
                'road_name': (0, 0, 255),        # Red
                'water_body': (255, 255, 0),     # Cyan
                'general_text': (128, 128, 128)  # Gray
            }.get(text_type, (0, 255, 0))
            
            # Draw filled rectangle with transparency
            pts = np.array(bbox, np.int32)
            cv2.fillPoly(overlay, [pts], color)
            
            # Add text label
            cv2.putText(image, 
                      f"{text_det['text']} ({text_type})",
                      (int(bbox[0][0]), int(bbox[0][1] - 5)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Apply transparency
        alpha = 0.3
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Draw boundaries
        for boundary in results['boundaries']:
            pts = boundary['coordinates']
            color = {
                'horizontal': (0, 0, 255),    # Red
                'vertical': (255, 0, 0),      # Blue
                'diagonal': (0, 255, 0)       # Green
            }.get(boundary['type'], (0, 0, 255))
            
            cv2.line(image, 
                    (int(pts[0][0]), int(pts[0][1])), 
                    (int(pts[1][0]), int(pts[1][1])), 
                    color, 2)
        
        # Draw symbols
        for symbol in results['symbols']:
            loc = symbol['location']
            size = symbol['size']
            
            # Draw bounding box
            cv2.rectangle(image,
                        (int(loc[0]), int(loc[1])),
                        (int(loc[0] + size[0]), int(loc[1] + size[1])),
                        (255, 0, 255), 2)  # Magenta
            
            # Add symbol label
            cv2.putText(image,
                      f"{symbol['type']}/{symbol['name']}",
                      (int(loc[0]), int(loc[1] - 5)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Save visualization
        output_dir = Path('output/visualizations')
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{Path(image_path).stem}_annotated.jpg"
        cv2.imwrite(str(output_path), image)
        
        self.logger.info(f"Saved visualization to {output_path}")

def main():
    # Initialize detector
    detector = ComprehensiveMapDetector()
    
    # Process all images in the dataset
    image_dir = Path('datasets/images')
    total_images = len(list(image_dir.glob('*.jpg')))
    
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for image_path in image_dir.glob('*.jpg'):
            try:
                results = detector.process_image(image_path)
                if results:
                    detector.visualize_results(image_path, results)
                pbar.update(1)
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                pbar.update(1)
                continue

if __name__ == "__main__":
    main() 