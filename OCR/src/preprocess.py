import cv2
import numpy as np
from typing import Tuple, Dict, Any
import json

class ImagePreprocessor:
    def __init__(self, config_path: str = 'config/detection_params.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.preprocess_config = self.config['preprocessing']
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply a series of image enhancement techniques."""
        # Apply bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(
            image,
            self.preprocess_config['bilateral_filter']['d'],
            self.preprocess_config['bilateral_filter']['sigma_color'],
            self.preprocess_config['bilateral_filter']['sigma_space']
        )
        
        # Convert to LAB color space for CLAHE
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.preprocess_config['clahe']['clip_limit'],
            tileGridSize=tuple(self.preprocess_config['clahe']['tile_grid_size'])
        )
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([cl, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def color_mask(self, image: np.ndarray, color: str) -> np.ndarray:
        """Create precise color masks using both RGB and HSV color spaces."""
        # Get color thresholds
        color_config = self.config['color_thresholds'][color]
        
        # RGB mask
        rgb_mask = cv2.inRange(
            image,
            np.array(color_config['lower']),
            np.array(color_config['upper'])
        )
        
        # HSV mask
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_lower, hsv_upper = np.array(color_config['hsv_range'])
        hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(rgb_mask, hsv_mask)
        
        # Noise reduction
        kernel = np.ones((color_config['noise_reduction'], color_config['noise_reduction']), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        return cleaned_mask
    
    def detect_lines(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced line detection with angle-based filtering."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(
            gray,
            self.config['line_detection']['canny_threshold1'],
            self.config['line_detection']['canny_threshold2']
        )
        
        # Hough transform
        lines = cv2.HoughLinesP(
            edges,
            self.config['line_detection']['rho'],
            self.config['line_detection']['theta'],
            self.config['line_detection']['hough_threshold'],
            minLineLength=self.config['line_detection']['min_line_length'],
            maxLineGap=self.config['line_detection']['max_line_gap']
        )
        
        # Filter lines by angle
        if lines is not None:
            filtered_lines = []
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                
                # Check if angle is significantly different from existing angles
                is_unique = True
                for existing_angle in angles:
                    if np.abs(angle - existing_angle) < self.config['line_detection']['min_angle_diff']:
                        is_unique = False
                        break
                
                if is_unique:
                    filtered_lines.append(line)
                    angles.append(angle)
            
            lines = np.array(filtered_lines)
        
        return edges, {
            'lines': lines,
            'count': len(lines) if lines is not None else 0
        }
    
    def validate_detection(self, image: np.ndarray, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate detection results and calculate confidence scores."""
        validation = {}
        
        # Validate color coverage
        for color in self.config['color_thresholds'].keys():
            mask = self.color_mask(image, color)
            coverage = np.count_nonzero(mask) / mask.size * 100
            expected = self.config['color_thresholds'][color]['coverage']
            validation[f'{color}_accuracy'] = 100 - min(100, abs(coverage - expected) / expected * 100)
        
        # Validate text detection confidence
        if 'text_detections' in results:
            confidences = [det['confidence'] for det in results['text_detections']]
            validation['text_confidence'] = np.mean(confidences) if confidences else 0
        
        # Validate symbol detection
        if 'symbol_detections' in results:
            validation['symbol_confidence'] = np.mean([
                det['confidence'] for det in results['symbol_detections']
            ]) if results['symbol_detections'] else 0
        
        # Calculate overall confidence
        weights = {
            'color': 0.4,
            'text': 0.3,
            'symbol': 0.3
        }
        
        color_accuracy = np.mean([validation[f'{color}_accuracy'] for color in self.config['color_thresholds'].keys()])
        validation['overall_confidence'] = (
            weights['color'] * color_accuracy +
            weights['text'] * validation.get('text_confidence', 0) +
            weights['symbol'] * validation.get('symbol_confidence', 0)
        )
        
        return validation

if __name__ == '__main__':
    # Test the preprocessor
    preprocessor = ImagePreprocessor()
    test_image = cv2.imread('datasets/raw_maps/M1.png')
    if test_image is not None:
        enhanced = preprocessor.enhance_image(test_image)
        cv2.imwrite('output/enhanced_M1.png', enhanced)
        print("Test image enhanced and saved.") 