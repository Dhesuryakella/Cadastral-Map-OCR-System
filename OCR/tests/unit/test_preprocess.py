import unittest
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocess import ImagePreprocessor

class TestImagePreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = ImagePreprocessor()
        cls.test_image = cv2.imread('datasets/raw_maps/M1.png')
        if cls.test_image is None:
            raise ValueError("Test image could not be loaded")
    
    def test_enhance_image(self):
        """Test image enhancement functionality"""
        enhanced = self.preprocessor.enhance_image(self.test_image)
        
        # Check that enhancement preserves image dimensions
        self.assertEqual(enhanced.shape, self.test_image.shape)
        
        # Check that enhancement increases contrast
        orig_std = np.std(cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY))
        enhanced_std = np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        self.assertGreater(enhanced_std, orig_std)
    
    def test_color_mask(self):
        """Test color masking for each color channel"""
        for color in ['red', 'blue', 'green', 'black']:
            mask = self.preprocessor.color_mask(self.test_image, color)
            
            # Check mask properties
            self.assertEqual(mask.dtype, np.uint8)
            self.assertEqual(mask.shape[:2], self.test_image.shape[:2])
            
            # Check if mask contains any detections
            self.assertGreater(np.count_nonzero(mask), 0)
    
    def test_detect_lines(self):
        """Test line detection functionality"""
        edges, line_info = self.preprocessor.detect_lines(self.test_image)
        
        # Check edge detection
        self.assertEqual(edges.shape[:2], self.test_image.shape[:2])
        self.assertGreater(np.count_nonzero(edges), 0)
        
        # Check line detection
        self.assertIn('lines', line_info)
        self.assertIn('count', line_info)
        if line_info['lines'] is not None:
            self.assertGreater(line_info['count'], 0)
    
    def test_validate_detection(self):
        """Test detection validation"""
        # Create mock detection results
        mock_results = {
            'text_detections': [
                {'confidence': 0.9},
                {'confidence': 0.85}
            ],
            'symbol_detections': [
                {'confidence': 0.88},
                {'confidence': 0.92}
            ]
        }
        
        validation = self.preprocessor.validate_detection(self.test_image, mock_results)
        
        # Check validation metrics
        self.assertIn('overall_confidence', validation)
        self.assertGreaterEqual(validation['overall_confidence'], 0)
        self.assertLessEqual(validation['overall_confidence'], 100)
        
        # Check color accuracies
        for color in ['red', 'blue', 'green', 'black']:
            self.assertIn(f'{color}_accuracy', validation)
            self.assertGreaterEqual(validation[f'{color}_accuracy'], 0)
            self.assertLessEqual(validation[f'{color}_accuracy'], 100)

if __name__ == '__main__':
    unittest.main() 