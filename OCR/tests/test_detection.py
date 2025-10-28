import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detect import CadastralMapExtractor
import cv2
import numpy as np

class TestCadastralMapExtractor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.extractor = CadastralMapExtractor()
        # Use absolute path for test image
        self.test_image_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets", "raw_maps", "M1.png"
        )
        print(f"Test image path: {self.test_image_path}")
        
        # Ensure the test image exists
        if not os.path.exists(self.test_image_path):
            raise FileNotFoundError(f"Test image not found at: {self.test_image_path}")
    
    def test_initialization(self):
        """Test if the extractor initializes properly"""
        self.assertIsNotNone(self.extractor)
        self.assertIsNotNone(self.extractor.reader)
    
    def test_image_loading(self):
        """Test if the system can load and process images"""
        # Verify file exists before loading
        self.assertTrue(os.path.exists(self.test_image_path), 
                       f"Test image not found at: {self.test_image_path}")
        
        img = cv2.imread(self.test_image_path)
        self.assertIsNotNone(img, "Failed to load test image")
        
        # Test image resizing
        resized = self.extractor._resize_if_needed(img)
        self.assertLessEqual(max(resized.shape[:2]), self.extractor.MAX_IMAGE_SIZE)
    
    def test_color_detection(self):
        """Test color-based feature detection"""
        # Verify file exists before loading
        self.assertTrue(os.path.exists(self.test_image_path),
                       f"Test image not found at: {self.test_image_path}")
        
        img = cv2.imread(self.test_image_path)
        self.assertIsNotNone(img, "Failed to load test image")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Test red detection (transport)
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        self.assertTrue(np.any(red_mask > 0), "No red features detected")
        
        # Test blue detection (water)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        self.assertTrue(np.any(blue_mask > 0), "No blue features detected")

if __name__ == '__main__':
    unittest.main() 