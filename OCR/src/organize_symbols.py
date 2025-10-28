import cv2
import numpy as np
from pathlib import Path
import logging

class SymbolOrganizer:
    def __init__(self):
        self.output_dir = Path("datasets/symbols")
        self.categories = ['water', 'terrain', 'transport', 'settlements', 'boundaries']
        
    def setup_directories(self):
        """Create category directories if they don't exist"""
        for category in self.categories:
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
    def extract_symbols(self, template_path):
        """Extract and organize symbols from the template"""
        print(f"Loading template from: {template_path}")
        template = cv2.imread(str(template_path))
        if template is None:
            raise ValueError(f"Could not load template: {template_path}")
            
        print(f"Successfully loaded template with shape: {template.shape}")
        
        # Create category directories
        self.setup_directories()
        
        # Extract symbols for each category
        regions = self._define_regions(template.shape)
        
        for category, region in regions.items():
            print(f"\nProcessing {category} symbols...")
            y1, y2, x1, x2 = region
            section = template[y1:y2, x1:x2]
            
            # Extract individual symbols
            symbols = self._extract_individual_symbols(section, category)
            
            # Save symbols
            for idx, symbol in enumerate(symbols):
                output_path = self.output_dir / category / f"{category}_{idx+1}.png"
                cv2.imwrite(str(output_path), symbol)
                
            print(f"Saved {len(symbols)} {category} symbols")
            
    def _define_regions(self, shape):
        """Define regions for each category in the template"""
        h, w = shape[:2]
        return {
            'water': (0, h//3, 0, w//2),
            'terrain': (h//3, 2*h//3, 0, w//2),
            'transport': (2*h//3, h, 0, w//2),
            'settlements': (0, h//2, w//2, w),
            'boundaries': (h//2, h, w//2, w)
        }
        
    def _extract_individual_symbols(self, section, category):
        """Extract individual symbols from a section"""
        # Convert to grayscale
        gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        symbols = []
        min_area = 100  # Minimum area to consider as a symbol
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Add padding
                pad = 5
                x = max(0, x-pad)
                y = max(0, y-pad)
                w = min(section.shape[1]-x, w+2*pad)
                h = min(section.shape[0]-y, h+2*pad)
                
                symbol = section[y:y+h, x:x+w]
                symbols.append(symbol)
                
        return symbols

def main():
    try:
        organizer = SymbolOrganizer()
        template_path = Path("datasets/symbols/Conventional_symbols.jpeg")
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
            
        organizer.extract_symbols(template_path)
        print("\n✅ Symbol organization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 