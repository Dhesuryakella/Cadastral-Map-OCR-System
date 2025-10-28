import cv2
import numpy as np
import easyocr
from pathlib import Path
import json

class SymbolTextExtractor:
    """Extract text and symbols from Survey of India conventional symbols image"""
    
    def __init__(self):
        print("🔍 Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'])
        
    def extract_text_and_symbols(self, image_path):
        """Extract text and corresponding symbols from the image"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Get dimensions
        h, w = img.shape[:2]
        print(f"Image dimensions: {w}x{h}")
        
        # Extract text
        print("\nExtracting text...")
        results = self.reader.readtext(img)
        
        # Organize results by category
        symbols_data = {
            "transport": [],
            "water_features": [],
            "settlements": [],
            "terrain": [],
            "infrastructure": [],
            "boundaries": []
        }
        
        # Process each detected text
        for bbox, text, conf in results:
            # Convert bbox points to integers
            bbox = [[int(x) for x in point] for point in bbox]
            x_min = min(p[0] for p in bbox)
            y_min = min(p[1] for p in bbox)
            x_max = max(p[0] for p in bbox)
            y_max = max(p[1] for p in bbox)
            
            # Extract symbol region (to the left of text)
            symbol_width = 50  # Approximate width of symbol
            symbol_region = img[y_min:y_max, max(0, x_min-symbol_width):x_min]
            
            # Determine category based on y-position and text content
            category = self._determine_category(text, y_min)
            
            # Save the data
            symbol_data = {
                "text": text,
                "confidence": float(conf),
                "text_bbox": bbox,
                "symbol_bbox": [max(0, x_min-symbol_width), y_min, x_min, y_max],
                "y_position": y_min
            }
            
            if category:
                symbols_data[category].append(symbol_data)
        
        # Sort each category by y-position
        for category in symbols_data:
            symbols_data[category].sort(key=lambda x: x["y_position"])
        
        return symbols_data
    
    def _determine_category(self, text, y_pos):
        """Determine symbol category based on text content and position"""
        text = text.lower()
        
        # Transport related keywords
        if any(keyword in text for keyword in ["highway", "road", "track", "railway", "bridge"]):
            return "transport"
            
        # Water features
        if any(keyword in text for keyword in ["stream", "river", "canal", "dam", "tank", "well"]):
            return "water_features"
            
        # Settlements
        if any(keyword in text for keyword in ["village", "town", "hut", "temple", "church", "mosque"]):
            return "settlements"
            
        # Terrain features
        if any(keyword in text for keyword in ["contour", "rock", "sand", "cliff", "grass", "scrub"]):
            return "terrain"
            
        # Infrastructure
        if any(keyword in text for keyword in ["post", "hospital", "school", "power", "mine"]):
            return "infrastructure"
            
        # Boundaries
        if any(keyword in text for keyword in ["boundary", "international", "state", "district"]):
            return "boundaries"
            
        # If no category matched, try to determine by y-position
        if y_pos < 200:
            return "transport"
        elif y_pos < 400:
            return "water_features"
        elif y_pos < 600:
            return "settlements"
        else:
            return "infrastructure"
    
    def visualize_results(self, image_path, symbols_data, output_path):
        """Visualize detected text and symbols"""
        img = cv2.imread(str(image_path))
        
        # Color scheme for different categories
        colors = {
            "transport": (0, 0, 255),      # Red
            "water_features": (255, 0, 0),  # Blue
            "settlements": (0, 255, 0),     # Green
            "terrain": (128, 128, 0),       # Olive
            "infrastructure": (255, 165, 0), # Orange
            "boundaries": (128, 0, 128)      # Purple
        }
        
        # Draw detections for each category
        for category, items in symbols_data.items():
            color = colors.get(category, (200, 200, 200))
            
            for item in items:
                # Draw text bounding box
                pts = np.array(item["text_bbox"], np.int32)
                cv2.polylines(img, [pts], True, color, 2)
                
                # Draw symbol bounding box
                x1, y1, x2, y2 = item["symbol_bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add text label
                cv2.putText(img, f"{category}: {item['text']}", 
                           (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
        
        # Save visualization
        cv2.imwrite(str(output_path), img)
        print(f"\n✅ Visualization saved to: {output_path}")

def main():
    """Main function to extract text and symbols"""
    print("🗺️ Survey of India Conventional Symbols Extractor")
    print("=" * 60)
    
    # Initialize extractor
    extractor = SymbolTextExtractor()
    
    # Process conventional symbols image
    image_path = Path(r"D:\OCR\datasets\symbols\Conventional_symbols.jpeg")
    if not image_path.exists():
        print(f"❌ Error: Could not find symbols file: {image_path}")
        return
        
    try:
        # Extract text and symbols
        symbols_data = extractor.extract_text_and_symbols(image_path)
        
        # Save results to JSON
        output_json = image_path.with_name("symbols_data.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(symbols_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Extracted data saved to: {output_json}")
        
        # Create visualization
        output_viz = image_path.with_name("symbols_visualization.jpg")
        extractor.visualize_results(image_path, symbols_data, output_viz)
        
        # Print summary
        print("\nExtracted Symbols Summary:")
        print("-" * 60)
        for category, items in symbols_data.items():
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"  - {item['text']} (confidence: {item['confidence']:.2f})")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 