import cv2
import numpy as np
from pathlib import Path
import json
from template_matcher import SymbolTemplateMatcher

class EnhancedMapProcessor:
    """Enhanced map processor using extracted symbol data"""
    
    def __init__(self, symbols_data_path):
        """Initialize with extracted symbols data"""
        print("🔄 Initializing Enhanced Map Processor...")
        
        # Load symbols data
        with open(symbols_data_path, 'r', encoding='utf-8') as f:
            self.symbols_data = json.load(f)
            
        # Load conventional symbols image
        symbols_img_path = Path(symbols_data_path).parent / "Conventional_symbols.jpeg"
        self.symbols_img = cv2.imread(str(symbols_img_path))
        if self.symbols_img is None:
            raise ValueError(f"Could not load symbols image: {symbols_img_path}")
            
        # Extract symbol templates
        self.symbol_templates = self._extract_symbol_templates()
        print(f"✅ Loaded {sum(len(cat) for cat in self.symbol_templates.values())} symbol templates")
        
    def _extract_symbol_templates(self):
        """Extract symbol templates from the symbols image using bbox data"""
        templates = {
            "transport": [],
            "water_features": [],
            "settlements": [],
            "terrain": [],
            "infrastructure": [],
            "boundaries": []
        }
        
        # Process each category
        for category, items in self.symbols_data.items():
            for item in items:
                if item["confidence"] > 0.5:  # Only use high confidence detections
                    # Get symbol region
                    x1, y1, x2, y2 = item["symbol_bbox"]
                    symbol = self.symbols_img[y1:y2, x1:x2].copy()
                    
                    if symbol.size > 0:  # Check if symbol region is valid
                        templates[category].append({
                            "template": symbol,
                            "name": item["text"],
                            "confidence": item["confidence"]
                        })
        
        return templates
    
    def process_map(self, map_path, threshold=0.7):
        """Process a map using the extracted symbol templates"""
        print(f"\n🗺️ Processing map: {map_path}")
        
        # Read target map
        map_img = cv2.imread(str(map_path))
        if map_img is None:
            raise ValueError(f"Could not load map: {map_path}")
            
        # Initialize results
        results = {
            "transport": [],
            "water_features": [],
            "settlements": [],
            "terrain": [],
            "infrastructure": [],
            "boundaries": []
        }
        
        # Process each category
        total_matches = 0
        for category, templates in self.symbol_templates.items():
            print(f"\nProcessing {category} symbols...")
            
            for template_data in templates:
                template = template_data["template"]
                name = template_data["name"]
                
                # Multi-scale template matching
                matches = self._match_template_multi_scale(
                    map_img, template, name, threshold=threshold
                )
                
                if matches:
                    results[category].extend(matches)
                    total_matches += len(matches)
                    print(f"  - Found {len(matches)} matches for {name}")
                    
        print(f"\n✅ Total matches found: {total_matches}")
        return results
    
    def _match_template_multi_scale(self, image, template, symbol_name, threshold=0.7, scale_range=(0.5, 2.0, 5)):
        """Match template at multiple scales"""
        matches = []
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Get scales to try
        scales = np.linspace(scale_range[0], scale_range[1], scale_range[2])
        
        for scale in scales:
            # Resize template
            width = int(template.shape[1] * scale)
            height = int(template.shape[0] * scale)
            if width < 10 or height < 10:  # Skip if too small
                continue
                
            resized = cv2.resize(gray_template, (width, height))
            
            # Template matching
            result = cv2.matchTemplate(gray_img, resized, cv2.TM_CCOEFF_NORMED)
            
            # Get matches above threshold
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                # Check if this match overlaps with existing matches
                new_match = {
                    "location": pt,
                    "scale": scale,
                    "confidence": float(result[pt[1], pt[0]]),
                    "name": symbol_name,
                    "width": width,
                    "height": height
                }
                
                if not self._has_overlap(new_match, matches):
                    matches.append(new_match)
                    
        return matches
    
    def _has_overlap(self, new_match, existing_matches, overlap_thresh=0.3):
        """Check if a new match overlaps with existing matches"""
        x1, y1 = new_match["location"]
        w1, h1 = new_match["width"], new_match["height"]
        rect1 = [x1, y1, x1 + w1, y1 + h1]
        
        for match in existing_matches:
            x2, y2 = match["location"]
            w2, h2 = match["width"], match["height"]
            rect2 = [x2, y2, x2 + w2, y2 + h2]
            
            # Calculate overlap
            x_left = max(rect1[0], rect2[0])
            y_top = max(rect1[1], rect2[1])
            x_right = min(rect1[2], rect2[2])
            y_bottom = min(rect1[3], rect2[3])
            
            if x_right < x_left or y_bottom < y_top:
                continue
                
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = w1 * h1
            area2 = w2 * h2
            overlap = intersection / min(area1, area2)
            
            if overlap > overlap_thresh:
                # Keep the one with higher confidence
                if new_match["confidence"] > match["confidence"]:
                    existing_matches.remove(match)
                    return False
                return True
                
        return False
    
    def visualize_results(self, map_path, results, output_path):
        """Create visualization of detected symbols"""
        img = cv2.imread(str(map_path))
        
        # Color scheme
        colors = {
            "transport": (0, 0, 255),      # Red
            "water_features": (255, 0, 0),  # Blue
            "settlements": (0, 255, 0),     # Green
            "terrain": (128, 128, 0),       # Olive
            "infrastructure": (255, 165, 0), # Orange
            "boundaries": (128, 0, 128)      # Purple
        }
        
        # Draw detections
        for category, matches in results.items():
            color = colors.get(category, (200, 200, 200))
            
            for match in matches:
                x, y = match["location"]
                w, h = match["width"], match["height"]
                
                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{match['name']}"
                cv2.putText(img, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        # Save visualization
        cv2.imwrite(str(output_path), img)
        print(f"\n✅ Visualization saved to: {output_path}")
        
        return img

def main():
    """Main function to process map with enhanced symbol detection"""
    print("🗺️ Enhanced Map Symbol Detector")
    print("=" * 60)
    
    try:
        # Initialize processor with extracted symbols data
        symbols_data_path = r"D:\OCR\datasets\symbols\symbols_data.json"
        processor = EnhancedMapProcessor(symbols_data_path)
        
        # Process M1.png
        map_path = Path(r"D:\OCR\M1.png")
        if not map_path.exists():
            print(f"❌ Error: Could not find map file: {map_path}")
            return
            
        # Detect symbols
        results = processor.process_map(map_path, threshold=0.6)
        
        # Create visualization
        output_path = map_path.with_name(f"{map_path.stem}_enhanced_symbols.jpg")
        processor.visualize_results(map_path, results, output_path)
        
        # Save detailed results
        output_json = map_path.with_name(f"{map_path.stem}_symbol_detections.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ Results saved to: {output_json}")
        print(f"✅ Visualization saved to: {output_path}")
        
        # Print summary
        print("\nDetected Symbols Summary:")
        print("-" * 60)
        total = 0
        for category, matches in results.items():
            if matches:
                print(f"\n{category.upper()}:")
                symbol_counts = {}
                for match in matches:
                    name = match["name"]
                    symbol_counts[name] = symbol_counts.get(name, 0) + 1
                    
                for name, count in symbol_counts.items():
                    print(f"  - {name}: {count} instances")
                    total += count
                    
        print(f"\nTotal symbols detected: {total}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 