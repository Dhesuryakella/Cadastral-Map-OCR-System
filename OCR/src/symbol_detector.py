import cv2
import numpy as np
import os
from pathlib import Path
import json
import logging

class SymbolDetector:
    def __init__(self):
        try:
            # Get absolute path to the symbols file
            self.symbols_path = Path("datasets/symbols/Conventional_symbols.jpeg").resolve()
            logging.info(f"Attempting to load symbols from: {self.symbols_path}")
            
            if not self.symbols_path.exists():
                raise FileNotFoundError(f"Symbols file not found at: {self.symbols_path}")
                
            if not self.symbols_path.is_file():
                raise ValueError(f"Path exists but is not a file: {self.symbols_path}")
                
            # Try to load the template
            template = cv2.imread(str(self.symbols_path))
            if template is None:
                raise ValueError(f"OpenCV failed to load image from: {self.symbols_path}")
                
            logging.info(f"Successfully loaded template with shape: {template.shape}")
            
            self.reference_symbols = self._extract_reference_symbols()
            self.trained_params = self._load_trained_parameters()
            
        except Exception as e:
            logging.error(f"Failed to initialize SymbolDetector: {str(e)}")
            raise
        
    def _extract_reference_symbols(self):
        """Extract reference symbols from the template"""
        try:
            template = cv2.imread(str(self.symbols_path))
            if template is None:
                raise ValueError(f"Could not load template from {self.symbols_path}")
                
            logging.info(f"Extracting reference symbols from template of shape: {template.shape}")
            
            # Define regions for different symbol types
            regions = {
                'water': [(50, 200, 300, 400)],    # Water bodies region
                'terrain': [(300, 200, 600, 400)],  # Mountains and terrain
                'transport': [(50, 0, 300, 200)]    # Roads and transport
            }
            
            symbols = {}
            for category, coords_list in regions.items():
                symbols[category] = []
                for x1, y1, x2, y2 in coords_list:
                    # Validate coordinates
                    if x2 > template.shape[1] or y2 > template.shape[0]:
                        logging.warning(f"Region coordinates ({x1}, {y1}, {x2}, {y2}) exceed template dimensions {template.shape[:2]}")
                        continue
                        
                    region = template[y1:y2, x1:x2]
                    # Convert to grayscale
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    # Threshold to get binary image
                    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    # Find contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    logging.info(f"Found {len(contours)} potential symbols in {category} region")
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:  # Filter small noise
                            x, y, w, h = cv2.boundingRect(contour)
                            symbol = region[y:y+h, x:x+w]
                            symbols[category].append({
                                'image': symbol,
                                'size': (w, h)
                            })
                            
            logging.info(f"Successfully extracted {sum(len(s) for s in symbols.values())} symbols across all categories")
            return symbols
            
        except Exception as e:
            logging.error(f"Error extracting reference symbols: {str(e)}")
            raise
    
    def _load_trained_parameters(self):
        """Load trained parameters from JSON file"""
        trained_file = Path("datasets/symbols/trained_symbols.json")
        if trained_file.exists():
            try:
                with open(trained_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading trained parameters: {str(e)}")
        return None
        
    def detect_symbols(self, image_path):
        """Detect symbols in the input image"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        results = {
            'water': [],
            'terrain': [],
            'transport': []
        }
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Use trained parameters if available
        if self.trained_params:
            # Water detection with trained ranges
            water_mask = np.zeros_like(hsv[:,:,0])
            for water_range in self.trained_params['water']:
                lower = np.array(water_range['lower'])
                upper = np.array(water_range['upper'])
                mask = cv2.inRange(hsv, lower, upper)
                water_mask = cv2.bitwise_or(water_mask, mask)
        else:
            # Fallback to default ranges
            water_masks = []
            light_blue_lower = np.array([90, 30, 30])
            light_blue_upper = np.array([110, 255, 255])
            water_masks.append(cv2.inRange(hsv, light_blue_lower, light_blue_upper))
            
            dark_blue_lower = np.array([110, 50, 30])
            dark_blue_upper = np.array([130, 255, 255])
            water_masks.append(cv2.inRange(hsv, dark_blue_lower, dark_blue_upper))
            
            water_mask = np.zeros_like(water_masks[0])
            for mask in water_masks:
                water_mask = cv2.bitwise_or(water_mask, mask)
                
        # Enhance water detection
        kernel_close = np.ones((5,5), np.uint8)
        kernel_open = np.ones((3,3), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel_close)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Find water contours
        water_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in water_contours:
            area = cv2.contourArea(contour)
            if area > 100:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h if h > 0 else 0
                
                if circularity > 0.6:
                    water_type = 'lake'
                elif aspect_ratio > 3 or aspect_ratio < 0.33:
                    water_type = 'river'
                else:
                    water_type = 'water_body'
                    
                results['water'].append({
                    'type': water_type,
                    'bbox': [x, y, w, h],
                    'area': area,
                    'circularity': circularity
                })
                
        # Terrain detection with trained ranges
        if self.trained_params:
            terrain_mask = np.zeros_like(hsv[:,:,0])
            
            # Brown terrain
            for brown_range in self.trained_params['terrain']['brown']:
                lower = np.array(brown_range['lower'])
                upper = np.array(brown_range['upper'])
                mask = cv2.inRange(hsv, lower, upper)
                terrain_mask = cv2.bitwise_or(terrain_mask, mask)
                
            # Green terrain
            for green_range in self.trained_params['terrain']['green']:
                lower = np.array(green_range['lower'])
                upper = np.array(green_range['upper'])
                mask = cv2.inRange(hsv, lower, upper)
                terrain_mask = cv2.bitwise_or(terrain_mask, mask)
        else:
            # Fallback to default ranges
            terrain_masks = []
            brown_lower = np.array([10, 30, 30])
            brown_upper = np.array([30, 255, 255])
            terrain_masks.append(cv2.inRange(hsv, brown_lower, brown_upper))
            
            green_lower = np.array([40, 30, 30])
            green_upper = np.array([80, 255, 255])
            terrain_masks.append(cv2.inRange(hsv, green_lower, green_upper))
            
            terrain_mask = np.zeros_like(terrain_masks[0])
            for mask in terrain_masks:
                terrain_mask = cv2.bitwise_or(terrain_mask, mask)
                
        # Clean up terrain mask
        kernel = np.ones((5,5), np.uint8)
        terrain_mask = cv2.morphologyEx(terrain_mask, cv2.MORPH_CLOSE, kernel)
        terrain_contours, _ = cv2.findContours(terrain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in terrain_contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                results['terrain'].append({
                    'type': 'terrain_feature',
                    'bbox': [x, y, w, h],
                    'area': cv2.contourArea(contour)
                })
                
        # Enhanced transport feature detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for transport features
        # Use bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Improved edge detection
        edges = cv2.Canny(gray, 50, 150)  # Increased thresholds
        
        # Enhanced morphological operations
        kernel = np.ones((3,3), np.uint8)  # Increased kernel size
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Stricter parameters for line detection
        min_line_length = 100  # Increased from 50
        max_line_gap = 10
        threshold = 50        # Increased threshold for Hough transform
        
        lines = cv2.HoughLinesP(edges, 
                               rho=1,
                               theta=np.pi/180,
                               threshold=threshold,
                               minLineLength=min_line_length,
                               maxLineGap=max_line_gap)
        
        if lines is not None:
            # Filter and merge similar lines
            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                # Stricter angle filtering: only keep lines that are close to horizontal or vertical
                is_horizontal = angle < 20 or angle > 160  # Within 20 degrees of horizontal
                is_vertical = 70 < angle < 110             # Within 20 degrees of vertical
                
                if (is_horizontal or is_vertical) and length >= min_line_length:
                    # Check for nearby parallel lines
                    is_duplicate = False
                    for i, existing in enumerate(filtered_lines):
                        ex1, ey1, ex2, ey2 = existing[0]
                        
                        # Calculate distance between midpoints
                        mid1 = ((x1+x2)/2, (y1+y2)/2)
                        mid2 = ((ex1+ex2)/2, (ey1+ey2)/2)
                        dist = np.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)
                        
                        # Calculate angle difference
                        existing_angle = np.abs(np.arctan2(ey2-ey1, ex2-ex1) * 180 / np.pi)
                        angle_diff = abs(angle - existing_angle)
                        
                        # If lines are close and parallel, consider it a duplicate
                        if dist < 30 and (angle_diff < 10 or abs(angle_diff - 180) < 10):
                            is_duplicate = True
                            # Keep the longer line
                            existing_length = np.sqrt((ex2-ex1)**2 + (ey2-ey1)**2)
                            if length > existing_length:
                                filtered_lines.pop(i)  # Use pop instead of remove for list modification
                                is_duplicate = False
                            break
                    
                    if not is_duplicate:
                        filtered_lines.append(line)
            
            # Add filtered lines to results
            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                results['transport'].append({
                    'type': 'road_or_railway',
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': length
                })
        
        return results
    
    def visualize_results(self, image_path, results, output_path):
        """Create visualization of detected symbols"""
        img = cv2.imread(image_path)
        viz_img = img.copy()
        
        # Draw water bodies in blue
        for water in results['water']:
            x, y, w, h = water['bbox']
            cv2.rectangle(viz_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(viz_img, 'Water', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        # Draw terrain features in green
        for terrain in results['terrain']:
            x, y, w, h = terrain['bbox']
            cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(viz_img, 'Terrain', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw transport features in red
        for transport in results['transport']:
            start = tuple(map(int, transport['start']))
            end = tuple(map(int, transport['end']))
            cv2.line(viz_img, start, end, (0, 0, 255), 2)
            
        # Add legend
        legend_y = 30
        cv2.putText(viz_img, 'Blue: Water Bodies', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(viz_img, 'Green: Terrain Features', (10, legend_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(viz_img, 'Red: Transport Routes', (10, legend_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, viz_img)
        return output_path

if __name__ == "__main__":
    # Test the detector
    detector = SymbolDetector()
    results = detector.detect_symbols("datasets/raw_maps/M1.png")
    
    # Print results
    print("\nDetection Results:")
    print("-" * 50)
    
    for category, items in results.items():
        print(f"\n{category.upper()}:")
        print(f"Found {len(items)} features")
        
        if category == 'water':
            total_area = sum(item['area'] for item in items)
            print(f"Total water body area: {total_area} pixels")
            
        elif category == 'terrain':
            total_area = sum(item['area'] for item in items)
            print(f"Total terrain area: {total_area} pixels")
            
        elif category == 'transport':
            total_length = sum(item['length'] for item in items)
            print(f"Total transport route length: {total_length:.2f} pixels")
            
    # Create visualization
    viz_path = "output/visualizations/symbol_detection_results.jpg"
    detector.visualize_results("datasets/raw_maps/M1.png", results, viz_path)
    print(f"\nVisualization saved to: {viz_path}") 