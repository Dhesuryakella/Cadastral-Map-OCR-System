import cv2
import numpy as np
import easyocr
import re
import pandas as pd
import os
import sys
from collections import defaultdict
import json
import torch
import gc
from pathlib import Path
import logging
import traceback
from symbol_detector import SymbolDetector

# Configure logging to show output on console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class CadastralMapExtractor:
    def __init__(self):
        """
        Initialize EasyOCR reader for Cadastral Map text extraction
        Following IIT Tirupati Navavishkar I-Hub Foundation requirements
        """
        try:
            logging.info("🚀 Initializing Cadastral Map OCR System...")
            logging.info("📚 Loading EasyOCR models...")
            
            # Set maximum image size for processing
            self.MAX_IMAGE_SIZE = 2000  # Maximum dimension for processing
        
            # Check if CUDA is available and optimize for GTX 1650
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                try:
                    # Set CUDA device
                    torch.cuda.set_device(0)  # Use first GPU
                    # Optimize CUDA settings for GTX 1650
                    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
                    torch.backends.cudnn.enabled = True
                    # Set optimal batch size for GTX 1650 (4GB VRAM)
                    self.batch_size = 4
                    logging.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
                    logging.info(f"📊 CUDA Version: {torch.version.cuda}")
                    logging.info(f"🧠 Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                except Exception as e:
                    logging.warning(f"GPU initialization failed, falling back to CPU: {str(e)}")
                    self.gpu_available = False
                    self.batch_size = 2
            else:
                self.batch_size = 2
                # Optimize CPU settings
                torch.set_num_threads(4)
                logging.info("Using CPU mode")
        
            # Configure PyTorch settings
            torch.set_grad_enabled(False)  # Disable gradient calculation for inference
        
            try:
                # Initialize EasyOCR with GPU settings
                self.reader = easyocr.Reader(['en'], gpu=self.gpu_available)
                logging.info(f"✅ EasyOCR initialized successfully {'(GPU mode)' if self.gpu_available else '(CPU mode)'}")
            except Exception as e:
                logging.error(f"Failed to initialize EasyOCR: {str(e)}")
                logging.error(traceback.format_exc())
                raise ValueError(f"EasyOCR initialization failed: {str(e)}")
                
            # Initialize symbol detector with trained parameters
            try:
                self.symbol_detector = SymbolDetector()
                logging.info("✅ Symbol detector initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize symbol detector: {str(e)}")
                logging.error(traceback.format_exc())
                raise ValueError(f"Symbol detector initialization failed: {str(e)}")
            
        except Exception as e:
            logging.error(f"Failed to initialize CadastralMapExtractor: {str(e)}")
            logging.error(traceback.format_exc())
            raise ValueError(f"OCR system initialization failed: {str(e)}")
        
    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            gc.collect()

    def _resize_if_needed(self, img):
        """Resize image if it's too large while maintaining aspect ratio"""
        height, width = img.shape[:2]
        max_dim = max(height, width)
        
        if max_dim > self.MAX_IMAGE_SIZE:
            scale = self.MAX_IMAGE_SIZE / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(img, (new_width, new_height),
                              interpolation=cv2.INTER_AREA)
        return img

    def extract_map_text(self, image_path):
        """
        Main extraction pipeline optimized for cadastral maps
        """
        if self.gpu_available:
            # Clear GPU memory before processing
            self._clear_gpu_memory()
            
        logging.info(f"\n🗺️  Processing Cadastral Map: {image_path}")
        logging.info("=" * 60)
        
        try:
            # Load and resize image if needed
            logging.info("Loading image...")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize large images for better performance
            img = self._resize_if_needed(img)
            logging.info(f"📐 Image dimensions: {img.shape[1]}x{img.shape[0]} pixels")
        
            # Convert to HSV once and reuse
            logging.info("Converting to HSV color space...")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
            # Create results dictionary to store intermediate results
            results_dict = {}
        
            # Step 1: Extract red numbers (survey/plot numbers) - High Priority
            logging.info("\n🔴 Step 1: Extracting red survey numbers...")
            try:
                results_dict['red'] = self._extract_red_numbers(img, hsv)
            except Exception as e:
                logging.error(f"Error in red number extraction: {str(e)}")
                logging.error(traceback.format_exc())
                raise
        
            # Step 2: Extract black text (place names) - High Priority  
            logging.info("⚫ Step 2: Extracting black place names...")
            try:
                results_dict['black'] = self._extract_black_text(img, hsv)
            except Exception as e:
                logging.error(f"Error in black text extraction: {str(e)}")
                logging.error(traceback.format_exc())
                raise
        
            # Step 3: Extract colored text (additional place names)
            logging.info("🟢 Step 3: Extracting colored text...")
            try:
                results_dict['colored'] = self._extract_colored_text(img, hsv)
            except Exception as e:
                logging.error(f"Error in colored text extraction: {str(e)}")
                logging.error(traceback.format_exc())
                raise

            # Step 4: Detect map symbols using trained detector
            logging.info("🗺️ Step 4: Detecting map symbols...")
            try:
                symbols = self.symbol_detector.detect_symbols(image_path)
                results_dict['symbols'] = symbols
            except Exception as e:
                logging.error(f"Error in symbol detection: {str(e)}")
                logging.error(traceback.format_exc())
                raise
        
            # Print symbol detection results
            self._print_symbol_summary(symbols)
        
            # Step 5: Enhanced OCR fallback
            logging.info("🔧 Step 5: Enhanced OCR fallback...")
            try:
                results_dict['enhanced'] = self._enhanced_ocr_fallback(img)
            except Exception as e:
                logging.error(f"Error in enhanced OCR: {str(e)}")
                logging.error(traceback.format_exc())
                raise
        
            # Free up memory
            del hsv
        
            # Step 6: Combine and deduplicate results
            logging.info("🔄 Step 6: Combining and deduplicating results...")
            try:
                all_results = self._combine_results(
                    results_dict['red'],
                    results_dict['black'],
                    results_dict['colored'],
                    results_dict['enhanced']
                )
            except Exception as e:
                logging.error(f"Error combining results: {str(e)}")
                logging.error(traceback.format_exc())
                raise
        
            # Free up intermediate results
            del results_dict
        
            # Step 7: Classify final results
            logging.info("📊 Step 7: Classifying characters and numbers...")
            try:
                characters, numbers = self._classify_results_iit_format(all_results)
            except Exception as e:
                logging.error(f"Error classifying results: {str(e)}")
                logging.error(traceback.format_exc())
                raise
        
            if self.gpu_available:
                # Clear GPU memory after processing
                self._clear_gpu_memory()
        
            return characters, numbers, all_results, symbols
    
        except Exception as e:
            logging.error(f"Error in extract_map_text: {str(e)}")
            logging.error(traceback.format_exc())
            raise ValueError(f"Processing failed: {str(e)}")

    def _extract_red_numbers(self, img, hsv):
        """Extract red region/survey numbers"""
        # Original parameters
        red_ranges = [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),    # Lower red
            (np.array([170, 100, 100]), np.array([180, 255, 255]))  # Upper red
        ]
        
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in red_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            red_mask |= mask
        
        kernel = np.ones((3, 3), np.uint8)  # Original larger kernel
        red_mask = cv2.morphologyEx(
            red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Apply mask efficiently using numpy operations
        red_text_img = cv2.bitwise_and(img, img, mask=red_mask)
        
        # Convert to RGB only the masked region
        mask_indices = np.where(red_mask > 0)
        if len(mask_indices[0]) > 0:
            red_text_rgb = cv2.cvtColor(red_text_img, cv2.COLOR_BGR2RGB)
        else:
            red_text_rgb = np.zeros_like(img)
        
        # OCR with optimized parameters for GTX 1650
        results = self.reader.readtext(
            red_text_rgb,
            allowlist='0123456789',
            width_ths=0.7,
            height_ths=0.7,
            paragraph=False,
            batch_size=self.batch_size  # Use optimized batch size
        )
        
        extracted = self._process_ocr_results(results, 'red_number')
        print(f"   → Found {len(extracted)} red numbers")
        
        if self.gpu_available:
            # Clear GPU memory after processing
            self._clear_gpu_memory()
            
        return extracted
    
    def _extract_black_text(self, img, hsv):
        """Extract black place names"""
        # Black text parameters tuned from training (4.95% ± 1.81% coverage)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 50, 100])
        
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        
        # Optimized cleanup based on training data
        # Minimal kernel to preserve text details
        kernel = np.ones((1, 1), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask
        black_text_img = cv2.bitwise_and(img, img, mask=black_mask)
        black_text_rgb = cv2.cvtColor(black_text_img, cv2.COLOR_BGR2RGB)
        
        # OCR for text
        results = self.reader.readtext(black_text_rgb,
                                      width_ths=0.8,
                                      height_ths=0.8,
                                      paragraph=False)
        
        extracted = self._process_ocr_results(results, 'black_text')
        print(f"   → Found {len(extracted)} black text items")
        return extracted
    
    def _extract_colored_text(self, img, hsv):
        """Extract text in other colors (green, blue, etc.)"""
        # Green text mask (5.72% ± 3.06% coverage from training)
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Blue text mask (0.32% ± 0.27% coverage from training)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Combine colored masks
        colored_mask = cv2.bitwise_or(green_mask, blue_mask)
        
        # Optimized cleanup based on training
        kernel = np.ones((1, 1), np.uint8)
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask
        colored_text_img = cv2.bitwise_and(img, img, mask=colored_mask)
        colored_text_rgb = cv2.cvtColor(colored_text_img, cv2.COLOR_BGR2RGB)
        
        # OCR for colored text
        results = self.reader.readtext(colored_text_rgb,
                                      width_ths=0.7,
                                      height_ths=0.7,
                                      paragraph=False)
        
        extracted = self._process_ocr_results(results, 'colored_text')
        print(f"   → Found {len(extracted)} colored text items")
        return extracted
    
    def _enhanced_ocr_fallback(self, img):
        """Fallback OCR on enhanced full image"""
        # Convert to grayscale and enhance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Additional preprocessing for better OCR
        # Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # Convert back to RGB for EasyOCR
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # OCR with moderate confidence
        results = self.reader.readtext(enhanced_rgb,
                                      width_ths=0.6,
                                      height_ths=0.6,
                                      paragraph=False)
        
        extracted = self._process_ocr_results(results, 'enhanced')
        print(f"   → Found {len(extracted)} items from enhanced OCR")
        return extracted
    
    def _process_ocr_results(self, results, source_type):
        """Process raw OCR results into standardized format"""
        processed = []
        
        for (bbox, text, confidence) in results:
            # Clean text
            cleaned_text = text.strip()
            
            # Skip very short or low confidence results
            if len(cleaned_text) < 1 or confidence < 0.3:
                continue
                
            # Calculate center point of bounding box
            center_x = int(np.mean([point[0] for point in bbox]))
            center_y = int(np.mean([point[1] for point in bbox]))
            
            processed.append({
                'text': cleaned_text,
                'bbox': bbox,
                'center': (center_x, center_y),
                'confidence': confidence,
                'source': source_type
            })
        
        return processed
    
    def _combine_results(
            self,
            red_results,
            black_results,
            colored_results,
            enhanced_results):
        """Combine results from different methods and remove duplicates"""
        all_results = red_results + black_results + colored_results + enhanced_results
        
        # Sort by confidence (highest first)
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates based on proximity and text similarity
        final_results = []
        proximity_threshold = 30  # pixels
        
        for result in all_results:
            is_duplicate = False
            
            for existing in final_results:
                # Check if texts are similar and locations are close
                if (self._text_similarity(result['text'], existing['text']) > 0.8 and
                    self._distance(result['center'], existing['center']) < proximity_threshold):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_results.append(result)
        
        print(
            f"   → Combined {
                len(all_results)} total → {
                len(final_results)} unique items")
        return final_results
    
    def _classify_results_iit_format(self, results):
        """
        Classify results into Characters and Numbers
        Characters: Place names like Konal, Gonal, Benakanahalli, etc.
        Numbers: Survey/plot numbers like 76, 20, 22, 24, etc.
        """
        characters = []  # Place names
        numbers = []     # Survey/plot numbers
        
        for result in results:
            text = result['text']
            
            # Numbers: Survey/plot numbers (1-4 digits, reasonable range)
            if re.match(r'^\d{1,4}$', text):
                try:
                    num = int(text)
                    if 1 <= num <= 9999:  # Reasonable range for survey numbers
                        numbers.append({
                            'number': text,
                            'location': result['center'],
                            'confidence': result['confidence'],
                            'source': result['source'],
                            'bbox': result.get('bbox', [])
                        })
                except ValueError:
                    continue
            
            # Characters: Place names (contain letters, reasonable length)
            elif re.search(r'[A-Za-z]', text) and 2 <= len(text) <= 50:
                # Clean up the place name
                cleaned_name = self._clean_place_name(text)
                
                # Additional filtering for valid place names
                if (cleaned_name and 
                    # Not just numbers
                    not re.match(r'^\d+$', cleaned_name) and
                    len(cleaned_name) >= 2):
                    
                    characters.append({
                        'name': cleaned_name,
                        'location': result['center'],
                        'confidence': result['confidence'],
                        'source': result['source'],
                        'bbox': result.get('bbox', [])
                    })
        
        # Sort and remove duplicates
        unique_characters = self._remove_duplicate_names(characters)
        unique_numbers = self._remove_duplicate_numbers(numbers)
        
        logging.info(f"   → Classified {len(unique_characters)} place names")
        logging.info(f"   → Classified {len(unique_numbers)} survey numbers")
        
        return unique_characters, unique_numbers
    
    def _clean_place_name(self, text):
        """Clean up common OCR errors in place names"""
        # Remove special characters but keep spaces, hyphens, and dots
        cleaned = re.sub(r'[^\w\s\-\.]', '', text)

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Remove dots without spaces (likely OCR artifacts)
        cleaned = re.sub(r'\.(?!\s|$)', '', cleaned)

        # Capitalize properly (first letter of each word)
        cleaned = cleaned.strip().title()
        
        # Filter out common OCR noise patterns
        noise_patterns = [
            r'^[0-9\s\-\.]+$',  # Only numbers, spaces, hyphens, dots
            r'^[A-Z]{1}$',      # Single letters
            r'^\s*$',           # Empty or whitespace only
            r'^[NSEW]$',        # Single cardinal directions
            r'^\d+[A-Za-z]$',   # Numbers followed by single letter
            r'^[A-Za-z]\d+$'    # Single letter followed by numbers
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, cleaned):
                return None
                
        return cleaned
    
    def _remove_duplicate_names(self, characters):
        """Remove duplicate place names using fuzzy matching"""
        if not characters:
            return []

        # Sort by confidence score first
        characters.sort(key=lambda x: x['confidence'], reverse=True)

        # Use a dictionary to track unique names
        unique_dict = {}
        unique_list = []
        
        for char_data in characters:
            name = char_data['name'].lower()

            # Check if this name is similar to any existing name
            is_duplicate = False
            for existing_name in unique_dict.keys():
                if self._text_similarity(name, existing_name) > 0.8:
                    is_duplicate = True
                    # If new entry has higher confidence, replace the old one
                    if char_data['confidence'] > unique_dict[existing_name]['confidence']:
                        unique_dict[existing_name] = char_data
                    break

            if not is_duplicate:
                unique_dict[name] = char_data

        # Convert back to list and sort by name
        unique_list = list(unique_dict.values())
        unique_list.sort(key=lambda x: x['name'])

        return unique_list
    
    def _remove_duplicate_numbers(self, numbers):
        """Remove duplicate numbers using spatial and value proximity"""
        if not numbers:
            return []

        # Sort by confidence score first
        numbers.sort(key=lambda x: x['confidence'], reverse=True)

        unique_dict = {}
        proximity_threshold = 30  # pixels
        
        for num_data in numbers:
            num = num_data['number']
            center = num_data['location']

            # Check for duplicates considering both number value and location
            is_duplicate = False
            for existing_num, existing_data in unique_dict.items():
                # Check if numbers are same or adjacent
                num_diff = abs(int(num) - int(existing_num))
                if num_diff <= 1:
                    # Check spatial proximity
                    if self._distance(
                            center, existing_data['location']) < proximity_threshold:
                        is_duplicate = True
                        # Keep the one with higher confidence
                        if num_data['confidence'] > existing_data['confidence']:
                            unique_dict[num] = num_data
                        break

            if not is_duplicate:
                unique_dict[num] = num_data

        # Convert back to list and sort by number value
        unique_list = list(unique_dict.values())
        unique_list.sort(key=lambda x: int(x['number']))

        return unique_list
    
    def _text_similarity(self, text1, text2):
        """Calculate similarity between two texts using optimized operations"""
        # Early exit for identical strings
        if text1 == text2:
            return 1.0
            
        # Convert to lowercase once
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Early exit for empty strings
        if not text1_lower or not text2_lower:
            return 0.0
            
        # Use sets for faster operations
        set1 = frozenset(text1_lower)  # Immutable set for better performance
        set2 = frozenset(text2_lower)
        
        # Calculate intersection and union using set operations
        intersection = len(set1 & set2)  # Faster than intersection()
        union = len(set1 | set2)  # Faster than union()
        
        return intersection / union if union > 0 else 0
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points using numpy"""
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
    
    def _print_symbol_summary(self, symbols):
        """Print summary of detected symbols"""
        logging.info("\nSymbol Detection Summary:")
        logging.info("-" * 50)
        
        for category, items in symbols.items():
            logging.info(f"\n{category.upper()}:")
            logging.info(f"Found {len(items)} features")

            if category == 'water':
                total_area = sum(item['area'] for item in items)
                logging.info(f"Total water body area: {total_area} pixels")

            elif category == 'terrain':
                total_area = sum(item['area'] for item in items)
                logging.info(f"Total terrain area: {total_area} pixels")

            elif category == 'transport':
                total_length = sum(item['length'] for item in items)
                logging.info(
                    f"Total transport route length: {
                        total_length:.2f} pixels")

    def save_detailed_results(self, characters, numbers, symbols, base_filename):
        """Save detailed detection results in JSON format"""
        try:
            # Create results directory if it doesn't exist
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Prepare detailed results dictionary
            symbol_data = []
            for category, items in symbols.items():
                for item in items:
                    if isinstance(item, dict):
                        # Handle dictionary format (from _detect_symbols)
                        location = ""
                        if 'bbox' in item:
                            location = f"at ({item['bbox'][0]}, {item['bbox'][1]})"
                        elif 'start' in item and 'end' in item:
                            location = f"from ({item['start'][0]}, {item['start'][1]}) to ({item['end'][0]}, {item['end'][1]})"
                        
                        symbol_data.append({
                            'Category': category,
                            'Type': item.get('type', 'Unknown'),
                            'Count/Area': location
                        })
                    elif isinstance(item, tuple) and len(item) >= 2:
                        # Handle tuple format (legacy)
                        symbol_data.append({
                            'Category': category,
                            'Type': item[0],
                            'Count/Area': item[1] if not isinstance(item[1], tuple) else f"at {item[1]}"
                        })
                    else:
                        # Handle other formats
                        symbol_data.append({
                            'Category': category,
                            'Type': str(item),
                            'Count/Area': 'N/A'
                        })
            
            detailed_results = {
                "text_detection": {
                    "place_names": [
                        {
                            "name": char["name"],
                            "location": char["location"],
                            "confidence": char["confidence"],
                            "source": char["source"],
                            "bbox": char.get("bbox", [])
                        } for char in characters
                    ],
                    "survey_numbers": [
                        {
                            "number": num["number"],
                            "location": num["location"],
                            "confidence": num["confidence"],
                            "source": num["source"],
                            "bbox": num.get("bbox", [])
                        } for num in numbers
                    ]
                },
                "symbol_detection": symbols,
                "statistics": {
                    "total_place_names": len(characters),
                    "total_survey_numbers": len(numbers),
                    "total_symbols": sum(len(items) for items in symbols.values()),
                    "symbol_counts": {
                        category: len(items) for category, items in symbols.items()
                    }
                }
            }
            
            # Save JSON file
            json_path = results_dir / f"{base_filename}_detailed.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, default=self.convert_numpy_types)
            print(f"✅ Detailed results saved to: {json_path}")
            
            return str(json_path)
            
        except Exception as e:
            print(f"❌ Error saving detailed results: {str(e)}")
            logging.error(f"Error saving detailed results: {str(e)}")
            logging.error(traceback.format_exc())
            return None
            
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
                        
    def save_results_iit_format(self, characters, numbers, symbols, output_path="cadastral_results"):
        """
        Save results with additional symbol information
        """
        print(f"\n💾 Saving results...")
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Update paths to save in results directory
        base_filename = Path(output_path).stem
        csv_path = results_dir / f"{base_filename}.csv"
        excel_path = results_dir / f"{base_filename}_detailed.xlsx"
        
        # Create main DataFrame for characters and numbers
        character_names = [char['name'] for char in characters]
        number_values = [num['number'] for num in numbers]
        
        max_len = max(len(character_names), len(number_values), 1)
        padded_characters = character_names + [''] * (max_len - len(character_names))
        padded_numbers = number_values + [''] * (max_len - len(number_values))
        
        main_df = pd.DataFrame({
            'Characters': padded_characters,
            'Numbers': padded_numbers
        })
        
        # Create symbols DataFrame
        symbol_data = []
        for category, items in symbols.items():
            for item in items:
                if isinstance(item, dict):
                    # Handle dictionary format (from _detect_symbols)
                    location = ""
                    if 'bbox' in item:
                        location = f"at ({item['bbox'][0]}, {item['bbox'][1]})"
                    elif 'start' in item and 'end' in item:
                        location = f"from ({item['start'][0]}, {item['start'][1]}) to ({item['end'][0]}, {item['end'][1]})"
                    
                    symbol_data.append({
                        'Category': category,
                        'Type': item.get('type', 'Unknown'),
                        'Count/Area': location
                    })
                elif isinstance(item, tuple) and len(item) >= 2:
                    # Handle tuple format (legacy)
                    symbol_data.append({
                        'Category': category,
                        'Type': item[0],
                        'Count/Area': item[1] if not isinstance(item[1], tuple) else f"at {item[1]}"
                    })
                else:
                    # Handle other formats
                    symbol_data.append({
                        'Category': category,
                        'Type': str(item),
                        'Count/Area': 'N/A'
                    })
        
        symbols_df = pd.DataFrame(symbol_data)
        
        try:
            # Save main results
            main_df.to_csv(csv_path, index=False)
            print(f"✅ Results saved to: {csv_path}")
            
            # Save detailed results
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    main_df.to_excel(writer, sheet_name='Text_Numbers', index=False)
                    symbols_df.to_excel(writer, sheet_name='Symbols', index=False)
                print(f"✅ Detailed Excel results saved to: {excel_path}")
            except Exception as e:
                print(f"⚠️  Could not save detailed Excel file: {str(e)}")
                excel_path = None
                
            # Save detailed JSON results
            json_path = self.save_detailed_results(characters, numbers, symbols, base_filename)
                
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            raise
            
        # Print summary
        self._print_final_summary(characters, numbers, symbols)
        
        return str(csv_path), str(excel_path) if excel_path else None
        
    def _print_final_summary(self, characters, numbers, symbols):
        """Print final summary of all detected elements"""
        print("\n" + "=" * 80)
        print("🎯 CADASTRAL MAP EXTRACTION RESULTS SUMMARY")
        print("=" * 80)
        
        # Text and numbers summary
        print(f"📍 Total Characters (Place Names): {len(characters)}")
        print(f"🔢 Total Numbers (Survey Numbers): {len(numbers)}")
        
        if characters:
            print(
                f"\n📍 Characters: {', '.join(char['name'] for char in characters[:10])}")
            if len(characters) > 10:
                print(f"   ... and {len(characters) - 10} more")
        
        if numbers:
            print(
                f"\n🔢 Numbers: {', '.join(num['number'] for num in numbers[:20])}")
            if len(numbers) > 20:
                print(f"   ... and {len(numbers) - 20} more")
        
        # Symbol summary
        print("\n🗺️ Map Symbols Summary:")
        total_symbols = sum(len(items) for items in symbols.values())
        print(f"Total Symbols Detected: {total_symbols}")
        
        for category, items in symbols.items():
            if items:
                print(f"- {category.title()}: {len(items)} features")
        
        print("=" * 80)

    def visualize_results(self, image_path, characters, numbers, symbols, output_path="extraction_results.jpg"):
        """Create visualization of extraction results including detected symbols"""
        print(f"\n🎨 Creating visualization...")
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Update output path to save in results directory
        output_path = results_dir / output_path
        
        img = cv2.imread(image_path)
        
        # Create a copy for visualization
        viz_img = img.copy()
        
        # Corrected color scheme according to Survey of India conventions
        colors = {
            "transport": (0, 0, 255),       # Red for transport features
            "water": (255, 0, 0),           # Blue for water features
            "boundaries": (0, 255, 0),      # Green for boundaries
            "settlements": (0, 0, 0),       # Black for settlements
            "terrain": (139, 69, 19),       # Brown for terrain
            "survey_numbers": (0, 0, 255),  # Red for survey numbers
            "place_names": (0, 0, 0)        # Black for place names
        }
        
        # Draw transport features first with thicker lines
        if 'transport' in symbols:
            for item in symbols['transport']:
                if isinstance(item, dict) and 'start' in item and 'end' in item:
                    start = tuple(map(int, item['start']))
                    end = tuple(map(int, item['end']))
                    # Draw thicker lines for transport features
                    cv2.line(viz_img, start, end, colors['transport'], 1)
        
        # Draw water features
        if 'water' in symbols:
            for item in symbols['water']:
                if isinstance(item, dict) and 'bbox' in item:
                    x, y, w, h = item['bbox']
                    cv2.rectangle(viz_img, (x, y), (x+w, y+h), colors['water'], 2)
                    cv2.putText(viz_img, 'Water', (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['water'], 2)
        
        # Draw terrain features
        if 'terrain' in symbols:
            for item in symbols['terrain']:
                if isinstance(item, dict) and 'bbox' in item:
                    x, y, w, h = item['bbox']
                    cv2.rectangle(viz_img, (x, y), (x+w, y+h), colors['terrain'], 2)
                    cv2.putText(viz_img, 'Terrain', (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['terrain'], 2)
        
        # Draw survey numbers in red circles
        for num_data in numbers:
            center = num_data['location']
            cv2.circle(viz_img, center, 15, colors["survey_numbers"], 2)
            cv2.putText(viz_img, str(num_data['number']), 
                       (center[0]-10, center[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["survey_numbers"], 2)
        
        # Draw place names in black
        for char_data in characters:
            center = char_data['location']
            cv2.circle(viz_img, center, 10, colors["place_names"], 2)
            display_name = char_data['name'][:12] + "..." if len(char_data['name']) > 12 else char_data['name']
            cv2.putText(viz_img, display_name, 
                       (center[0]-30, center[1]+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["place_names"], 1)
        
        # Add legend
        legend_y = 30
        legend_items = [
            ("Red: Transport Features", colors["transport"]),
            ("Blue: Water Features", colors["water"]),
            ("Green: Boundaries", colors["boundaries"]),
            ("Black: Settlements", colors["settlements"]),
            ("Brown: Terrain", colors["terrain"]),
            ("Red: Survey Numbers", colors["survey_numbers"]),
            ("Black: Place Names", colors["place_names"])
        ]
        
        # Draw legend with background
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + (i * 25)
            # Draw white background for better visibility
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(viz_img, (8, y_pos-text_height-2), (12+text_width, y_pos+2), (255,255,255), -1)
            cv2.putText(viz_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imwrite(str(output_path), viz_img)
        print(f"✅ Visualization saved to: {output_path}")
        
        return str(output_path)

    def detect_water_features(self, image):
        """Specialized water feature detection"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define blue color range for water
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Create mask for blue regions
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        water_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Classify based on shape
                if circularity > 0.6:
                    water_features.append(('lake', contour))
                else:
                    water_features.append(('river', contour))
        
        return water_features


class TopographicalSymbolDetector:
    def __init__(self):
        """Initialize the symbol detector with conventional symbols template"""
        try:
            logging.info("Initializing TopographicalSymbolDetector...")

            # Load symbol configuration data
            logging.info("Loading symbols data...")
            self.symbols_data = self._load_symbols_data()
            logging.info("Symbols data loaded successfully")

            # Load conventional symbols template
            logging.info("Loading conventional symbols template...")
            self.conventional_symbols = self._load_conventional_symbols()
            logging.info("Conventional symbols template loaded successfully")

            # Extract reference symbols
            logging.info("Extracting reference symbols...")
            self.reference_symbols = self._extract_reference_symbols()
            logging.info("Reference symbols extracted successfully")

        except Exception as e:
            error_msg = f"Failed to initialize TopographicalSymbolDetector: {
                str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise ValueError(error_msg)

    def _load_symbols_data(self):
        """Load symbol configuration data"""
        try:
            # Get the project root directory (parent of src)
            project_root = Path(__file__).resolve().parent.parent
            symbols_data_path = project_root / "datasets" / "symbols" / "symbols_data.json"

            logging.info(f"Loading symbols data from: {symbols_data_path}")

            if not symbols_data_path.exists():
                logging.warning(
                    f"Symbols data file not found at: {symbols_data_path}")
                logging.warning("Using default color ranges")
                return {
                    # Black/dark
                    'transport': {'color_range': {'lower': [0, 0, 0], 'upper': [180, 255, 50]}},
                    # Blue
                    'water': {'color_range': {'lower': [90, 50, 50], 'upper': [130, 255, 255]}},
                    # Black
                    'settlements': {'color_range': {'lower': [0, 0, 0], 'upper': [180, 50, 100]}},
                    # Green/brown
                    'terrain': {'color_range': {'lower': [20, 50, 50], 'upper': [60, 255, 255]}},
                    # Black
                    'boundaries': {'color_range': {'lower': [0, 0, 0], 'upper': [180, 50, 100]}}
                }

            with open(symbols_data_path, 'r') as f:
                data = json.load(f)
                logging.info("Symbols data loaded successfully")
                return data

        except Exception as e:
            error_msg = f"Error loading symbols data: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise ValueError(error_msg)

    def _load_conventional_symbols(self):
        """Load the conventional symbols template image"""
        try:
            # Get the project root directory (parent of src)
            project_root = Path(__file__).resolve().parent.parent
            template_path = project_root / "datasets" / \
                "symbols" / "CONVENTIOANL_SYMBOLS.jpeg"

            logging.info(f"Loading template from: {template_path}")

            if not template_path.exists():
                error_msg = f"Could not find conventional symbols template at: {template_path}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            img = cv2.imread(str(template_path))
            if img is None:
                error_msg = f"Could not load conventional symbols template: {template_path}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            logging.info(f"Template loaded successfully: {img.shape}")
            return img

        except Exception as e:
            error_msg = f"Error loading conventional symbols template: {
                str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise ValueError(error_msg)

    def _extract_reference_symbols(self):
        """Extract individual symbols from the conventional symbols template"""
        try:
            if self.conventional_symbols is None:
                raise ValueError("Conventional symbols template not loaded")

            print(f"\nTemplate image shape: {self.conventional_symbols.shape}")
            if len(self.conventional_symbols.shape) != 3:
                raise ValueError(
                    f"Invalid template image format: expected 3 channels, got {
                        len(
                            self.conventional_symbols.shape)}")

            reference_symbols = {
                'terrain': [],
                'transport': [],
                'settlements': [],
                'boundaries': [],
                'water': []
            }

            # Define regions in the conventional symbols image for each category
            # Coordinates based on CONVENTIONAL_SYMBOLS.jpeg layout
            regions = {
                'transport': [(50, 50, 300, 200)],      # Roads, railways, etc.
                'water': [(50, 250, 300, 400)],         # Rivers, lakes, etc.
                'settlements': [(350, 50, 600, 200)],   # Villages, towns, etc.
                # Hills, vegetation, etc.
                'terrain': [(350, 250, 600, 400)],
                # District, state boundaries, etc.
                'boundaries': [(50, 450, 300, 600)]
            }

            print("\nExtracting reference symbols from template:")
            print("-" * 50)

            # Extract symbols from each region
            for category, region_list in regions.items():
                try:
                    print(f"\nProcessing {category} symbols:")
                    for region_idx, (x1, y1, x2, y2) in enumerate(region_list):
                        try:
                            print(
                                f"  Processing region {
                                    region_idx + 1} at ({x1},{y1},{x2},{y2})")

                            # Validate coordinates
                            if x1 < 0 or y1 < 0:
                                print(
                                    f"  Warning: Negative coordinates ({x1},{y1})")
                                continue

                            if x2 > self.conventional_symbols.shape[1] or y2 > self.conventional_symbols.shape[0]:
                                print(
                                    f"  Warning: Coordinates ({x2},{y2}) exceed image dimensions {self.conventional_symbols.shape[:2]}")
                                continue

                            if x2 <= x1 or y2 <= y1:
                                print(
                                    f"  Warning: Invalid region dimensions: width={
                                        x2 -
                                        x1}, height={
                                        y2 -
                                        y1}")
                                continue

                            region = self.conventional_symbols[y1:y2, x1:x2]
                            print(f"  Region shape: {region.shape}")

                            if region.size == 0:
                                print(f"  Warning: Empty region")
                                continue

                            # Convert to grayscale for better matching
                            try:
                                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                            except Exception as e:
                                print(
                                    f"  Error converting to grayscale: {
                                        str(e)}")
                                continue

                            # Use adaptive thresholding for better symbol
                            # extraction
                            try:
                                thresh = cv2.adaptiveThreshold(
                                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                            except Exception as e:
                                print(f"  Error during thresholding: {str(e)}")
                                continue

                            # Find contours
                            try:
                                contours, _ = cv2.findContours(
                                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                print(
                                    f"  Found {
                                        len(contours)} potential symbols")
                            except Exception as e:
                                print(f"  Error finding contours: {str(e)}")
                                continue

                            valid_symbols = 0
                            for contour_idx, contour in enumerate(contours):
                                try:
                                    x, y, w, h = cv2.boundingRect(contour)
                                    area = w * h

                                    if area < 50:  # Minimum size threshold
                                        continue

                                    # Add padding around symbol
                                    pad = 5
                                    x_pad = max(0, x - pad)
                                    y_pad = max(0, y - pad)
                                    w_pad = min(
                                        region.shape[1] - x_pad, w + 2 * pad)
                                    h_pad = min(
                                        region.shape[0] - y_pad, h + 2 * pad)

                                    if w_pad <= 0 or h_pad <= 0:
                                        print(
                                            f"  Warning: Invalid padded dimensions for contour {contour_idx}")
                                        continue

                                    symbol = region[y_pad:y_pad +
                                                    h_pad, x_pad:x_pad + w_pad]
                                    if symbol.size == 0:
                                        print(
                                            f"  Warning: Empty symbol for contour {contour_idx}")
                                        continue

                                    reference_symbols[category].append({
                                        'image': symbol.copy(),  # Make a copy to ensure independence
                                        'bbox': [x_pad, y_pad, w_pad, h_pad],
                                        'area': w_pad * h_pad
                                    })
                                    valid_symbols += 1

                                except Exception as e:
                                    print(
                                        f"  Error processing contour {contour_idx}: {
                                            str(e)}")
                                    continue

                            print(f"  Added {valid_symbols} valid symbols")

                        except Exception as e:
                            print(
                                f"  Error processing region {
                                    region_idx +
                                    1}: {
                                    str(e)}")
                            continue

                except Exception as e:
                    print(f"Error processing category {category}: {str(e)}")
                    continue

            total_symbols = sum(len(symbols)
                                for symbols in reference_symbols.values())
            print(f"\nExtracted {total_symbols} total reference symbols:")
            for category, symbols in reference_symbols.items():
                print(f"  • {category}: {len(symbols)} symbols")

            if total_symbols == 0:
                raise ValueError(
                    "No reference symbols could be extracted from the template")

            return reference_symbols

        except Exception as e:
            error_msg = f"Error extracting reference symbols: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise ValueError(error_msg)

    def _detect_symbols(self, img):
        """Detect symbols using block-based template matching"""
        try:
            results = {}
            
            # Convert input image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"Processing image of size: {img.shape[1]}x{img.shape[0]}")
            
            # Define block size and overlap
            block_size = 300  # Increased block size for better context
            overlap = 150     # Increased overlap for better detection at boundaries
            height, width = img.shape[:2]
            
            for category in self.reference_symbols.keys():
                try:
                    print(f"\nProcessing {category} symbols:")
                    category_results = []
                    
                    # Get color range for this category
                    color_range = self.symbols_data.get(category, {}).get('color_range', {})
                    if not color_range:
                        print(f"No color range defined for {category}, skipping...")
                        continue
                        
                    lower = np.array(color_range['lower'])
                    upper = np.array(color_range['upper'])
                    
                    # Create color mask
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    
                    # Apply morphological operations to clean up mask
                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    # For transport features, apply additional filtering
                    if category == 'transport':
                        # Increase minimum line length and threshold for more selective detection
                        min_line_length = 100  # Increased from 50 to 100
                        max_line_gap = 5       # Reduced from 10 to 5
                        min_votes = 100        # Increased from 50 to 100
                        
                        # Apply additional preprocessing to reduce noise
                        # Use bilateral filter to reduce noise while preserving edges
                        mask = cv2.bilateralFilter(mask, 9, 75, 75)
                        
                        # Use probabilistic Hough transform for line detection with stricter parameters
                        lines = cv2.HoughLinesP(mask, 
                                              rho=1,
                                              theta=np.pi/180,
                                              threshold=min_votes,
                                              minLineLength=min_line_length,
                                              maxLineGap=max_line_gap)
                        
                        if lines is not None:
                            # Filter lines based on angle and proximity
                            filtered_lines = []
                            for line in lines:
                                x1, y1, x2, y2 = line[0]
                                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                                
                                # Only keep lines that are roughly horizontal (0±30°) or vertical (90±30°)
                                if (angle < 30 or (angle > 60 and angle < 120) or angle > 150):
                                    # Check if line is too close to existing lines
                                    is_duplicate = False
                                    for existing in filtered_lines:
                                        ex1, ey1, ex2, ey2 = existing[0]
                                        # Calculate distance between midpoints
                                        mid1 = ((x1+x2)/2, (y1+y2)/2)
                                        mid2 = ((ex1+ex2)/2, (ey1+ey2)/2)
                                        dist = np.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)
                                        if dist < 20:  # If lines are closer than 20 pixels
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        filtered_lines.append(line)
                            
                            # Add filtered lines to results
                            for line in filtered_lines:
                                x1, y1, x2, y2 = line[0]
                                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                                category_results.append({
                                    'type': 'road_or_railway',
                                    'start': (x1, y1),
                                    'end': (x2, y2),
                                    'length': length
                                })
                    else:
                        # Process each block with overlap for other categories
                        total_blocks = ((height - overlap) // (block_size - overlap)) * ((width - overlap) // (block_size - overlap))
                        processed_blocks = 0
                        
                        for y in range(0, height - overlap, block_size - overlap):
                            for x in range(0, width - overlap, block_size - overlap):
                                try:
                                    processed_blocks += 1
                                    if processed_blocks % 10 == 0:
                                        print(f"Processing block {processed_blocks}/{total_blocks} for {category}")
                                        
                                    # Extract block
                                    block_end_y = min(y + block_size, height)
                                    block_end_x = min(x + block_size, width)
                                    block = gray[y:block_end_y, x:block_end_x]
                                    block_mask = mask[y:block_end_y, x:block_end_x]
                                    
                                    # Skip if block has no relevant colors
                                    if cv2.countNonZero(block_mask) < block_size * 0.1:  # Increased threshold
                                        continue
                                        
                                    # Enhance block contrast
                                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                                    block = clahe.apply(block)
                                    
                                    # Match against reference symbols
                                    ref_count = len(self.reference_symbols[category])
                                    for ref_idx, ref_symbol in enumerate(self.reference_symbols[category]):
                                        try:
                                            template = cv2.cvtColor(ref_symbol['image'], cv2.COLOR_BGR2GRAY)
                                            template = clahe.apply(template)
                                            
                                            # Try different scales
                                            for scale in [0.75, 1.0, 1.25]:  # Reduced scale range
                                                try:
                                                    scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                                                    if scaled_template.shape[0] > block.shape[0] or scaled_template.shape[1] > block.shape[1]:
                                                        continue
                                                        
                                                    result = cv2.matchTemplate(block, scaled_template, cv2.TM_CCOEFF_NORMED)
                                                    threshold = 0.85  # Increased threshold for higher confidence
                                                    
                                                    locations = np.where(result >= threshold)
                                                    for pt in zip(*locations[::-1]):
                                                        # Convert coordinates to original image space
                                                        orig_x = x + pt[0]
                                                        orig_y = y + pt[1]
                                                        w = int(ref_symbol['bbox'][2] * scale)
                                                        h = int(ref_symbol['bbox'][3] * scale)
                                                        
                                                        # Check if this detection overlaps with existing ones
                                                        is_duplicate = False
                                                        for existing in category_results:
                                                            if self._check_overlap(
                                                                [orig_x, orig_y, w, h],
                                                                existing['bbox'],
                                                                threshold=0.5  # 50% overlap threshold
                                                            ):
                                                                is_duplicate = True
                                                                break
                                                        
                                                        if not is_duplicate:
                                                            category_results.append({
                                                                'type': category,
                                                                'bbox': [orig_x, orig_y, w, h],
                                                                'area': w * h,
                                                                'confidence': float(result[pt[1], pt[0]])
                                                            })
                                                except Exception as e:
                                                    print(f"Error processing scale {scale} for template {ref_idx}: {str(e)}")
                                                    continue
                                        except Exception as e:
                                            print(f"Error processing reference symbol {ref_idx}/{ref_count}: {str(e)}")
                                            continue
                                except Exception as e:
                                    print(f"Error processing block at ({x},{y}): {str(e)}")
                                    continue
                    
                    # Remove overlapping detections with stricter threshold
                    print(f"Found {len(category_results)} potential matches for {category}")
                    category_results = self._remove_overlapping(category_results, iou_threshold=0.3)
                    print(f"After removing overlaps: {len(category_results)} matches for {category}")
                    results[category] = category_results
                    
                except Exception as e:
                    print(f"Error processing category {category}: {str(e)}")
                    results[category] = []
            
            return results
            
        except Exception as e:
            print(f"Error in symbol detection: {str(e)}")
            traceback.print_exc()
            return {}

    def _check_overlap(self, bbox1, bbox2, threshold=0.5):
        """Check if two bounding boxes overlap more than the threshold"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return False

        # Calculate intersection area
        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        # Calculate IoU
        iou = intersection / union if union > 0 else 0

        return iou > threshold


def main():
    """
    Main function for Cadastral Map OCR Processing
    """
    print("🗺️  CADASTRAL MAP OCR SYSTEM")
    print("=" * 80)
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        print("Example: python script.py M1.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Validate input file
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    # Get file info
    file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
    print(f"📁 Input file: {image_path} ({file_size:.1f} MB)")
    
    try:
        # Initialize extractor with enhanced settings
        print("\n🚀 Initializing detection system...")
        extractor = CadastralMapExtractor()
        
        # Load symbol templates
        print("📚 Loading symbol templates...")
        symbols_data_path = r"D:\OCR\datasets\symbols\symbols_data.json"
        if os.path.exists(symbols_data_path):
            with open(symbols_data_path, 'r') as f:
                symbol_templates = json.load(f)
            print(f"✅ Loaded symbol templates from {symbols_data_path}")
        else:
            print("⚠️ Symbol templates not found, using default templates")
            symbol_templates = None
        
        # Process the cadastral map
        print("\n🔍 Processing map...")
        characters, numbers, all_results, symbols = extractor.extract_map_text(
            image_path)
        
        # Save results in IIT Tirupati format
        print("\n💾 Saving results...")
        csv_path, excel_path = extractor.save_results_iit_format(
            characters, numbers, symbols, "cadastral_output"
        )
        
        # Create enhanced visualization
        print("\n🎨 Creating visualization...")
        viz_path = extractor.visualize_results(
            image_path, characters, numbers, symbols, "extraction_results.jpg"
        )
        
        # Print detailed analysis
        print("\n📊 DETECTION ANALYSIS")
        print("=" * 80)
        
        # Text detection stats
        total_text = len(characters) + len(numbers)
        avg_confidence = 0
        if total_text > 0:
            confidences = ([r["confidence"] for r in all_results])
            avg_confidence = sum(confidences) / len(confidences)
        
        print(f"\nText Detection Statistics:")
        print(f"- Total text elements detected: {total_text}")
        print(f"- Place names detected: {len(characters)}")
        print(f"- Survey numbers detected: {len(numbers)}")
        print(f"- Average confidence score: {avg_confidence:.2%}")
        
        # Symbol detection stats
        total_symbols = sum(len(items) for items in symbols.values())
        print(f"\nSymbol Detection Statistics:")
        for category, items in symbols.items():
            if items:
                print(f"\n{category.upper()}:")
                if isinstance(items[0], tuple):
                    print(f"- {len(items)} instances")
                    for item in items[:5]:
                        print(f"  • {item[0]}")
                    if len(items) > 5:
                        print(f"  • ... and {len(items) - 5} more")
                else:
                    for item in items:
                        print(f"  • {item}")
        
        print(f"\nTotal symbols detected: {total_symbols}")
        
        # Output files
        print("\n📁 OUTPUT FILES:")
        print(f"- CSV Results: {csv_path}")
        if excel_path:
            print(f"- Excel Results: {excel_path}")
        print(f"- Visualization: {viz_path}")
        
        print("\n✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()