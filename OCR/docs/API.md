# Indian Topographical Map OCR System API Documentation

## Main Classes

### CadastralMapExtractor

Main class for processing cadastral maps and extracting features.

```python
extractor = CadastralMapExtractor()
```

#### Methods

##### extract_map_text(image_path)

Process a map image and extract text and symbols.

Parameters:

- `image_path`: Path to the input map image

Returns:

- `characters`: List of detected place names
- `numbers`: List of detected survey numbers
- `all_results`: Combined detection results
- `symbols`: Dictionary of detected map symbols

##### visualize_results(image_path, characters, numbers, symbols, output_path)

Create visualization of detection results.

Parameters:

- `image_path`: Original map image path
- `characters`: Detected place names
- `numbers`: Detected survey numbers
- `symbols`: Detected symbols
- `output_path`: Path to save visualization

### TopographicalSymbolDetector

Detector for Survey of India conventional map symbols.

```python
detector = TopographicalSymbolDetector()
```

#### Symbol Categories

1. Transport Features (Red)

   - National highways
   - State highways
   - Railways

2. Water Features (Blue)

   - Rivers
   - Lakes
   - Tanks

3. Settlements (Black)

   - Cities
   - Towns
   - Villages
   - Religious places

4. Boundaries (Green)

   - International
   - State
   - District

5. Terrain (Brown)
   - Contours
   - Forest
   - Elevation

## Usage Example

```python
from src.detect import CadastralMapExtractor

# Initialize extractor
extractor = CadastralMapExtractor()

# Process map
characters, numbers, results, symbols = extractor.extract_map_text("map.png")

# Generate visualization
extractor.visualize_results("map.png", characters, numbers, symbols)
```

## Output Formats

1. CSV Output (`output/cadastral_output.csv`)

   - Simple list of detected text and numbers

2. Excel Output (`output/cadastral_output_detailed.xlsx`)

   - Sheet 1: Text and Numbers
   - Sheet 2: Detected Symbols

3. Visualization (`output/extraction_results.jpg`)
   - Color-coded visualization of all detections
