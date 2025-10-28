# User Guide

## Introduction

This guide provides detailed instructions for using the Indian Topographical Map OCR System. Whether you're processing a single map or working with multiple files, this guide will help you get the most accurate results.

## Table of Contents

1. [Map Preparation](#map-preparation)
2. [Running the System](#running-the-system)
3. [Understanding Results](#understanding-results)
4. [Troubleshooting](#troubleshooting)

## Map Preparation

### Image Requirements

1. **Supported Formats**

   - PNG (recommended)
   - JPEG
   - TIFF

2. **Resolution**

   - Minimum: 1000x1000 pixels
   - Recommended: 2000x2000 pixels
   - Maximum: 8000x8000 pixels

3. **Quality Guidelines**
   - Clear, well-lit images
   - Minimal glare or shadows
   - No physical damage or marks
   - Proper color balance

### Pre-processing Tips

1. **Image Enhancement**

   - Adjust contrast if needed
   - Remove any background noise
   - Ensure text is clearly visible
   - Maintain original colors

2. **File Organization**
   ```
   datasets/
   └── raw_maps/
       ├── map1.png
       ├── map2.png
       └── map3.png
   ```

## Running the System

### Command Line Usage

1. **Single Map Processing**

```bash
python src/detect.py datasets/raw_maps/map1.png
```

2. **Batch Processing**

```bash
python src/detect.py --batch datasets/raw_maps/
```

3. **Advanced Options**

```bash
python src/detect.py map.png --output custom_output --dpi 300 --debug
```

### Python API Usage

1. **Basic Processing**

```python
from src.detect import CadastralMapExtractor

extractor = CadastralMapExtractor()
results = extractor.extract_map_text("map.png")
```

2. **Custom Configuration**

```python
extractor = CadastralMapExtractor(
    max_image_size=3000,
    enable_gpu=True,
    debug_mode=True
)
```

## Understanding Results

### Output Files

1. **Visualization (`extraction_results.jpg`)**

   - Color-coded feature detection
   - Text annotations
   - Symbol markers

2. **Detailed Analysis (`cadastral_output_detailed.xlsx`)**

   - Sheet 1: Text Elements
     - Place names
     - Survey numbers
     - Coordinates
   - Sheet 2: Symbols
     - Type
     - Location
     - Confidence

3. **Simple Results (`cadastral_output.csv`)**
   - Basic text listing
   - Quick reference format

### Color Coding

| Feature     | Color | Example            |
| ----------- | ----- | ------------------ |
| Transport   | Red   | Highways, Railways |
| Water       | Blue  | Rivers, Lakes      |
| Settlements | Black | Towns, Villages    |
| Boundaries  | Green | Borders            |
| Terrain     | Brown | Contours           |

### Interpreting Results

1. **Text Detection**

   - Bold text: High confidence (>90%)
   - Regular text: Medium confidence (70-90%)
   - Italic text: Low confidence (<70%)

2. **Symbol Detection**
   - Solid circles: Definite matches
   - Dashed circles: Probable matches
   - Dotted circles: Possible matches

## Troubleshooting

### Common Issues

1. **Poor Text Recognition**

   - Check image resolution
   - Verify contrast levels
   - Ensure clean source image

2. **Missing Symbols**

   - Verify color accuracy
   - Check for overlapping features
   - Adjust detection thresholds

3. **System Performance**
   - Reduce image size
   - Enable GPU acceleration
   - Clear output directory

### Error Messages

| Error             | Cause             | Solution            |
| ----------------- | ----------------- | ------------------- |
| Image Load Failed | Invalid file path | Check file location |
| GPU Error         | CUDA issues       | Verify GPU setup    |
| Memory Error      | Image too large   | Reduce image size   |

### Getting Help

1. Check documentation:

   - Technical guide
   - API reference
   - Examples

2. Common solutions:

   - Restart application
   - Update dependencies
   - Clear cache

3. Contact support:
   - File issue on GitHub
   - Email support team
   - Join user forum
