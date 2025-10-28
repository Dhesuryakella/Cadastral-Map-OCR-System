# Cadastral Map OCR System Documentation

## System Overview

The Cadastral Map OCR System is designed to extract and analyze information from cadastral maps using computer vision and OCR techniques. The system processes maps to identify and extract:

- Place names
- Survey numbers
- Map symbols (water bodies, terrain features, transport routes)

## Components

### 1. Text Detection & Recognition

#### Place Names Detection
- Uses EasyOCR for text recognition
- Handles multiple text colors and orientations
- Confidence scoring for each detection

#### Survey Numbers Detection
- Specialized detection for red-colored numbers
- Number validation and classification
- Location tracking for spatial analysis

### 2. Symbol Detection

#### Water Bodies
- Template matching for water features
- Area calculation
- Blue color range detection

#### Terrain Features
- Pattern recognition for terrain symbols
- Area and type classification
- Elevation indication detection

#### Transport Routes
- Line detection for roads and railways
- Length calculation
- Connection point analysis

### 3. Graphical User Interface

#### Main Window
- File selection interface
- Processing progress display
- Results visualization

#### Results Display
- Tabbed interface for different feature types
- Sortable data tables
- Location coordinates display

## File Structure

```
src/
├── detect.py           # Main detection logic
├── gui.py             # GUI implementation
├── symbol_detector.py # Symbol detection
└── ...

datasets/
├── raw_maps/         # Input maps
├── symbols/          # Symbol templates
└── training_data/    # Training data
```

## Configuration

### Detection Parameters
Located in `config/detection_params.json`:
- Text confidence thresholds
- Symbol matching parameters
- Color range definitions

### Model Settings
Located in `config/model_config.json`:
- OCR model parameters
- GPU/CPU settings
- Batch processing options

## Output Formats

### JSON Output
```json
{
    "characters": [
        {
            "name": "Place Name",
            "confidence": 0.95,
            "location": [x, y]
        }
    ],
    "numbers": [
        {
            "number": "123",
            "confidence": 0.98,
            "location": [x, y]
        }
    ],
    "symbols": {
        "water": [...],
        "terrain": [...],
        "transport": [...]
    }
}
```

### Excel Output
- Sheet 1: Place Names
- Sheet 2: Survey Numbers
- Sheet 3: Map Features

## Performance Optimization

### CPU Mode
- Batch size: 2
- Thread count: 4
- Memory optimization settings

### GPU Mode (if available)
- CUDA optimization
- Batch size: 4
- cuDNN settings

## Error Handling

### Common Issues
1. Image loading errors
2. OCR confidence issues
3. Symbol detection failures

### Logging
- Processing logs in `logs/processing/`
- Error logs in `logs/errors/`
- Debug information in `logs/debug/`

## Best Practices

### Input Images
- Recommended resolution: 1000-2000 pixels
- Supported formats: PNG, JPG
- Clear, high-contrast scans

### Processing
- Regular template updates
- Confidence threshold adjustment
- Regular log monitoring

## Maintenance

### Regular Tasks
1. Update symbol templates
2. Clean log files
3. Backup configuration files

### Updates
1. Check for EasyOCR updates
2. Update detection parameters
3. Refresh training data 
