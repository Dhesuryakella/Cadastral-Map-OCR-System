# Testing Guide for Cadastral Map OCR System

## Prerequisites Check

1. Verify Python Installation:

```bash
python --version  # Should be 3.8 or higher
```

2. Check Dependencies:

```bash
pip list | findstr "opencv-python numpy pandas matplotlib easyocr torch"
```

3. Verify CUDA (if using GPU):

```bash
nvidia-smi  # Check GPU and CUDA version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## System Testing Steps

### 1. Basic GUI Launch Test

```bash
cd D:\OCR
python src/gui.py
```

Expected: GUI window should open with:

- File selection option
- Processing options
- Results display area

### 2. Text Detection Testing

1. Test Place Names Detection:

```bash
# Using GUI:
1. Click "Browse" and select M1.png
2. Enable "Detect Text" option
3. Click "Process Map"
4. Check output/cadastral_output.csv for place names

# Using Command Line:
python src/detect.py M1.png --text-only
```

Expected Results:

- Should detect 16 place names
- Check output/cadastral_output.csv
- Verify names like "Aadahal", "Al Khurd", etc.

2. Test Survey Numbers:

```bash
# Same process as above, verify:
- Numbers (4, 18, 20, 22, etc.)
- Check confidence scores in detailed Excel report
```

### 3. Symbol Detection Verification

1. Transport Features:

```bash
python src/detect.py M1.png --symbol-type transport
```

Expected:

- ~1,398 road segments
- Check output/M1_detected_symbols.jpg

2. Settlements:

```bash
python src/detect.py M1.png --symbol-type settlements
```

Expected:

- 2 locations (1 city, 1 village)
- Visual confirmation in extraction_results.jpg

3. Boundaries:

```bash
python src/detect.py M1.png --symbol-type boundaries
```

Expected:

- ~202 boundary segments
- Check boundary lines in visualization

### 4. Accuracy Testing

1. Run Validation:

```bash
python src/validate_detection.py M1.png
```

Check metrics in output/detection_accuracy.csv:

- Overall accuracy: ~93.5%
- False positive rate: ~2.1%
- False negative rate: ~4.4%

### 5. Output File Verification

1. Check CSV Output:

```bash
# View CSV content
type output\cadastral_output.csv
```

Verify:

- Header row present
- Place names column
- Survey numbers column

2. Check Excel Report:

```bash
# Open detailed Excel report
start output\cadastral_output_detailed.xlsx
```

Verify sheets:

- Text_Numbers sheet
- Symbols sheet
- Confidence scores

3. Check Visualizations:

```bash
# View detection visualization
start output\extraction_results.jpg
```

Verify:

- Detected text marked
- Symbols highlighted
- Color-coded categories

### 6. Performance Testing

1. Processing Time:

```bash
python src/evaluate_accuracy.py --timing
```

Expected:

- Average processing time: ~45 seconds
- Memory usage report

2. GPU Performance (if available):

```bash
python src/detect.py M1.png --gpu
```

Compare processing times:

- GPU vs CPU mode
- Memory utilization

### 7. Error Handling Tests

1. Test Invalid Input:

```bash
python src/detect.py nonexistent.png
```

Expected:

- Proper error message
- Error logged in logs/error.log

2. Test Corrupt Image:

```bash
python src/detect.py corrupt_image.jpg
```

Expected:

- Graceful error handling
- User-friendly error message

### 8. Configuration Testing

1. Modify Detection Parameters:

```bash
# Edit config/detection_config.yaml
# Change confidence thresholds
python src/detect.py M1.png --config modified_config.yaml
```

Verify:

- Changes reflected in results
- Performance impact

### 9. Batch Processing Test

1. Process Multiple Maps:

```bash
python src/process_map.py datasets/raw_maps/*.png
```

Verify:

- All maps processed
- Individual results generated
- Summary report created

### 10. Final Validation

1. Check All Output Files:

```bash
dir output
```

Verify presence of:

- cadastral_output.csv
- cadastral_output_detailed.xlsx
- extraction_results.jpg
- detection_accuracy.csv
- M1_detected_symbols.jpg

2. Check Logs:

```bash
type logs\ocr_system.log
```

Verify:

- Processing steps logged
- No unexpected errors

## Troubleshooting Common Issues

1. If GUI doesn't launch:

```bash
python -c "import tkinter; tkinter.Tk()"
```

2. If CUDA issues:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

3. If memory errors:

```bash
# Reduce batch size in config/detection_config.yaml
# Set max_image_size to 1500
```

## Performance Benchmarks

Expected performance metrics:

- Text Detection: 95% accuracy
- Symbol Detection: 92% accuracy
- IoU Score: 0.85
- Processing Time: 45 seconds (CPU) / 15 seconds (GPU)

## Contact Support

For issues and support:

- Check logs/error.log for detailed error messages
- Contact system administrator
- Reference documentation in docs/ directory
