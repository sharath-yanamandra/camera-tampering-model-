# camera-tampering-model
# Camera Tampering Detection System

A real-time computer vision system for detecting camera tampering and obstruction using RTSP camera feeds. The system analyzes chromaticity and gradient direction histograms to identify unauthorized interference with camera views.

## 🎯 Features

- **Real-time RTSP Stream Processing**: Connects to IP cameras via RTSP protocol
- **Dual Detection Methods**: 
  - Chromaticity analysis for color-based obstructions
  - Gradient direction analysis for physical displacement detection
- **Intelligent Alert System**: Visual and console alerts with tampering type classification
- **GUI Interface**: Modern Tkinter interface with controls and statistics
- **Automatic Logging**: JSON-based detection logs with timestamps
- **Connection Recovery**: Automatic RTSP reconnection on network issues
- **Statistical Monitoring**: Real-time FPS, detection rates, and performance metrics

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python numpy pillow
```

### Configuration

Edit the RTSP configuration in the script:

```python
RTSP_URL = "rtsp://admin:administrator@192.168.29.74:554/ch0_0.264"
RTSP_TIMEOUT = 10  # seconds
```

### Running the System

```bash
# Run full detection system
python tampering_detector.py

# Test RTSP connection only
python tampering_detector.py test

# Show help information
python tampering_detector.py help
```

## 🎮 Controls

### GUI Mode (when PIL/Pillow available)
- **Quit Button**: Stop the detection system
- **Reset Button**: Reset detector calibration
- **Save Stats Button**: Export current statistics to JSON

### Console Mode (OpenCV only)
- **'q' key**: Quit the system
- **'s' key**: Save statistics
- **'r' key**: Reset detector
- **Ctrl+C**: Force quit

## 🔧 Technical Details

### Detection Algorithm

The system uses a dual-histogram approach:

1. **Short-term Pool**: Maintains recent frames for immediate comparison
2. **Long-term Pool**: Stores reference frames for baseline establishment
3. **Chromaticity Analysis**: HSV color histogram comparison
4. **Gradient Analysis**: Sobel gradient direction histogram comparison

### Tampering Types Detected

- **Color Obstruction**: Spray paint, colored coverings
- **Physical Displacement**: Camera movement, angle changes
- **Severe Tampering**: Combined color and physical interference
- **Combined Tampering**: Moderate interference across multiple metrics

### Thresholds (Configurable)

```python
CHROMA_THRESHOLD = 45.0      # Color difference sensitivity
GRADIENT_THRESHOLD = 25.0    # Physical change sensitivity  
COMBINED_THRESHOLD = 35.0    # Combined metric threshold
```

## 📊 Output and Logging

### Console Output
- Real-time frame processing statistics
- Detection alerts with classification
- FPS and performance metrics
- Pool status and calibration progress

### JSON Logs
Automatic logging to `tampering_logs/` directory:

```json
{
  "frames_processed": 1250,
  "tampering_detections": 3,
  "chroma_violations": 2,
  "gradient_violations": 1,
  "average_chroma_diff": 12.34,
  "average_gradient_diff": 8.92,
  "fps": 28.5,
  "timestamp": "2025-07-08T14:30:15",
  "rtsp_url": "rtsp://..."
}
```

## 🏗️ Architecture

### Core Components

1. **CameraTamperingDetector**: Main detection engine
2. **HistogramQueue**: Circular buffer for frame storage
3. **TkinterDisplayManager**: Advanced GUI interface
4. **SimpleDisplayManager**: Basic OpenCV display

### Key Classes

```python
class CameraTamperingDetector:
    - detect_tampering()        # Main detection logic
    - chromaticity_diff()       # Color analysis
    - gradient_direction_diff() # Physical change analysis
    - reset_detector()          # Calibration reset
```

## 🔬 Algorithm Parameters

### Pool Sizes
- **Short-term Pool**: 15 frames (immediate analysis)
- **Long-term Pool**: 30 frames (baseline reference)
- **Update Interval**: Every 3 frames

### Processing Parameters
- **Frame Resize**: 300x300 pixels (for consistent processing)
- **Histogram Bins**: 16 bins for gradient analysis
- **Alert Duration**: 3 seconds visual alert
- **Reconnection Timeout**: 2-second intervals

## 🛠️ Customization

### Adjusting Sensitivity

For higher sensitivity (more detections):
```python
CHROMA_THRESHOLD = 30.0    # Lower = more sensitive
GRADIENT_THRESHOLD = 20.0  # Lower = more sensitive
```

For lower sensitivity (fewer false positives):
```python
CHROMA_THRESHOLD = 60.0    # Higher = less sensitive
GRADIENT_THRESHOLD = 40.0  # Higher = less sensitive
```

### Processing Resolution

Modify detection resolution for performance tuning:
```python
detector = CameraTamperingDetector(
    img_width=400,    # Higher = more accurate, slower
    img_height=400,   # Lower = faster, less accurate
)
```

## 🔍 Troubleshooting

### Common Issues

1. **RTSP Connection Failed**
   - Verify camera IP address and credentials
   - Check network connectivity
   - Ensure RTSP port (554) is accessible

2. **High False Positive Rate**
   - Increase detection thresholds
   - Allow longer calibration period
   - Check for environmental factors (lighting changes)

3. **Poor Performance**
   - Reduce processing resolution
   - Close unnecessary applications
   - Check CPU usage

### Debug Mode

Enable detailed logging by modifying the console update frequency:
```python
if (current_time - last_console_update) >= 1.0:  # More frequent updates
```

## 📈 Performance Optimization

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Network**: Stable connection to camera
- **Display**: Optional for headless operation

### Optimization Tips
1. Use lower resolution cameras when possible
2. Optimize network bandwidth
3. Run on dedicated hardware for production
4. Monitor system resources during operation

## 🔒 Security Considerations

- Store RTSP credentials securely
- Use encrypted connections when available
- Implement access controls for log files
- Regular security updates for dependencies


## 🤝 Contributing

For improvements or bug fixes:
1. Test thoroughly with various camera types
2. Validate detection accuracy
3. Document any parameter changes
4. Include performance impact analysis

## 📞 Support

For technical issues:
1. Check RTSP connection with test mode
2. Verify all dependencies are installed
3. Review console output for error messages
4. Examine saved log files for patterns
