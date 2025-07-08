# Face Recognition Project

A simple face recognition system built with OpenCV and Python that can capture face data, train a model, and perform real-time face recognition using a webcam.

## Features

- **Face Detection**: Detect faces in real-time using Haar cascades
- **Data Collection**: Capture and store face data for training
- **Face Recognition**: Recognize faces using K-Nearest Neighbors (KNN) algorithm
- **Real-time Processing**: Live face recognition through webcam feed

## Project Structure

```
face_recognition_project/
├── face_detect.py          # Basic face detection and webcam testing
├── face_data.py           # Face data collection and dataset creation
├── face_recognition.py    # Face recognition implementation
├── face_dataset/          # Directory to store face datasets (.npy files)
├── haarcascade_frontalface_alt.xml  # Haar cascade classifier file
└── README.md             # This file
```

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- Webcam/Camera

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy
   ```
3. Download the Haar cascade file:
   - Download `haarcascade_frontalface_alt.xml` from OpenCV's GitHub repository
   - Place it in the project root directory

4. Create the dataset directory:
   ```bash
   mkdir face_dataset
   ```

## Usage

### 1. Test Your Camera (Optional)

First, test if your camera is working properly:

```bash
python face_detect.py
```

This will open a window showing your webcam feed. Press 'q' to quit.

### 2. Collect Face Data

To add a new person to the recognition system:

```bash
python face_data.py
```

- Enter the person's name when prompted
- Position your face in front of the camera
- The system will automatically capture face samples every 10 frames
- Move your head slightly to capture different angles
- Press 'q' when you have enough samples (recommended: 50-100 samples)

The face data will be saved as `[name].npy` in the `face_dataset/` directory.

### 3. Run Face Recognition

To start the face recognition system:

```bash
python face_recognition.py
```

- The system will load all face data from the `face_dataset/` directory
- Point your camera at faces to see real-time recognition
- Recognized faces will be labeled with the person's name
- Press 'q' to quit

## How It Works

### Face Detection
- Uses Haar cascade classifiers to detect faces in video frames
- Focuses on the largest detected face for better accuracy

### Data Collection
- Captures face regions and resizes them to 100x100 pixels
- Stores face data as flattened numpy arrays
- Applies offset padding around detected faces for better feature capture

### Face Recognition
- Uses K-Nearest Neighbors (KNN) algorithm with k=5
- Calculates Euclidean distance between face features
- Matches faces based on the most common label among nearest neighbors

## Configuration

You can modify these parameters in the code:

- **Face detection sensitivity**: Change `detectMultiScale` parameters in `face_cascade.detectMultiScale(gray_frame, 1.3, 5)`
- **Data collection frequency**: Modify `skip % 10 == 0` in `face_data.py`
- **Face image size**: Change `(100, 100)` in resize operations
- **KNN neighbors**: Adjust `k=5` parameter in the `knn` function

## Troubleshooting

### Common Issues

1. **Camera not working**: 
   - Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or higher numbers
   - Check if other applications are using the camera

2. **Haar cascade file not found**:
   - Ensure `haarcascade_frontalface_alt.xml` is in the project directory
   - Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades

3. **Poor recognition accuracy**:
   - Collect more training samples per person
   - Ensure good lighting conditions
   - Try different angles and expressions during data collection

4. **Face not detected**:
   - Check lighting conditions
   - Ensure face is clearly visible and not too close/far from camera
   - Adjust detection parameters

## Technical Details

- **Face Detection**: Haar cascade classifier (`haarcascade_frontalface_alt.xml`)
- **Feature Extraction**: Raw pixel values from 100x100 face images
- **Classification**: K-Nearest Neighbors with Euclidean distance
- **Real-time Processing**: OpenCV VideoCapture for webcam input

## Limitations

- Works best with good lighting conditions
- Requires frontal face orientation for optimal performance
- Recognition accuracy depends on training data quality and quantity
- Single face recognition per frame (focuses on largest detected face)

## Future Improvements

- Add support for multiple face recognition per frame
- Implement deep learning models for better accuracy
- Add face verification confidence scores
- Include data augmentation for better training
- Add GUI interface for easier use

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.