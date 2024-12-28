# Character Recognition Using Image Moments

This project is a Python-based application that uses **shape-based recognition techniques** (Hu, R, and Zernike moments) to detect and recognize characters in images. It includes a user-friendly graphical interface built with `Tkinter` for easy image selection and character recognition.

## Features

- **Graphical User Interface**: Interactive GUI with modern styling using `ttk`.
- **Shape-Based Recognition**:
  - Hu Moments
  - R Moments
  - Zernike Moments
- **Comparison Methods**:
  - Three custom comparison algorithms to match extracted features.
- **Easy Configuration**: Dependencies managed via `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/elifpulukcu/Character-Recognition-Using-Image-Moments.git
   cd Character-Recognition-Using-Image-Moments
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## How It Works

1. **Image Selection**: Use the "Select Image" button to upload an image.
2. **Feature Extraction**:
   - The selected image is thresholded and segmented into connected components.
   - Shape-based moments (Hu, R, or Zernike) are computed for each detected character.
3. **Comparison**: Extracted features are compared against a database of pre-stored feature vectors to determine the recognized characters.
4. **Results Display**:
   - Recognized characters are displayed on the GUI.
   - Bounding boxes are drawn around the detected characters in the input image.

## Planned Improvements

This project is functional, but there are areas where further improvements can be made:

1. **Accuracy Enhancements**:
   - Investigating incorrect predictions for certain images.
   - Exploring alternative feature extraction methods, such as:
     - Gradient-based methods (e.g., HOG).
     - Neural network-based approaches (e.g., CNNs).
   - Improving the comparison algorithms to handle more diverse character sets.

2. **Preprocessing Pipeline**:
   - Adding adaptive thresholding techniques for better image segmentation.
   - Incorporating morphological operations to clean noisy images.

3. **Data Expansion**:
   - Expanding the reference database (`.npy` files) with more fonts, sizes, and variations.
   - Creating a robust dataset for diverse testing.

4. **Error Feedback Mechanism**:
   - Providing users with a way to correct predictions and save this data for future model training.

## Project Structure

```
Character-Recognition-Using-Image-Moments/
 ├── Database/
 │    ├── sourceHu0.npy
 │    ├── sourceHu1.npy
 │    └── ...
 ├── main.py
 ├── LICENSE
 ├── .gitignore
 ├── requirements.txt
 └── README.md
```

## Requirements

- Python 3.7+
- Required Python packages are listed in `requirements.txt`:
  - Pillow
  - numpy

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Developed by [Elif Pulukçu](https://github.com/elifpulukcu).
