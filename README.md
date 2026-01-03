# Handwritten Digit Recognition

Implementation of a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset.

## Project Structure

```
.
├── digit.png
├── LICENSE
├── model.keras
├── predict.py
├── README.md
├── requirements.txt
├── statistics
│   ├── diagnostics.png
│   └── summary.png
└── train.py
```

## Usage

1. Clone the repository and `cd` into it
```bash
git clone https://github.com/eko-071/digit-recognition.git
cd digit-recognition/
```

2. Save the image being classified in the folder as `digit.png`.
> Note: Input images should be grayscale with a black background and a white digit, similar to MNIST

3. Install required packages
```
pip install -r requirements.txt
```

4. Run the script
```
python predict.py
```

## Training the Model

To train the model from scratch and generate evaluation plots:
```
python train.py
```

The script:
- Trains a CNN on the MNIST dataset
- Uses 5-fold cross-validation
- Saves training statistics in `statistics/`
- Saves the trained model as `model.keras`

## Results

- Achieves ~99% accuracy on the MNIST dataset.
- `diagnostics.png` shows two plots: cross-entropy loss vs. epochs (top) and classification accuracy vs. epochs (bottom), with training and validation curves for each cross-validation fold.
- `summary.png` shows a box plot of classification accuracy values obtained from all cross-validation folds.
> Note: Training data is blue, and testing data is orange.

## Requirements

- `numpy`
- `matplotlib`
- `tensorflow`
- `scikit-learn`

## References

- [But what is a neural network? | Deep learning chapter 1 - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&t=2s)
- [How to Develop a CNN for MNIST Handwritten Digit Classification - Jason Brownlee](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)