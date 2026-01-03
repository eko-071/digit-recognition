import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def prepare_image(image_path):
    image = load_img(image_path, color_mode="grayscale", target_size=(28,28))
    image = img_to_array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype("float32") / 255.0

    return image

def predict_digit(image_path):
    model = load_model("model.keras")
    image = prepare_image(image_path)
    predictions = model.predict(image, verbose=0)
    digit = np.argmax(predictions)
    return digit

if __name__ == "__main__":
    image_path = "digit.png"
    digit = predict_digit(image_path)
    print(f"Predicted Digit: {digit}")
