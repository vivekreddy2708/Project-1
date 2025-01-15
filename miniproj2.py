import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load and preprocess tf.keras.datasets.mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
image=x_train[5]

# Reshape and normalize the images
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))  # Adding channel dimension
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))  # Adding channel dimension
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]
plt.imshow(image,cmap='gray')
plt.show()
# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 digit classes (0-9)
])

# First convolutional block


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")


# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 28x28 pixels (as the model expects this size)
    img_resized = cv2.resize(image, (28, 28))
    # Normalize the image to range [0, 1]
    img_normalized = img_resized / 255.0
    # Reshape to the format (1, 28, 28, 1)
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)
    return img_reshaped

# Function for prediction
def predict_digit(image, model):
    # Preprocess the image (resize, normalize)
    processed_image = preprocess_image(image)
    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    probability = np.max(predictions)
    return predicted_class, probability

# Open webcam (use 0 for the default webcam)
cap = cv2.VideoCapture(2)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate the frame for better viewing (optional)
    

    # Convert the frame to grayscale for better digit recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bw_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)
    ibw_frame = cv2.bitwise_not(bw_frame)

    # Define the region of interest (ROI) for digit recognition
    bbox_size = (60, 60)
    center_x, center_y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    bbox = [
        (center_x - bbox_size[0] // 2, center_y - bbox_size[1] // 2),
        (center_x + bbox_size[0] // 2, center_y + bbox_size[1] // 2)
    ]
    
    # Crop the region of interest (ROI) where the handwritten digit is expected to appear
    roi = ibw_frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    # Get prediction for the ROI
    digit, prob = predict_digit(roi, model)
    prediction_text = f"Predicted: {digit}"
    probability_text = f"Probability: {prob*100:.2f}%"

    # Draw the bounding box and display the prediction
    color = (255, 255, 255) 
    cv2.rectangle(ibw_frame, bbox[0], bbox[1], color, 3)
    result_img = np.zeros((200, 400, 3), dtype=np.uint8)  # Blank canvas for text
    cv2.putText(result_img, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_img, probability_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the result in a separate window
    cv2.imshow("Prediction & Probability", result_img)

    # Display the grayscale webcam feed
    cv2.imshow('Handwritten Digit Recognition (Grayscale)', ibw_frame)

    # Display the frame with prediction information
   

    # Exit the loop when 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
