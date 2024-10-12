from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
import os


# Image augmentation for training set
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training data from 'dataset/train'
train_data = datagen.flow_from_directory(
    'backend/dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Test data from 'dataset/test'
test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'backend/dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'

)

# Load MobileNetV2 model pre-trained on ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  # Modify '10' based on your number of food categories

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to ensure everything is correct
model.summary()

# Train the model
history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=15,  # You can adjust the number of epochs
    validation_data=test_data,
    validation_steps=len(test_data)
)

# Evaluate model on test data
test_loss, test_acc = model.evaluate(test_data)
print(f'Test accuracy: {test_acc:.2f}')

# Save the trained model
model.save('food_classification_model.h5')

# Load the saved model
model = load_model('food_classification_model.h5')

# Define the path to the directory containing your test images
test_images_dir = 'backend/dataset/test_images'  # Updated path to your test images directory

# List to store predictions
predictions = []

# Loop through each image in the test_images directory
for img_file in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_file)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)

    # Append the prediction to the list
    predictions.append((img_file, predicted_class[0]))

# Print the predictions
for img_name, pred in predictions:
    print(f'Image: {img_name}, Predicted class: {pred}')

