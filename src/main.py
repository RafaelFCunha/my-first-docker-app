import os
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_from_tensorflow():
    """
    Function to download and extract horse or human dataset from TensorFlow website.

    This function downloads two zip files containing the horse or human dataset and the validation dataset from TensorFlow.
    Then, it extracts the contents of the 'horse-or-human.zip' file to the '/tmp/horse-or-human' directory.

    Returns:
    Tuple: (path_to_horse_dir, path_to_human_dir)
    """
    # Removing existing zip files if they exist
    if os.path.exists('horse-or-human.zip'):
        os.remove('horse-or-human.zip')
    if os.path.exists('validation-horse-or-human.zip'):
        os.remove('validation-horse-or-human.zip')

    # Removing existing directories if they exist
    if os.path.exists('/tmp/horse-or-human'):
        shutil.rmtree('/tmp/horse-or-human')

    # Downloading horse-or-human dataset
    print("Downloading horse-or-human dataset...")
    !wget https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip
    
    # Downloading validation horse-or-human dataset
    print("Downloading validation horse-or-human dataset...")
    !wget https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip

    # Extracting the downloaded dataset
    print("Extracting the downloaded dataset...")
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/horse-or-human')
    zip_ref.close()
    print("Successfully extracted 'horse-or-human.zip' file.")

    # Directory with horse training images
    train_horse_dir = '/tmp/horse-or-human/horses'

    # Directory with human training images
    train_human_dir = '/tmp/horse-or-human/humans'

    print("Paths to horse and human directories:")
    print("Horse directory:", train_horse_dir)
    print("Human directory:", train_human_dir)

    return train_horse_dir, train_human_dir

def create_model():
    """
    Function to create a convolutional neural network model for image classification.

    This function configures the parameters of the model including several convolutional layers,
    max pooling layers, and dense layers for classification.

    Returns:
    tf.keras.models.Sequential: The configured convolutional neural network model.
    """
    print("Creating the model...")
    
    # Configuring parameters of the model
    model = tf.keras.models.Sequential([
        # Desired input shape is 300x300 with 3 color channels
        # First convolution (filters over each pixel of the image)
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Fourth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Fifth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        # 512 hidden neuron layers
        tf.keras.layers.Dense(512, activation='relu'),
        # Single output neuron. It will contain values from 0-1 where 0 is for the class ('horses') and 1 for the other class ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    print("Model created successfully.")
    return model

def compile_and_generate_data(model):
    """
    Function to compile the model and generate data for training.

    This function compiles the model with specified loss function, optimizer, and metrics.
    It also creates an ImageDataGenerator for preprocessing images and generates training data.

    Args:
    model (tf.keras.models.Sequential): The configured convolutional neural network model.

    Returns:
    tf.keras.preprocessing.image.DirectoryIterator: The generator for training data.
    """
    print("Compiling the model...")
    
    # Compiling the model
    model.compile(loss='binary_crossentropy',  
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['acc'])
    
    print("Model compiled successfully.")
    
    print("Generating training data...")
    
    # ImageDataGenerator for preprocessing images
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow training images in batches using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            '/tmp/horse-or-human/',  
            target_size=(300, 300),  
            batch_size=128,
            class_mode='binary')
    
    print("Training data generated successfully.")
    
    return train_generator

def train_and_save_model(model, train_generator, epochs=15, steps_per_epoch=8):
    """
    Function to train the model and save it.

    This function trains the model using the provided generator for training data.
    It specifies the number of epochs and steps per epoch for training.
    After training, it saves the trained model for future use.

    Args:
    model (tf.keras.models.Sequential): The configured convolutional neural network model.
    train_generator (tf.keras.preprocessing.image.DirectoryIterator): The generator for training data.
    epochs (int): Number of epochs for training. Default is 15.
    steps_per_epoch (int): Number of steps per epoch for training. Default is 8.

    Returns:
    tf.keras.callbacks.History: Object containing training history.
    """
    print("Training the model...")
    
    # Training the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,  
        epochs=epochs,
        verbose=1)
    
    print("Model trained successfully.")

    print("Saving the model...")
    
    # Saving the trained model
    model.save('classificador.h5')
    
    print("Model saved successfully.")
    
    return history

def execute():
    """
    Encapsulate previous functions and train the model
    """
    train_horse_dir, train_human_dir = get_data_from_tensorflow()
    model = create_model()
    train_generator = compile_and_generate_data(model)
    history = train_and_save_model(model, train_generator, epochs=15, steps_per_epoch=8)
    print("Model Trained")

if __name__ == "__main__":
    execute()