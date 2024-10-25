from tensorflow import keras
from tensorflow.keras import layers

#Defines the model as sequential, because it is a sequence of layers and the task is simple
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),  #Normalize pixels for images
    layers.Conv2D(32, (3, 3), activation='relu'),  #First convolutional layer, it has a 3x3 filter cuz it's more efficient and fast for training
    layers.MaxPooling2D(),  #This focus on the more important characteristics of the image
    layers.Conv2D(64, (3, 3), activation='relu'),  #Second convolutional layer with more neurons, we add more neurons to detect more complex patterns
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer with even more neurons
    layers.MaxPooling2D(),
    layers.Flatten(),  #I don't know what this does exactly... 
    layers.Dense(128, activation='relu'),  #Dense layer with 128 neurons
    layers.Dense(128, activation='relu'),  #Second dense layer with 128 neurons because the model is more complex, because we got 4 options to classify
    layers.Dense(4, activation='softmax')  #We needed 4 neurons, one for each option, softmax is to ctegorize with more than 2 options
])

#Compile model, we use adam optimizer, categorical crossentropy loss function and accuracy metric
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Model summary for debugging
model.summary()
