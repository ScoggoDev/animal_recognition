import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

#Dataset
dataset_dir = "data/"

#Load images from directory
train_dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=(150, 150),  #Size images
    batch_size=100,          #Batch size, I put 100 cuz I did a 10x10 grid of images
    label_mode="categorical",  #This is categorical because there are 4 animals, if there were 2 it would be binary, I still dont know more labels
    subset="training",       #Use as atraining dataset
    validation_split=0.2,    #use 20% for validation and 80 for training
    seed=123                 #seed to replicate
)

#Show images
class_names = train_dataset.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(100):
        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[tf.argmax(labels[i]).numpy()])
        plt.axis("off")
plt.show()
