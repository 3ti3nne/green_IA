import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Set mixed precision policy
# => utilise des opérations de calcul en demi-precision (float16)
# accelere l'entrainement tout en maintenant une precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Load data
# charge les données ensemble train et ensemble test, x = image y = etiquettes
# cifar10 ensemble de données
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
# normalise les valeurs des pixels en les divisant par
x_train, x_test = x_train / 50.0, x_test / 50.0

# Define model
# check différentes parties de l'image a la loupe
# conv => extract les caracteristiques des images en entrée
# maxPooling => recule un peu et examine l'essentiel de plus loin
# dense => transforme les caracteristiques extraites en représentation adaptée a la tache du modele

model = Sequential(
    [
        Conv2D(32, (3, 3), activation="ELU", input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="ELU"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="ELU"),
        Flatten(),
        Dense(300, activation="ELU"),
        Dense(
            100, dtype="float16"
        ),  # => reduit la précision des valeurs, économise mémoire et accélère calculs
    ]
)

# Compile and train model
# compile en specifiant l'optimiseur
model.compile(
    optimizer="RMSprop",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# train le model avec 10 epochs en utilisant l'algo d'opti
model.fit(x_train, y_train, validation_split=0.2, epochs=10)
