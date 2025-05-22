import tensorflow as tf 
import tensorflow_datasets as tfds  #Data to training 
import matplotlib.pyplot as plt

datos, metadatos = tfds.load('fashion_mnist', as_supervised = True, with_info = True)

datos_entrenamiento, datos_pruebas = datos['train'], datos ['test']

nombres_clases = metadatos.features['label'].names

# print(nombres_clases), show names of the labels 
def normalizar (imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255 #here switch 0-255 to 0-1
    return imagenes, etiquetas

#Normalize the data with the funtion that just create 
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

#add to cache (use cache memory and not at local disk, to faster training)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

#show image from training data

for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen = imagen.numpy().reshape((28,28)) #resize

#paint image
plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#paint all category
for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
    imagen - imagen.numpy().reshape((28,28))
    plt.subplot(5,5,i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(nombres_clases[etiqueta])
plt.show()

#Create Model

modelo = tf.kera.Sequential([
    tf.kera.layer.Flatten (input_shape = (28,28,1)), #1 black adn whitE, 28X28 is the image size
    tf.keras.layer.Dense(50, activation = tf.nn.relu),
    tf.keras.layer.Dense(50, activation = tf.nn.relu),
    tf.keras.layer.Dense(10, activation = tf.nn.softmax), #for sorting networks
])