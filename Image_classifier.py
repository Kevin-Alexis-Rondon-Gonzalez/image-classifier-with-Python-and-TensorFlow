import tensorflow as tf 
import tensorflow_datasets as tfds  #Data to training 
import matplotlib.pyplot as plt
import math

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

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten (input_shape = (28,28,1)), #1 black adn whitE, 28X28 is the image size
    tf.keras.layers.Dense(50, activation = tf.nn.relu),
    tf.keras.layers.Dense(50, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax) #for sorting networks
])

# Compile Model
modelo.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

    #How training with small lote
num_ej_entrenamiento = metadatos.splits['train'].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples
# print(num_ej_entrenamiento, num_ej_pruebas)

tamano_lote = 32
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(tamano_lote)
datos_pruebas = datos_pruebas.batch(tamano_lote)


#Training 
historial = modelo.fit(datos_entrenamiento, epochs = 10, steps_per_epoch = math.ceil(num_ej_entrenamiento/tamano_lote))

#find loss
plt.xlabel('# Epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history["loss"])
plt.show()