# src/entrenar.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def entrenar_modelo():
    print("Iniciando entrenamiento del modelo...")

    ruta_datos = os.path.join(os.path.dirname(__file__), '..', 'data')
    img_size = (64, 64)
    batch_size = 32
    epochs = 10

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        ruta_datos,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        ruta_datos,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    print("Clases encontradas:", train_generator.class_indices)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    modelo_guardado = os.path.join(os.path.dirname(__file__), '..', 'modelo_capsulas.keras')
    model.save(modelo_guardado)
    print(f"✅ Modelo guardado exitosamente en: {modelo_guardado}")

    # Mostrar gráfico
    plt.plot(hist.history['accuracy'], label='Entrenamiento')
    plt.plot(hist.history['val_accuracy'], label='Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.title('Precisión del modelo')
    plt.show()