import cv2
import numpy as np
import os
import time
import tensorflow as tf
from src.entrenar import entrenar_modelo

# Carga el modelo fuera de src
modelo_path = os.path.join(os.path.dirname(__file__), '..', 'modelo_capsulas.keras')
if not os.path.exists(modelo_path):
    raise FileNotFoundError("No se encontro el modelo 'modelo_capsulas.keras' fuera de la carpeta src.")
modelo = tf.keras.models.load_model(modelo_path)

clases = ['ok', 'rechazado']
COLOR_OK = (0, 255, 0)
COLOR_RECHAZADO = (0, 0, 255)

seleccion_iniciada = False
punto_inicial = (0, 0)
punto_final = (0, 0)
rectangulo_listo = False
ultima_clasificacion = None
ultimo_porcentaje = 0
ultimo_tiempo_interaccion = time.time()

TIEMPO_MAX_ESPERA = 8  # segundos

def mostrar_mensaje(frame, texto, posicion=(10, 30), font_scale=0.8):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(texto, font, font_scale, 2)
    x, y = posicion
    # fondo negro
    cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
    # texto blanco
    cv2.putText(frame, texto, (x, y), font, font_scale, (255, 255, 255), 2)

def seleccionar_rectangulo(event, x, y, flags, param):
    global seleccion_iniciada, punto_inicial, punto_final, rectangulo_listo, ultima_clasificacion, ultimo_tiempo_interaccion

    if event == cv2.EVENT_LBUTTONDOWN:
        seleccion_iniciada = True
        punto_inicial = (x, y)
        punto_final = (x, y)
        rectangulo_listo = False
        ultima_clasificacion = None
        ultimo_tiempo_interaccion = time.time()

    elif event == cv2.EVENT_MOUSEMOVE and seleccion_iniciada:
        punto_final = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        seleccion_iniciada = False
        punto_final = (x, y)
        rectangulo_listo = True
        ultimo_tiempo_interaccion = time.time()

def clasificar_imagen(imagen):
    imagen = cv2.resize(imagen, (64, 64))
    imagen = imagen.astype('float32') / 255.0
    imagen = np.expand_dims(imagen, axis=0)  # (1, 64, 64, 3)
    predicciones = modelo.predict(imagen, verbose=0)[0]
    indice = np.argmax(predicciones)
    clase = clases[indice]
    porcentaje = predicciones[indice] * 100
    return clase, porcentaje

def guardar_imagen(imagen, clase):
    carpeta = os.path.join('data', clase)
    os.makedirs(carpeta, exist_ok=True)
    timestamp = int(time.time())
    filename = os.path.join(carpeta, f"capsula_{timestamp}.jpg")
    cv2.imwrite(filename, imagen)

def clasificar_en_vivo():
    global rectangulo_listo, ultima_clasificacion, ultimo_porcentaje, punto_inicial, punto_final, ultimo_tiempo_interaccion

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Clasificador de capsulas")
    cv2.setMouseCallback("Clasificador de capsulas", seleccionar_rectangulo)

    roi = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar instrucciones al iniciar
        if time.time() - ultimo_tiempo_interaccion > TIEMPO_MAX_ESPERA and not rectangulo_listo and not seleccion_iniciada:
            mostrar_mensaje(frame, "Instrucciones:", (10, 30), font_scale=0.9)
            mostrar_mensaje(frame, "Selecciona con el mouse la capsula para analizar", (10, 65), font_scale=0.8)
            mostrar_mensaje(frame, "Presiona 'o' para OK, 'r' para rechazado, 'e' para entrenar, 'c' para continuar", (10, 100), font_scale=0.7)
            mostrar_mensaje(frame, "Presiona 'q' o ESC para salir", (10, 135), font_scale=0.7)

        # Dibuja rectangulo azul mientras arrastras
        if seleccion_iniciada:
            cv2.rectangle(frame, punto_inicial, punto_final, (255, 0, 0), 2)

        # Cuando ya terminaste selección (mouse suelto)
        if rectangulo_listo:
            x1, y1 = punto_inicial
            x2, y2 = punto_final
            roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

            if roi.size > 0:
                try:
                    clase, porcentaje = clasificar_imagen(roi)
                    ultima_clasificacion = clase
                    ultimo_porcentaje = porcentaje
                    color = COLOR_OK if clase == 'ok' else COLOR_RECHAZADO

                    # Rectangulo verde o rojo con texto de resultado
                    cv2.rectangle(frame, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), color, 2)
                    mostrar_mensaje(frame, f"{clase.upper()} ({porcentaje:.1f}%)", (min(x1,x2), min(y1,y2) - 10))
                except Exception as e:
                    print(f"[ERROR] Clasificacion: {str(e)}")
                    mostrar_mensaje(frame, "Error al clasificar", (10, 60))

        if ultima_clasificacion:
            mostrar_mensaje(frame, "Presiona 'o' para OK, 'r' para rechazado", (10, frame.shape[0] - 50))
            mostrar_mensaje(frame, "Presiona 'e' para entrenar, 'c' para continuar", (10, frame.shape[0] - 20))

        cv2.imshow("Clasificador de capsulas", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27 or tecla == ord('q'):  # ESC o q para salir
            break
        elif tecla == ord('o') and ultima_clasificacion and roi is not None:
            guardar_imagen(roi, 'ok')
            print("[INFO] Imagen guardada en carpeta 'ok'")
        elif tecla == ord('r') and ultima_clasificacion and roi is not None:
            guardar_imagen(roi, 'rechazado')
            print("[INFO] Imagen guardada en carpeta 'rechazado'")
        elif tecla == ord('e'):
            mostrar_mensaje(frame, "Entrenando modelo...")
            cv2.imshow("Clasificador de capsulas", frame)
            cv2.waitKey(1)
            entrenar_modelo()
            rect_dibujo = False
            rect_x, rect_y, rect_w, rect_h = 0, 0, 0, 0
            mensaje = "Entrenamiento finalizado. Selecciona una capsula para analizar"
            mensaje_tiempo = time.time()
            # recargar modelo
            global modelo
            modelo = tf.keras.models.load_model(modelo_path)
            mostrar_mensaje(frame, "Modelo actualizado")
            cv2.imshow("Clasificador de capsulas", frame)
            cv2.waitKey(1000)
        elif tecla == ord('c'):
            ultima_clasificacion = None
            ultimo_porcentaje = 0
            rectangulo_listo = False
            roi = None

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    clasificar_en_vivo()