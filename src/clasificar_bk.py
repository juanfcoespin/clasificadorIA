import cv2
import numpy as np
import os
import sys
import tensorflow as tf

# Agrega la carpeta src al path para importar entrenar_modelo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from entrenar import entrenar_modelo

MODELO_PATH = os.path.join(os.path.dirname(__file__), '..', 'modelo_capsulas.keras')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

def cargar_modelo():
    if not os.path.exists(MODELO_PATH):
        print("Modelo no encontrado. Entrenando uno nuevo...")
        entrenar_modelo()
    return tf.keras.models.load_model(MODELO_PATH)

def detectar_capsula(frame):
    # Convertir a escala de grises y aplicar desenfoque
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 50, 150)

    # Buscar contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 50 < w < 200 and 50 < h < 200:
            return frame[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def preprocesar_imagen(img):
    return cv2.resize(img, (64, 64)) / 255.0

def clasificar_en_vivo():
    modelo = cargar_modelo()
    clases = ['ok', 'rechazado']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cv2.namedWindow("Clasificador de Cápsulas")

    seleccionando = False
    x0, y0 = -1, -1
    rect_final = None

    def seleccionar_rectangulo(event, x, y, flags, param):
        nonlocal x0, y0, rect_final, seleccionando
        if event == cv2.EVENT_LBUTTONDOWN:
            seleccionando = True
            x0, y0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and seleccionando:
            rect_final = (x0, y0, x - x0, y - y0)
        elif event == cv2.EVENT_LBUTTONUP:
            seleccionando = False
            rect_final = (x0, y0, x - x0, y - y0)

    cv2.setMouseCallback("Clasificador de Cápsulas", seleccionar_rectangulo)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_mostrar = frame.copy()
        capsula, coords = detectar_capsula(frame)

        if capsula is not None:
            entrada = preprocesar_imagen(capsula)
            prediccion = modelo.predict(np.expand_dims(entrada, axis=0))[0]
            clase = np.argmax(prediccion)
            precision = prediccion[clase]

            color = (0, 255, 0) if clase == 0 else (0, 0, 255)
            x, y, w, h = coords
            cv2.rectangle(frame_mostrar, (x, y), (x + w, y + h), color, 2)

            texto = f"{clases[clase]} ({precision*100:.1f}%)"
            cv2.putText(frame_mostrar, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        elif rect_final:
            x, y, w, h = rect_final
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                cv2.rectangle(frame_mostrar, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame_mostrar, "Presiona 'o' (ok), 'r' (rechazado)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            cv2.putText(frame_mostrar, "No se ha identificado ningun objeto a clasificar", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame_mostrar, "Dibuja un rectangulo o presiona 'q' para salir", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Clasificador de Cápsulas", frame_mostrar)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key in [ord('o'), ord('r')] and rect_final:
            clase = 'ok' if key == ord('o') else 'rechazado'
            ruta = os.path.join(DATASET_PATH, clase)
            os.makedirs(ruta, exist_ok=True)
            x, y, w, h = rect_final
            roi = frame[y:y+h, x:x+w]
            nombre_archivo = os.path.join(ruta, f"capsula_{len(os.listdir(ruta))}.jpg")
            cv2.imwrite(nombre_archivo, roi)

            cv2.putText(frame_mostrar, "Presiona 'e' para reentrenar o 'c' para continuar", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("Clasificador de Cápsulas", frame_mostrar)

            # Esperar acción del usuario
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord('e'):
                    entrenar_modelo()
                    break
                elif k == ord('c'):
                    break

            rect_final = None  # limpiar selección

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)