import cv2
import os
from datetime import datetime

def capturar_interactivo(save_dir="data"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    label = "ok"
    instrucciones = [
        "c: Capturar imagen",
        "l: Cambiar clase",
        "q: Salir"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break

        # Mostrar clase actual
        cv2.putText(frame, f"Clase: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if label == "ok" else (0, 0, 255), 2)

        # Mostrar instrucciones
        y_offset = 60
        for linea in instrucciones:
            cv2.putText(frame, linea, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25

        cv2.imshow("Captura Interactiva", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            frozen_frame = frame.copy()
            cv2.destroyWindow("Captura Interactiva")


            # Copia del frame congelado para superponer el texto
            mensaje = "Selecciona la region con el mouse y presiona ESPACIO para capturarla. Caso contrario digita ESC"
            print("\n➡" + mensaje)
            cv2.putText(frozen_frame, mensaje, (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            roi = cv2.selectROI("Selecciona region", frozen_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Selecciona region")

            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                recorte = frozen_frame[y:y+h, x:x+w]
                output_dir = os.path.join(save_dir, label)
                os.makedirs(output_dir, exist_ok=True)
                filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                ruta_completa = os.path.join(output_dir, filename)
                cv2.imwrite(ruta_completa, recorte)
                print(f"\n✅ Imagen guardada en: {ruta_completa}")
            else:
                print("\n⚠ Recorte cancelado.")

            # Restaurar ventana de captura
            cv2.namedWindow("Captura Interactiva")

        elif key == ord('l'):
            label = "rechazado" if label == "ok" else "ok"
            print(f"\n🔄 Clase cambiada a: {label}")

        elif key == ord('q'):
            print("\n🚪 Saliendo del modo interactivo...")
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)