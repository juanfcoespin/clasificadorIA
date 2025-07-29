import cv2
import os
from datetime import datetime

def capturar_interactivo(save_dir="data"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    label = "ok"

    print("\n=== Modo Interactivo de Captura ===")
    print("Presiona 'c' para capturar y recortar imagen.")
    print("Presiona 'l' para cambiar entre 'ok' y 'rechazado'.")
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break

        # Mostrar clase actual en la imagen
        cv2.putText(frame, f"Clase actual: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0) if label == "ok" else (0, 0, 255), 2)

        cv2.imshow("Captura Interactiva", frame)
        key = cv2.waitKey(1) & 0xFF  # Asegura lectura correcta en todos los sistemas

        if key == ord('c'):
            # Congelar el frame actual
            frozen_frame = frame.copy()
            cv2.destroyWindow("Captura Interactiva")  # Cierra ventana antes de seleccionar ROI

            print("\n➡ Selecciona la región con el mouse y presiona ENTER. ESC para cancelar.")
            roi = cv2.selectROI("Selecciona región", frozen_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Selecciona región")

            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                recorte = frozen_frame[y:y+h, x:x+w]

                output_dir = os.path.join(save_dir, label)
                os.makedirs(output_dir, exist_ok=True)
                filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                ruta_completa = os.path.join(output_dir, filename)
                cv2.imwrite(ruta_completa, recorte)

                print(f"\n✅ Imagen recortada guardada en: {ruta_completa}")
            else:
                print("\n⚠ Recorte cancelado.")

            # Reabrir ventana principal tras ROI
            cv2.namedWindow("Captura Interactiva")

        elif key == ord('l'):
            label = "rechazado" if label == "ok" else "ok"
            print(f"\n🔄 Clase cambiada a: {label}")

        elif key == ord('q'):
            print("\n🚪 Saliendo del modo interactivo...")
            break

    cap.release()
    cv2.destroyAllWindows()