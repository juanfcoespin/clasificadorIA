from src.captura import capturar_interactivo
from src.entrenar import entrenar_modelo
from src.clasificar import clasificar_en_vivo

def mostrar_menu():
    print("\n=== Techtools - ClasificadorIA: Menú Principal ===")
    print("1. Captura interactiva para img ok y rechazadas (cambiar opcion con tecla 'l')")
    print("2. Entrenar modelo con imágenes existentes")
    print("3. Clasificación en vivo con cámara")
    print("4. Salir")

def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ").strip()


        if opcion == "1":
            capturar_interactivo()
        elif opcion == "2":
            entrenar_modelo()
        elif opcion == "3":
            clasificar_en_vivo()
        elif opcion == "4":
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida. Intenta nuevamente.")

if __name__ == "__main__":
    main()