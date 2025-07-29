from src.captura import capturar_interactivo

def mostrar_menu():
    print("\n=== Techtools - ClasificadorIA: Menú Principal ===")
    print("1. Captura interactiva para img ok y rechazadas (cambiar opcion con tecla 'l')")
    print("2. Salir")

def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ").strip()


        if opcion == "1":
            capturar_interactivo()
        elif opcion == "2":
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida. Intenta nuevamente.")

if __name__ == "__main__":
    main()