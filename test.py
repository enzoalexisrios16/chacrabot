import cv2
from ultralytics import YOLO

# Prueba con 0; si no abre, cambia a 1 o 2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: No se pudo abrir la cámara. Revisa permisos o cambia el índice a 1 o 2.")
    exit()

print("✅ Cámara conectada. Cargando modelo...")
model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al leer fotograma.")
        break

    # Detección general (todas las clases para probar)
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Prueba Camara", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()