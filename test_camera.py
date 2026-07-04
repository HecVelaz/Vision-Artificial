import cv2

url = "http://172.16.238.214:8080/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Cámara conectada. Presioná q para salir.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer la imagen.")
        break

    frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Camara IP", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()