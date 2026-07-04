import cv2
import numpy as np

# ===============================
# 1) CONFIGURACIÓN DE LA CÁMARA
# ===============================

url = "http://172.16.238.214:8080/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Cámara conectada. Presioná 'q' para salir.")


# ===============================
# 2) DETECTAR ÁREA DEL BLÍSTER
# ===============================

def obtener_mascara_blister(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contornos, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros_like(gray)

    if contornos:
        contorno_max = max(contornos, key=cv2.contourArea)
        cv2.drawContours(mask, [contorno_max], -1, 255, -1)
    else:
        mask[:] = 255

    return mask


# ===============================
# 3) CLASIFICAR PASTILLAS POR COLOR
# ===============================

def detectar_colores_pastillas(img, mask_area):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.bitwise_and(hsv, hsv, mask=mask_area)

    # Kitadol: amarillo
    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([38, 255, 255])

    # Etidol: naranja claro
    lower_orange_light = np.array([14, 100, 100])
    upper_orange_light = np.array([22, 255, 255])

    # Alfazina: naranja oscuro
    lower_orange_dark = np.array([5, 120, 100])
    upper_orange_dark = np.array([13, 255, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange_light = cv2.inRange(hsv, lower_orange_light, upper_orange_light)
    mask_orange_dark = cv2.inRange(hsv, lower_orange_dark, upper_orange_dark)

    mask_total = cv2.bitwise_or(
        mask_yellow,
        cv2.bitwise_or(mask_orange_light, mask_orange_dark)
    )

    kernel = np.ones((5, 5), np.uint8)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_total = cv2.medianBlur(mask_total, 5)

    contornos, _ = cv2.findContours(
        mask_total,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contornos:
        area = cv2.contourArea(cnt)

        if area < 300:
            continue

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        mask_obj = np.zeros(mask_total.shape, np.uint8)
        cv2.drawContours(mask_obj, [cnt], -1, 255, -1)

        hue = cv2.mean(hsv, mask=mask_obj)[0]

        if 25 <= hue <= 38:
            tipo = "Kitadol"
            color_draw = (0, 255, 255)

        elif 14 <= hue < 22:
            tipo = "Etidol"
            color_draw = (0, 165, 255)

        elif 5 <= hue < 14:
            tipo = "Alfazina"
            color_draw = (0, 100, 255)

        else:
            tipo = "Desconocida"
            color_draw = (255, 255, 255)

        cv2.drawContours(img, [cnt], -1, color_draw, 2)
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
        cv2.putText(
            img,
            tipo,
            (cx - 40, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_draw,
            2
        )

        print(f"{tipo}: centro=({cx}, {cy}), hue={hue:.2f}, area={area:.1f}")

    return img


# ===============================
# 4) LOOP PRINCIPAL
# ===============================

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer la cámara.")
        break

    frame = cv2.resize(frame, (800, 600))

    mask_blister = obtener_mascara_blister(frame)
    resultado = detectar_colores_pastillas(frame.copy(), mask_blister)

    cv2.imshow("Clasificacion de pastillas", resultado)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()