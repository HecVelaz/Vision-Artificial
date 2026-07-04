import cv2
import numpy as np

# ============================================================
# CONFIGURACIÓN DE CÁMARA IP
# ============================================================

url = "http://192.168.0.2:8080/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo conectar a la cámara.")
    exit()

print("Cámara conectada. Presiona 'q' para salir.")


# ============================================================
# PARÁMETROS AJUSTABLES
# ============================================================

ANCHO_MOSTRAR = 1000

# Zoom digital
ZOOM = 1.8

# Filtros de tamaño
AREA_MIN = 700
AREA_MAX = 8000

# Filtro de forma circular
CIRCULARIDAD_MIN = 0.60
RATIO_MIN = 0.65
RATIO_MAX = 1.45


# ============================================================
# RANGOS HSV DE COLORES
# ============================================================

COLORES = {
    "Fucsia": {
        "lower": np.array([145, 50, 80]),
        "upper": np.array([179, 255, 255]),
        "bgr": (255, 0, 255)
    },

    "Azul": {
        "lower": np.array([85, 50, 50]),
        "upper": np.array([125, 255, 255]),
        "bgr": (255, 0, 0)
    },

    "Verde": {
        "lower": np.array([35, 50, 50]),
        "upper": np.array([85, 255, 255]),
        "bgr": (0, 255, 0)
    }
}


# ============================================================
# FUNCIONES
# ============================================================

def aplicar_zoom(frame, zoom=1.8):
    if zoom <= 1.0:
        return frame

    alto, ancho = frame.shape[:2]

    nuevo_ancho = int(ancho / zoom)
    nuevo_alto = int(alto / zoom)

    x1 = (ancho - nuevo_ancho) // 2
    y1 = (alto - nuevo_alto) // 2

    x2 = x1 + nuevo_ancho
    y2 = y1 + nuevo_alto

    recorte = frame[y1:y2, x1:x2]
    frame_zoom = cv2.resize(recorte, (ancho, alto))

    return frame_zoom


def detectar_pastillas_por_color(frame, hsv):
    detecciones = []

    for nombre_color, datos in COLORES.items():
        lower = datos["lower"]
        upper = datos["upper"]
        color_bgr = datos["bgr"]

        mask = cv2.inRange(hsv, lower, upper)

        # Limpieza de ruido
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contornos, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contornos:
            area = cv2.contourArea(cnt)

            if area < AREA_MIN or area > AREA_MAX:
                continue

            perimetro = cv2.arcLength(cnt, True)

            if perimetro == 0:
                continue

            circularidad = 4 * np.pi * area / (perimetro * perimetro)

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h)

            if circularidad < CIRCULARIDAD_MIN:
                continue

            if not (RATIO_MIN <= ratio <= RATIO_MAX):
                continue

            M = cv2.moments(cnt)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            detecciones.append({
                "color": nombre_color,
                "bgr": color_bgr,
                "contorno": cnt,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "area": area,
                "circularidad": circularidad
            })

    return detecciones


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al recibir frame.")
        break

    frame = aplicar_zoom(frame, ZOOM)

    alto_original, ancho_original = frame.shape[:2]
    escala = ANCHO_MOSTRAR / ancho_original
    nuevo_alto = int(alto_original * escala)

    frame = cv2.resize(frame, (ANCHO_MOSTRAR, nuevo_alto))

    output = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detecciones = detectar_pastillas_por_color(frame, hsv)

    conteo = {
        "Fucsia": 0,
        "Azul": 0,
        "Verde": 0
    }

    for i, det in enumerate(detecciones, start=1):
        color_nombre = det["color"]
        color_bgr = det["bgr"]

        conteo[color_nombre] += 1

        cnt = det["contorno"]
        x = det["x"]
        y = det["y"]
        w = det["w"]
        h = det["h"]
        cx = det["cx"]
        cy = det["cy"]
        area = det["area"]
        circularidad = det["circularidad"]

        cv2.drawContours(output, [cnt], -1, color_bgr, 2)
        cv2.rectangle(output, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.circle(output, (cx, cy), 5, (255, 255, 255), -1)

        cv2.putText(
            output,
            f"{i} {color_nombre}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr,
            2
        )

        cv2.putText(
            output,
            f"A={int(area)} C={circularidad:.2f}",
            (x, y + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color_bgr,
            1
        )

    total = len(detecciones)

    cv2.putText(
        output,
        f"TOTAL: {total}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Fucsia={conteo['Fucsia']} Azul={conteo['Azul']} Verde={conteo['Verde']}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Zoom={ZOOM} Area={AREA_MIN}-{AREA_MAX}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    # Máscara total para depuración
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for datos in COLORES.values():
        mask = cv2.inRange(hsv, datos["lower"], datos["upper"])
        mask_total = cv2.bitwise_or(mask_total, mask)

    cv2.imshow("Conteo y clasificacion de pastillas por color", output)
    cv2.imshow("Mascara total de colores", mask_total)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()