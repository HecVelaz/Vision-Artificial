import cv2
import numpy as np

# ===============================
# CONFIGURACIÓN DE LA CÁMARA IP
# ===============================

url = "http://192.168.0.2:8080/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo conectar a la cámara.")
    exit()

print("Cámara conectada. Presiona 'q' para salir.")


# ===============================
# FUNCIONES AUXILIARES
# ===============================

def area_interseccion(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    return inter_w * inter_h


def se_superpone_con_pastilla(hole_box, present_boxes, umbral=0.30):
    """
    Si el supuesto hueco se superpone con una pastilla detectada,
    se descarta para evitar falsos huecos.
    """
    hx, hy, hw, hh = hole_box
    area_hueco = hw * hh

    if area_hueco == 0:
        return False

    for pb in present_boxes:
        px, py, pw, ph, tipo = pb
        inter = area_interseccion(hole_box, (px, py, pw, ph))
        proporcion = inter / area_hueco

        if proporcion > umbral:
            return True

    return False


# ===============================
# BUCLE PRINCIPAL
# ===============================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al recibir frame.")
        break

    img = frame.copy()

    # ============================================
    # BLOQUE 1: PREPROCESAMIENTO
    # ============================================

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

    grad = cv2.magnitude(gx, gy)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    _, th = cv2.threshold(
        grad,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        cv2.imshow("Deteccion de Huecos y Pastillas", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # Contorno mayor = blíster
    c = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)

    blister_roi = cv2.bitwise_and(img, img, mask=mask)

    contours_mask, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    c_max = max(contours_mask, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c_max)

    offset_x = x
    offset_y = y

    blister_cropped = blister_roi[y:y + h, x:x + w]
    mask_cropped = mask[y:y + h, x:x + w]

    if blister_cropped.size == 0:
        continue

    # ============================================
    # BLOQUE 2: DETECCIÓN DE PASTILLAS PRESENTES
    # NUEVOS COLORES: FUCSIA, AZUL Y VERDE
    # ============================================

    image = blister_cropped.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # -------------------------------
    # Rangos HSV nuevos
    # -------------------------------

    # Fucsia / rosado
    lower_fucsia = np.array([150, 70, 80])
    upper_fucsia = np.array([175, 255, 255])

    # Azul
    lower_azul = np.array([90, 70, 70])
    upper_azul = np.array([110, 255, 255])

    # Verde
    lower_verde = np.array([45, 60, 70])
    upper_verde = np.array([70, 255, 255])

    mask_fucsia = cv2.inRange(hsv, lower_fucsia, upper_fucsia)
    mask_azul = cv2.inRange(hsv, lower_azul, upper_azul)
    mask_verde = cv2.inRange(hsv, lower_verde, upper_verde)

    # Máscara total de pastillas
    mask_pills = cv2.bitwise_or(mask_fucsia, mask_azul)
    mask_pills = cv2.bitwise_or(mask_pills, mask_verde)

    # Limpieza morfológica
    kernel = np.ones((7, 7), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_pills, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours_present, _ = cv2.findContours(
        mask_cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    present_boxes = []
    areas_pills = []

    for c in contours_present:
        area = cv2.contourArea(c)

        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Filtro de forma
        ratio = w / float(h)
        if ratio < 0.4 or ratio > 2.2:
            continue

        # Máscara individual de la pastilla
        mask_obj = np.zeros(mask_cleaned.shape, dtype=np.uint8)
        cv2.drawContours(mask_obj, [c], -1, 255, -1)

        hue_prom = cv2.mean(hsv, mask=mask_obj)[0]

        # Clasificación por color
        if 150 <= hue_prom <= 175:
            tipo = "Fucsia"
        elif 90 <= hue_prom <= 110:
            tipo = "Azul"
        elif 45 <= hue_prom <= 70:
            tipo = "Verde"
        else:
            tipo = "Desconocida"

        present_boxes.append((x, y, w, h, tipo))
        areas_pills.append(area)

    area_prom = np.median(areas_pills) if len(areas_pills) > 0 else 800

    # ============================================
    # BLOQUE 3: DETECCIÓN DE HUECOS
    # ============================================
    # Ahora el fondo blanco puede meter ruido, por eso:
    # - hacemos más estricta la máscara blanca
    # - descartamos huecos que se superponen con pastillas
    # - filtramos por forma y área

    lower_white = np.array([0, 0, 145])
    upper_white = np.array([180, 55, 255])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_white = cv2.bitwise_and(mask_white, mask_cropped)

    kernel = np.ones((7, 7), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours_holes, _ = cv2.findContours(
        mask_white,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    missing_boxes = []
    centroides_huecos = []

    for c in contours_holes:
        area = cv2.contourArea(c)

        # Filtro por tamaño parecido a una pastilla
        if area < 0.55 * area_prom or area > 1.50 * area_prom:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Evitar falsos huecos encima de pastillas detectadas
        if se_superpone_con_pastilla((x, y, w, h), present_boxes, umbral=0.30):
            continue

        ratio = w / float(h)

        # Para pastillas circulares o casi circulares
        if not (0.55 < ratio < 1.45):
            continue

        extent = area / float(w * h)

        # El hueco debe ocupar una parte razonable del rectángulo
        if extent < 0.35:
            continue

        shape = "hueco"

        M = cv2.moments(c)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cx_global = cx + offset_x
            cy_global = cy + offset_y

            centroides_huecos.append((cx, cy, cx_global, cy_global))
        else:
            cx, cy = 0, 0

        missing_boxes.append((x, y, w, h, shape))

    if len(centroides_huecos) > 0:
        for (cx, cy, cx_global, cy_global) in centroides_huecos:
            print(f"Hueco ROI=({cx},{cy}) | Frame=({cx_global},{cy_global})")

    # ============================================
    # BLOQUE 4: VISUALIZACIÓN EN TIEMPO REAL
    # ============================================

    output = image.copy()

    # Dibujar pastillas detectadas
    for (x, y, w, h, tipo) in present_boxes:
        if tipo == "Fucsia":
            color = (255, 0, 255)
        elif tipo == "Azul":
            color = (255, 0, 0)
        elif tipo == "Verde":
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        cv2.putText(
            output,
            tipo,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # Dibujar huecos detectados
    for (x, y, w, h, shape) in missing_boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.putText(
            output,
            shape,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    # Dibujar centroides
    for (cx, cy, cx_global, cy_global) in centroides_huecos:
        cv2.circle(output, (cx, cy), 4, (255, 0, 0), -1)

        cv2.putText(
            output,
            f"({cx},{cy})",
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 0, 0),
            1
        )

    cv2.imshow("Deteccion de Huecos y Pastillas - Colores Nuevos", output)

    # Ventanas de depuración
    cv2.imshow("Mascara pastillas color", mask_cleaned)
    cv2.imshow("Mascara huecos blancos", mask_white)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ===============================
# LIBERAR RECURSOS
# ===============================

cap.release()
cv2.destroyAllWindows()