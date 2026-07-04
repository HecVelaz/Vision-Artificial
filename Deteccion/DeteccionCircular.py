import cv2
import numpy as np
from collections import deque
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
# 1.0 = sin zoom
# 1.5 = zoom moderado
# 1.8 = recomendado
# 2.0 = zoom fuerte
ZOOM = 1.8

# Parámetros de HoughCircles
DP = 1.2
MIN_DIST = 40

PARAM1 = 110
PARAM2 = 46

MIN_RADIUS = 23
MAX_RADIUS = 38

# Filtro para evitar círculo interno + externo duplicado
DISTANCIA_CENTRO_DUPLICADO = 25

# Modo:
# "externo" = se queda con el círculo de mayor radio
# "interno" = se queda con el círculo de menor radio
MODO_CIRCULO = "externo"


# ============================================================
# FUNCIÓN DE ZOOM DIGITAL
# ============================================================

def aplicar_zoom(frame, zoom=1.8):
    """
    Aplica zoom digital recortando el centro de la imagen
    y redimensionando al tamaño original.
    """

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


# ============================================================
# FILTRO DE CÍRCULOS DUPLICADOS
# ============================================================

def filtrar_circulos_por_centro(circles, distancia_centro=25, modo="externo"):
    """
    HoughCircles puede detectar el círculo interno y externo
    de una misma cavidad.

    Esta función agrupa círculos con centros cercanos y deja
    solo uno por grupo.

    modo="externo" -> conserva el de mayor radio.
    modo="interno" -> conserva el de menor radio.
    """

    if circles is None:
        return []

    circles = np.uint16(np.around(circles))

    lista = []

    for c in circles[0, :]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        lista.append((cx, cy, r))

    grupos = []

    for circ in lista:
        cx, cy, r = circ
        agregado = False

        for grupo in grupos:
            gx, gy, gr = grupo[0]

            distancia = np.sqrt((cx - gx) ** 2 + (cy - gy) ** 2)

            if distancia < distancia_centro:
                grupo.append(circ)
                agregado = True
                break

        if not agregado:
            grupos.append([circ])

    circulos_filtrados = []

    for grupo in grupos:
        if modo == "externo":
            elegido = max(grupo, key=lambda c: c[2])
        else:
            elegido = min(grupo, key=lambda c: c[2])

        circulos_filtrados.append(elegido)

    # Ordenar de arriba hacia abajo y de izquierda a derecha
    circulos_filtrados = sorted(circulos_filtrados, key=lambda c: (c[1], c[0]))

    return circulos_filtrados

# ============================================================
# ESTABILIZACIÓN TEMPORAL
# ============================================================

historial_cantidades = deque(maxlen=10)
ultimos_circulos_validos = []

CANTIDAD_ESPERADA = 10
MIN_FRAMES_VALIDOS = 6
# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al recibir frame.")
        break

    # Aplicar zoom digital
    frame = aplicar_zoom(frame, ZOOM)

    # Redimensionar imagen
    alto_original, ancho_original = frame.shape[:2]
    escala = ANCHO_MOSTRAR / ancho_original
    nuevo_alto = int(alto_original * escala)

    frame = cv2.resize(frame, (ANCHO_MOSTRAR, nuevo_alto))

    output = frame.copy()

    # Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Suavizado
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # ========================================================
    # DETECCIÓN DE CÍRCULOS
    # ========================================================

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MIN_DIST,
        param1=PARAM1,
        param2=PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    cantidad_cruda = 0

    if circles is not None:
        cantidad_cruda = len(circles[0])

    circulos_filtrados = filtrar_circulos_por_centro(
        circles,
        distancia_centro=DISTANCIA_CENTRO_DUPLICADO,
        modo=MODO_CIRCULO
    )

    cantidad_filtrada = len(circulos_filtrados)
    # Guardar cantidad detectada en el historial
    historial_cantidades.append(cantidad_filtrada)

    # Contar cuántas veces apareció la cantidad esperada en los últimos frames
    veces_cantidad_esperada = historial_cantidades.count(CANTIDAD_ESPERADA)

    # Si se detectaron los 10 círculos, guardamos esa detección como válida
    if cantidad_filtrada == CANTIDAD_ESPERADA:
        ultimos_circulos_validos = circulos_filtrados.copy()

    # Si en este frame detecta 9, pero recientemente venía detectando 10,
    # usamos la última detección válida de 10 círculos
    if cantidad_filtrada != CANTIDAD_ESPERADA and veces_cantidad_esperada >= MIN_FRAMES_VALIDOS:
        circulos_filtrados = ultimos_circulos_validos.copy()
        cantidad_filtrada = len(circulos_filtrados)
    # ========================================================
    # DIBUJO DE CÍRCULOS FILTRADOS
    # ========================================================

    for i, circle in enumerate(circulos_filtrados, start=1):
        cx, cy, r = circle

        # Círculo externo filtrado
        cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)

        # Centro
        cv2.circle(output, (cx, cy), 4, (255, 0, 0), -1)

        # Número de cavidad
        cv2.putText(
            output,
            f"{i}",
            (cx - 8, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        # Radio
        cv2.putText(
            output,
            f"r={r}",
            (cx - 18, cy - r - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

        # Coordenadas
        cv2.putText(
            output,
            f"({cx},{cy})",
            (cx + 5, cy + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 0, 0),
            1
        )

        # Círculo interno de análisis sugerido
        radio_analisis = int(r * 0.65)
        cv2.circle(output, (cx, cy), radio_analisis, (255, 255, 0), 1)

    # ========================================================
    # TEXTO EN PANTALLA
    # ========================================================

    cv2.putText(
        output,
        f"Circulos crudos: {cantidad_cruda}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Circulos filtrados: {cantidad_filtrada}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Zoom={ZOOM} Param2={PARAM2} R={MIN_RADIUS}-{MAX_RADIUS}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Filtro: {MODO_CIRCULO} DistCentro={DISTANCIA_CENTRO_DUPLICADO}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    # ========================================================
    # MOSTRAR VENTANAS
    # ========================================================

    cv2.imshow("Deteccion de Circulos Filtrados", output)
    cv2.imshow("Imagen gris suavizada", gray_blur)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# LIBERAR RECURSOS
# ============================================================

cap.release()
cv2.destroyAllWindows()