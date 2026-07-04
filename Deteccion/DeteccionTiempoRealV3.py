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
ZOOM = 1.8

# HoughCircles
DP = 1.2
MIN_DIST = 40

PARAM1 = 110
PARAM2 = 48

MIN_RADIUS = 20
MAX_RADIUS = 45

# Filtro de duplicados
DISTANCIA_CENTRO_DUPLICADO = 25
MODO_CIRCULO = "externo"

# Estabilización temporal
CANTIDAD_ESPERADA = 10
MIN_FRAMES_VALIDOS = 6
historial_cantidades = deque(maxlen=10)
ultimos_circulos_validos = []

# Umbral de ocupación por color
# Si dentro del círculo hay más de este porcentaje de color, se considera ocupado
UMBRAL_COLOR = 0.15

# Radio interno para analizar color
# Usamos solo el centro de la cavidad para evitar bordes negros o reflejos
FACTOR_RADIO_ANALISIS = 0.65


# ============================================================
# RANGOS HSV DE PASTILLAS
# ============================================================

# Fucsia / rosado
lower_fucsia = np.array([150, 70, 80])
upper_fucsia = np.array([175, 255, 255])

# Azul
lower_azul = np.array([85, 50, 50])
upper_azul = np.array([125, 255, 255])

# Verde
lower_verde = np.array([45, 60, 70])
upper_verde = np.array([70, 255, 255])


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


def filtrar_circulos_por_centro(circles, distancia_centro=25, modo="externo"):
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

    circulos_filtrados = sorted(circulos_filtrados, key=lambda c: (c[1], c[0]))

    return circulos_filtrados


def analizar_color_en_circulo(hsv, cx, cy, radio):
    """
    Analiza si dentro del círculo hay fucsia, azul o verde.
    Devuelve:
    - tipo detectado
    - porcentaje de ocupación
    """

    mask_circulo = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.circle(mask_circulo, (cx, cy), radio, 255, -1)

    pixeles_totales = cv2.countNonZero(mask_circulo)

    if pixeles_totales == 0:
        return "hueco", 0.0, None

    mask_fucsia = cv2.inRange(hsv, lower_fucsia, upper_fucsia)
    mask_azul = cv2.inRange(hsv, lower_azul, upper_azul)
    mask_verde = cv2.inRange(hsv, lower_verde, upper_verde)

    pixeles_fucsia = cv2.countNonZero(
        cv2.bitwise_and(mask_fucsia, mask_fucsia, mask=mask_circulo)
    )

    pixeles_azul = cv2.countNonZero(
        cv2.bitwise_and(mask_azul, mask_azul, mask=mask_circulo)
    )

    pixeles_verde = cv2.countNonZero(
        cv2.bitwise_and(mask_verde, mask_verde, mask=mask_circulo)
    )

    ocupacion_fucsia = pixeles_fucsia / pixeles_totales
    ocupacion_azul = pixeles_azul / pixeles_totales
    ocupacion_verde = pixeles_verde / pixeles_totales

    ocupaciones = {
        "Fucsia": ocupacion_fucsia,
        "Azul": ocupacion_azul,
        "Verde": ocupacion_verde
    }

    tipo = max(ocupaciones, key=ocupaciones.get)
    ocupacion = ocupaciones[tipo]

    if ocupacion >= UMBRAL_COLOR:
        return tipo, ocupacion, ocupaciones
    else:
        return "Hueco", ocupacion, ocupaciones


def color_bgr_por_tipo(tipo):
    if tipo == "Fucsia":
        return (255, 0, 255)
    elif tipo == "Azul":
        return (255, 0, 0)
    elif tipo == "Verde":
        return (0, 255, 0)
    else:
        return (0, 0, 255)
def filtrar_por_dos_columnas(circulos, tolerancia_columna=45):
    """
    Filtra círculos falsos que aparecen entre las dos columnas del blíster.

    El blíster real tiene solo 2 columnas.
    Se estiman las columnas usando los círculos más a la izquierda
    y más a la derecha.
    """

    if len(circulos) == 0:
        return []

    # Ordenar por X
    ordenados_x = sorted(circulos, key=lambda c: c[0])

    # Tomar extremos para estimar las columnas
    n = len(ordenados_x)
    n_extremos = min(4, max(2, n // 3))

    izquierda = ordenados_x[:n_extremos]
    derecha = ordenados_x[-n_extremos:]

    x_col_izq = int(np.median([c[0] for c in izquierda]))
    x_col_der = int(np.median([c[0] for c in derecha]))

    filtrados = []

    for c in circulos:
        cx, cy, r = c

        dist_izq = abs(cx - x_col_izq)
        dist_der = abs(cx - x_col_der)

        # Aceptar solo si está cerca de una de las dos columnas reales
        if dist_izq <= tolerancia_columna or dist_der <= tolerancia_columna:
            filtrados.append(c)

    # Ordenar de arriba hacia abajo y de izquierda a derecha
    filtrados = sorted(filtrados, key=lambda c: (c[1], c[0]))

    return filtrados

def filtrar_por_grilla_auto(circulos, cantidad_esperada=10, tolerancia_x=45, tolerancia_y=45):
    """
    Filtra círculos falsos usando la geometría esperada del blíster.

    Si el blíster está vertical:
        2 columnas x 5 filas

    Si el blíster está horizontal:
        5 columnas x 2 filas

    Para blíster de 12:
        vertical: 2 columnas x 6 filas
        horizontal: 6 columnas x 2 filas
    """

    if len(circulos) == 0:
        return [], "desconocida", 0, 0

    xs = [c[0] for c in circulos]
    ys = [c[1] for c in circulos]

    ancho = max(xs) - min(xs)
    alto = max(ys) - min(ys)

    # Determinar orientación
    if ancho > alto:
        orientacion = "horizontal"

        if cantidad_esperada == 10:
            columnas = 5
            filas = 2
        elif cantidad_esperada == 12:
            columnas = 6
            filas = 2
        else:
            columnas = 5
            filas = 2

    else:
        orientacion = "vertical"

        if cantidad_esperada == 10:
            columnas = 2
            filas = 5
        elif cantidad_esperada == 12:
            columnas = 2
            filas = 6
        else:
            columnas = 2
            filas = 5

    # Crear líneas ideales de grilla
    x_refs = np.linspace(min(xs), max(xs), columnas)
    y_refs = np.linspace(min(ys), max(ys), filas)

    celdas = {}

    for c in circulos:
        cx, cy, r = c

        idx_col = int(np.argmin(np.abs(x_refs - cx)))
        idx_fila = int(np.argmin(np.abs(y_refs - cy)))

        dx = abs(cx - x_refs[idx_col])
        dy = abs(cy - y_refs[idx_fila])

        # Rechazar círculos que caen entre filas/columnas reales
        if dx > tolerancia_x or dy > tolerancia_y:
            continue

        key = (idx_fila, idx_col)

        # Si hay más de un círculo en la misma celda,
        # conservar el de mayor radio
        if key not in celdas:
            celdas[key] = c
        else:
            if r > celdas[key][2]:
                celdas[key] = c

    # Ordenar por fila y columna
    circulos_filtrados = []

    for fila in range(filas):
        for col in range(columnas):
            key = (fila, col)
            if key in celdas:
                circulos_filtrados.append(celdas[key])

    return circulos_filtrados, orientacion, filas, columnas


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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
    # Filtro geométrico automático:
    # vertical  -> 2 columnas x 5 filas
    # horizontal -> 5 columnas x 2 filas
    circulos_filtrados, orientacion_blister, filas_blister, columnas_blister = filtrar_por_grilla_auto(
    circulos_filtrados,
    cantidad_esperada=CANTIDAD_ESPERADA,
    tolerancia_x=45,
    tolerancia_y=45
    )
    cantidad_filtrada = len(circulos_filtrados)

    # ========================================================
    # ESTABILIZACIÓN TEMPORAL
    # ========================================================

    historial_cantidades.append(cantidad_filtrada)
    veces_cantidad_esperada = historial_cantidades.count(CANTIDAD_ESPERADA)

    if cantidad_filtrada == CANTIDAD_ESPERADA:
        ultimos_circulos_validos = circulos_filtrados.copy()

    if cantidad_filtrada != CANTIDAD_ESPERADA and veces_cantidad_esperada >= MIN_FRAMES_VALIDOS:
        circulos_filtrados = ultimos_circulos_validos.copy()
        cantidad_filtrada = len(circulos_filtrados)

    # ========================================================
    # ANÁLISIS DE COLOR POR CÍRCULO
    # ========================================================

    resultados = []

    for i, circle in enumerate(circulos_filtrados, start=1):
        cx, cy, r = circle

        radio_analisis = int(r * FACTOR_RADIO_ANALISIS)

        tipo, ocupacion, ocupaciones = analizar_color_en_circulo(
            hsv,
            cx,
            cy,
            radio_analisis
        )

        resultados.append({
            "id": i,
            "cx": cx,
            "cy": cy,
            "r": r,
            "radio_analisis": radio_analisis,
            "tipo": tipo,
            "ocupacion": ocupacion
        })

    # ========================================================
    # DIBUJO
    # ========================================================

    for res in resultados:
        cx = res["cx"]
        cy = res["cy"]
        r = res["r"]
        radio_analisis = res["radio_analisis"]
        tipo = res["tipo"]
        ocupacion = res["ocupacion"]

        color = color_bgr_por_tipo(tipo)

        # Círculo externo
        cv2.circle(output, (cx, cy), r, color, 2)

        # Círculo interno usado para analizar color
        cv2.circle(output, (cx, cy), radio_analisis, (255, 255, 0), 1)

        # Centro
        cv2.circle(output, (cx, cy), 4, (255, 0, 0), -1)

        # ID
        cv2.putText(
            output,
            str(res["id"]),
            (cx - 8, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        # Tipo detectado
        cv2.putText(
            output,
            tipo,
            (cx - 32, cy - r - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2
        )

        # Ocupación
        cv2.putText(
            output,
            f"{ocupacion:.2f}",
            (cx - 18, cy + r + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1
        )
        cv2.putText(
            output,
            f"Orientacion: {orientacion_blister} | Grilla: {filas_blister}x{columnas_blister}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            2
        )

    # ========================================================
    # TEXTO EN PANTALLA
    # ========================================================

    huecos = sum(1 for r in resultados if r["tipo"] == "Hueco")
    fucsias = sum(1 for r in resultados if r["tipo"] == "Fucsia")
    azules = sum(1 for r in resultados if r["tipo"] == "Azul")
    verdes = sum(1 for r in resultados if r["tipo"] == "Verde")

    cv2.putText(
        output,
        f"Crudos: {cantidad_cruda} | Filtrados: {cantidad_filtrada}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Fucsia={fucsias} Azul={azules} Verde={verdes} Huecos={huecos}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Zoom={ZOOM} Param2={PARAM2} R={MIN_RADIUS}-{MAX_RADIUS} UmbralColor={UMBRAL_COLOR}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Historial 10/10: {veces_cantidad_esperada}/{len(historial_cantidades)}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        2
    )

    # ========================================================
    # MÁSCARA TOTAL DE COLORES
    # ========================================================

    mask_fucsia = cv2.inRange(hsv, lower_fucsia, upper_fucsia)
    mask_azul = cv2.inRange(hsv, lower_azul, upper_azul)
    mask_verde = cv2.inRange(hsv, lower_verde, upper_verde)

    mask_color_total = cv2.bitwise_or(mask_fucsia, mask_azul)
    mask_color_total = cv2.bitwise_or(mask_color_total, mask_verde)

    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow("Deteccion circulos + color", output)
    cv2.imshow("Mascara colores Fucsia Azul Verde", mask_color_total)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()