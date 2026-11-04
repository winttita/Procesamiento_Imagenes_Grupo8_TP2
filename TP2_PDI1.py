# # TP N° 2 - Procesamiento de Imágenes I (IA 4.4)
#
# Resolvemos el Problema 1 (monedas y dados) y el Problema 2 (patentes). Las imágenes
# (`monedas.jpg` e `img01.png` a `img12.png`) tienen que estar en la misma carpeta que este
# archivo. Solo usamos `cv2`, `numpy` y `matplotlib`, como en las prácticas, y `pathlib` para
# armar las rutas.
#
# En cada paso mostramos la imagen y escribimos por pantalla lo que se obtiene, como pide el
# aviso del enunciado. Las figuras quedan abiertas al final; el script termina al cerrarlas.
#
# Para ejecutarlo: python TP2_PDI1.py
#
# ### Referencias al material de la cátedra
#
# Cada decisión lleva su fuente entre corchetes. Las rutas son relativas a la carpeta de la
# materia:
#
# | Sigla | Archivo |
# |---|---|
# | U2-T | `U2 - Transformacion y Filtrado/PDI_TUIA_U2_Transformacion_y_Filtrado.pdf` |
# | U3-T | `U3 - Deteccion de Bordes y Segmentacion/PDI_TUIA_U3_Segmentacion.pdf` |
# | U3S | `U3 - Deteccion de Bordes y Segmentacion/Código/PDI_U3_Segmentacion.py` |
# | U5-T | `U5 - Color/PDI_TUIA_U5_Color.pdf` |
# | U5p3 | `U5 - Color/Codigo/PDI_U5_Color_p3.py` |
# | U6-T | `U6 - Analisis Morfologico/PDI_TUIA_U6_Morfologia.pdf` |
# | U6p1 | `U6 - Analisis Morfologico/Codigo/PI_U6_Morfologia_p1_Fundamentos.py` |
# | U6p2 | `U6 - Analisis Morfologico/Codigo/PI_U6_Morfologico_p2_Reconstruccion.py` |
# | U6p4 | `U6 - Analisis Morfologico/Codigo/PI_U6_Morfologico_p4_Grises.py` |
# | Cel | `U6 - Analisis Morfologico/Extras/clasifico_celulas.py` |
# | AB | `U6 - Analisis Morfologico/Extras/clasifico_Letras_AB.py` |
# | U7-T | `U7 - Descriptores y Clasificacion/PDI_TUIA_U7_Descriptores.pdf` |
# | U7M | `U7 - Descriptores y Clasificacion/Código/PDI_TUIA_U7_Descriptores_p1_Momentos.py` |
# | U1L | `U1 - Introduccion/Ejercicios/PDI_U1_p1_Letras.py` |
#
# `[U6p2:84-92]` quiere decir líneas 84 a 92 de ese archivo, y `[U3-T p.58]` la página 58 del
# apunte. También usamos los dos archivos de texto del enunciado: `Factor de Forma.txt` e
# `info_patentes.txt`.

# Primero importamos las librerías y copiamos tres funciones de la práctica de U6: `imshow` para
# mostrar imágenes [U6p1:6-19], `imreconstruct` para la reconstrucción morfológica [U6p2:24-33] e
# `imfillhole` para rellenar huecos [U6p2:84-92]. Las dejamos igual que en el archivo original.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CARPETA = Path(__file__).parent      # Las imagenes estan en la misma carpeta que este archivo

# Funcion para mostrar imagenes (copiada de la practica de U6)
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

# Reconstruccion morfologica: dilata el marcador sin salirse de la mascara hasta que no cambia mas
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection
    return expanded_intersection

# Rellenado de huecos usando la reconstruccion (no rellena huecos que tocan el borde)
def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh

# ## Problema 1: detección y clasificación de monedas y dados
#
# ### 1.A Segmentación
#
# El fondo tiene un degradé muy marcado, oscuro a la izquierda y claro a la derecha. Con un fondo
# así el umbralado global puede fallar porque la iluminación no es pareja [U3-T p.58]. La
# alternativa del apunte es un umbral local con top-hat [U3-T p.58, U6-T p.36], que sirve cuando
# todos los objetos son más claros que el fondo. Acá no pasa: a la izquierda las monedas son más
# claras que el fondo y a la derecha las doradas son más oscuras. Por eso segmentamos a partir de
# los bordes. Canny marca dónde cambia la intensidad, sin importar si el objeto es más claro o
# más oscuro que lo que lo rodea [U3-T pp.20-28].
#
# Cargamos la imagen y la pasamos a RGB [AB:6-7].

# --- Cargo imagen --------------------------------------------------------------------
img = cv2.imread(str(CARPETA / 'monedas.jpg'), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # OpenCV carga en BGR, la paso a RGB
H, W = img.shape[:2]
print(f"Imagen original: {W} x {H} píxeles")
imshow(img, color_img=True, title="Imagen original")

# La foto mide 3871 x 2681 píxeles, y la reconstrucción de `imfillhole` va dilatando de a un
# píxel por vez, así que en ese tamaño tarda mucho. Para que corra rápido la achicamos a un
# cuarto con `cv2.resize` [U7M:134]. Antes de achicarla la suavizamos con un filtro gaussiano
# grande (13 x 13): un pasabajos se usa como preprocesamiento para sacar los detalles chicos
# antes de segmentar objetos más grandes [U2-T p.29], y el relieve de las monedas son justamente
# detalles chicos. Las áreas que aparecen más abajo están medidas sobre la imagen achicada.
#
# Después aplicamos Canny. El apunte sugiere arrancar con un gaussiano de 3 x 3 y sigma 2 y
# ajustar desde ahí [U3-T p.30]; con esos valores varias monedas quedaban mal segmentadas, así
# que lo subimos a 5 x 5 con sigma 1.5. Los umbrales de Canny (20 y 50) también los ajustamos
# mirando el resultado, partiendo de los de la práctica [U3S:382-389].

# --- Suavizo y achico la imagen -----------------------------------------------------------
ESC = 4                                                                 # Factor de achicado
gray_full = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray_suave = cv2.GaussianBlur(gray_full, ksize=(13, 13), sigmaX=0)      # Pasabajos para sacar detalles chicos
gray = cv2.resize(gray_suave, (W//ESC, H//ESC))                         # Imagen en gris para segmentar
img_small = cv2.resize(img, (W//ESC, H//ESC))                           # Imagen color achicada (para el color)
print(f"Imagen de trabajo: {gray.shape[1]} x {gray.shape[0]} píxeles")

# --- Bordes con Canny ----------------------------------------------------------------------
blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=1.5)                 # Canny pide suavizar antes
edges = cv2.Canny(blur, threshold1=20, threshold2=50)
print(f"Píxeles de borde: {np.count_nonzero(edges)}")

plt.figure()
ax1 = plt.subplot(121); imshow(gray, new_fig=False, title="Gris, achicada")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(edges, new_fig=False, title="Bordes (Canny)")
plt.show(block=False)

# Canny solo no alcanza: en algunas monedas doradas el borde queda abierto y después no se puede
# rellenar. Lo resolvemos con color. Pasamos la imagen a HSV [U5-T p.13, U5p3:8-9] y miramos el
# canal S. La saturación es la "pureza" del color [U5-T p.12]: el fondo es gris y casi no tiene,
# mientras que el dorado de las monedas tiene bastante. Umbralamos S con Otsu [U6p4:90, U3-T
# p.57] y unimos esa máscara con los bordes, con la unión de conjuntos de la práctica de U6
# [U6p1:40-41]. Así las monedas doradas quedan marcadas aunque su contorno no cierre.

# --- Zonas saturadas (monedas doradas) ------------------------------------------------------
img_hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)    # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
h, s, v = cv2.split(img_hsv)                            # Me quedo con el canal S (saturacion)
umbral_s, s_bin = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(f"Umbral de Otsu sobre la saturación: {umbral_s:.0f}")

# --- Union: bordes de Canny + zonas saturadas ------------------------------------------------
bordes = np.zeros_like(edges)
bordes[np.logical_or(edges, s_bin)] = 255

plt.figure()
ax1 = plt.subplot(131); imshow(s, new_fig=False, title="Canal S (saturación)")
plt.subplot(132, sharex=ax1, sharey=ax1); imshow(s_bin, new_fig=False, title="S umbralado (Otsu)")
plt.subplot(133, sharex=ax1, sharey=ax1); imshow(bordes, new_fig=False, title="Bordes + zonas saturadas")
plt.show(block=False)

# Ahora cerramos los contornos que quedaron cortados con una clausura y una dilatación
# [U6p1:74-75, 112-113; U6-T pp.7-8 y 17]. La clausura rellena los agujeros más chicos que el
# elemento estructural [U6-T p.17]. Después rellenamos el interior de cada objeto con
# `imfillhole` [U6p2:84-92; U6-T p.29] y sacamos lo que quedó suelto con una apertura [U6p2:199,
# AB:24; U6-T p.16].
#
# Etiquetamos con componentes conectadas [U3S:333; U3-T pp.15-19] y las pintamos con un mapa de
# color, como en la práctica [Cel:18-25].

# --- Cierro contornos, relleno huecos y limpio ----------------------------------------------
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
bordes_cl = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, se)  # Clausura: une bordes cortados
bordes_cd = cv2.dilate(bordes_cl, se)                       # Dilatacion: termina de cerrar los contornos
filled = imfillhole(bordes_cd)                              # Relleno el interior de cada objeto

se_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask = cv2.morphologyEx(filled, cv2.MORPH_OPEN, se_op)      # Apertura: saco los restos chicos

plt.figure()
ax1 = plt.subplot(221); imshow(bordes_cl, new_fig=False, title="Clausura")
plt.subplot(222, sharex=ax1, sharey=ax1); imshow(bordes_cd, new_fig=False, title="Dilatación")
plt.subplot(223, sharex=ax1, sharey=ax1); imshow(filled, new_fig=False, title="Huecos rellenos")
plt.subplot(224, sharex=ax1, sharey=ax1); imshow(mask, new_fig=False, title="Apertura")
plt.show(block=False)

# --- Componentes conectadas ----------------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
print(f"Componentes encontradas: {num_labels-1}")

labels_color = np.uint8(255/(num_labels-1)*labels)                  # Llevo el rango de valores a [0 255] para diferenciar mejor los colores
im_color = cv2.applyColorMap(labels_color, cv2.COLORMAP_JET)
im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)                # El mapa de color que se aplica está en BGR --> convierto a RGB
imshow(im_color, color_img=True, title=f"Objetos etiquetados ({num_labels-1})")

# Para separar monedas de dados usamos el factor de forma de `Factor de Forma.txt` y de la
# práctica de células [Cel:52-56]. El apunte lo llama compacidad (o circularidad) y no depende
# del tamaño del objeto [U7-T p.21]. Lo multiplicamos por 4π para que un círculo dé 1: ρ =
# 4πA/P². Las monedas son casi círculos y los dados son cuadrados (un cuadrado perfecto da π/4 ≈
# 0.785).
#
# El umbral de la práctica es 0.8 [Cel:28]. Lo subimos a 0.82 porque el dado más redondo da
# 0.792, muy cerca de 0.8; la moneda menos redonda da 0.852, así que 0.82 queda en el medio.
# Antes de calcular ρ descartamos los objetos de menos de 1500 píxeles, igual que la práctica
# descarta las células chicas [Cel:44-46].

# --- Separo monedas y dados con el factor de forma -------------------------------------------
RHO_TH = 0.82       # Factor de forma (rho): arriba es moneda, abajo es dado
AREA_TH = 1500      # Umbral de area: los objetos mas chicos son ruido
monedas = []
dados = []
labeled_objetos = np.zeros_like(img_small)

for i in range(1, num_labels):
    # --- Remuevo objetos con area chica -------------------------------------
    if stats[i, cv2.CC_STAT_AREA] < AREA_TH:
        continue

    # --- Calculo rho ---------------------------------------------------------
    obj = (labels == i).astype(np.uint8)
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)
    rho = 4 * np.pi * area/(perimeter**2)

    # --- Clasifico -----------------------------------------------------------
    if rho > RHO_TH:
        monedas.append(i)
        labeled_objetos[obj == 1, 1] = 255      # Moneda --> verde
    else:
        dados.append(i)
        labeled_objetos[obj == 1, 0] = 255      # Dado --> rojo
    print(f"Objeto {i:2d} --> area: {stats[i, cv2.CC_STAT_AREA]:5d}  rho: {rho:.3f}  --> {'moneda' if rho > RHO_TH else 'dado'}")

print(f"Monedas: {len(monedas)}  /  Dados: {len(dados)}")
imshow(labeled_objetos, color_img=True, title="Segmentación: monedas en verde, dados en rojo")

# ### 1.B Clasificación de monedas
#
# Hay tres tipos: 10 centavos (plateadas y chicas), 1 peso (bimetálicas) y 50 centavos (doradas y
# grandes). Usamos dos descriptores. El primero es el área, que es la cantidad de píxeles del
# objeto [U7-T p.19], como en la práctica de células [Cel:29 y 45]. El área separa bien a las de
# 10 centavos: miden entre 4655 y 4964 píxeles, y el resto arranca en 6601. Elegimos 5800 como
# umbral.
#
# Entre 1 peso y 50 centavos la diferencia de área es de un 10% (hasta 6943 contra desde 7711), y
# preferimos no depender solo de eso. Usamos también la saturación media de cada moneda, del
# canal S que ya calculamos [U5-T p.12]. La de 50 centavos es dorada entera y tiene saturación
# alta (desde 107.4). La de 1 peso tiene el anillo plateado, que baja el promedio (hasta 83.7).
# El umbral queda en 95.
#
# La saturación sola no alcanza para las de 10 centavos: algunas llegan a 52.7 y la de 1 peso más
# baja tiene 55.4. Por eso primero miramos el área y después la saturación.

# --- Clasifico las monedas por area y por saturacion -------------------------------------------
AREA_10C = 5800     # Debajo de este area --> 10 centavos
S_TH = 95           # Saturacion media: debajo --> 1 peso (bimetalica) / arriba --> 50 centavos (dorada)

cuentas = {'10 centavos': 0, '1 peso': 0, '50 centavos': 0}
labeled_monedas = np.zeros_like(img_small)

for i in monedas:
    area = stats[i, cv2.CC_STAT_AREA]
    sat = s[labels == i].mean()             # Saturacion promedio dentro de la moneda

    if area < AREA_10C:
        tipo, canal = '10 centavos', 2      # Azul
    elif sat < S_TH:
        tipo, canal = '1 peso', 1           # Verde
    else:
        tipo, canal = '50 centavos', 0      # Rojo

    cuentas[tipo] += 1
    labeled_monedas[labels == i, canal] = 255
    print(f"Moneda {i:2d} --> area: {area:5d}  saturación media: {sat:5.1f}  --> {tipo}")

print(cuentas)
print(f"Total de monedas: {sum(cuentas.values())}")
imshow(labeled_monedas, color_img=True, title="10 centavos: azul  /  1 peso: verde  /  50 centavos: rojo")

# ### 1.C Dados
#
# Los puntos de los dados son chicos, así que trabajamos sobre la imagen original sin achicar.
# Recortamos cada dado con su bounding box [U3S:367-368], multiplicada por el factor de achicado
# y con unos píxeles de margen, y la recortamos como en la práctica de letras [U1L:254].
# Binarizamos con Otsu [U6p4:90, U3-T p.57] y complementamos [U6p1:36] para que los puntos queden
# en blanco.
#
# Etiquetamos y contamos solo las componentes que parecen un punto de la cara de arriba. Tienen
# que ser redondas, con el mismo factor de forma de antes [Cel:52-56], y tener un área entre 1000
# y 4000 píxeles. Los puntos de las caras laterales se ven achatados y miden menos de 1000
# píxeles. El fondo oscuro que entra en el recorte forma componentes mucho más grandes, y para
# eso está el límite de 4000. Marcamos cada punto contado con un círculo verde [U3S:366].

# --- Cuento los puntos de la cara superior de cada dado ----------------------------------------
AREA_PUNTO_MIN = 1000       # Los puntos de las caras laterales son mas chicos
AREA_PUNTO_MAX = 4000       # El fondo que entra en el recorte es mas grande
valores = []

plt.figure(figsize=(12, 8))
for k, i in enumerate(dados):
    # --- Recorto el dado en la imagen original --------------------------------
    x, y, w, h = stats[i, :4]
    dado = img[(y-3)*ESC:(y+h+3)*ESC, (x-3)*ESC:(x+w+3)*ESC]

    # --- Binarizo y dejo los puntos en blanco ---------------------------------
    dado_gray = cv2.cvtColor(dado, cv2.COLOR_RGB2GRAY)
    _, dado_bin = cv2.threshold(dado_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    puntos_bin = cv2.bitwise_not(dado_bin)
    n, lab_p, st_p, cen_p = cv2.connectedComponentsWithStats(puntos_bin)

    # --- Cuento los puntos redondos de tamano correcto ------------------------
    puntos = 0
    dado_out = dado.copy()
    for j in range(1, n):
        if not (AREA_PUNTO_MIN < st_p[j, cv2.CC_STAT_AREA] < AREA_PUNTO_MAX):
            continue
        obj = (lab_p == j).astype(np.uint8)
        ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rho = 4 * np.pi * cv2.contourArea(ext_contours[0])/(cv2.arcLength(ext_contours[0], True)**2)
        if rho > RHO_TH:
            puntos += 1
            cv2.circle(dado_out, tuple(np.int32(cen_p[j])), 25, color=(0, 255, 0), thickness=-1)   # Marco el punto contado

    valores.append(puntos)
    print(f"Dado {k+1} --> cara superior: {puntos}")

    plt.subplot(2, 3, 3*k + 1); imshow(dado, new_fig=False, color_img=True, title=f"Dado {k+1}")
    plt.subplot(2, 3, 3*k + 2); imshow(puntos_bin, new_fig=False, title="Puntos (Otsu + complemento)")
    plt.subplot(2, 3, 3*k + 3); imshow(dado_out, new_fig=False, color_img=True, title=f"Puntos contados: {puntos}")
plt.show(block=False)

print(f"Cantidad de dados: {len(dados)}  /  Valores: {valores}")

# ## Problema 2: detección de patentes
#
# ### 2.A Detección de la placa
#
# De `info_patentes.txt` sabemos tres cosas de los caracteres: la relación alto/ancho está entre
# 1.5 y 3, van en grupos de 3 y cada patente tiene dos grupos. Ubicar la placa a partir de eso es
# lo único del TP que no tiene un ejemplo directo en el material. Lo armamos con piezas que sí
# vimos:
#
# - Umbralado global con `cv2.threshold` [U3S:690; U3-T pp.56-57].
# - Componentes conectadas con `stats` (bounding box y área) [U3S:333; U3-T p.18].
# - El filtrado por tamaño y forma de la práctica de células [Cel:38-46]. La relación alto/ancho
#    es el alargamiento del apunte [U7-T p.22].
#
# La búsqueda de seis componentes alineadas es nuestra, hecha con `numpy`. Para cada imagen:
#
# 1. Para cada umbral T entre 20 y 240 (de a 5) binarizamos la imagen en gris y etiquetamos las
#    componentes.
# 2. Nos quedamos con las que parecen un carácter: relación alto/ancho entre 1.5 y 3, área de al
#    menos 20 píxeles y alto entre 10 y 45.
# 3. Buscamos seis alineadas sobre una recta (la placa puede estar inclinada), con alturas
#    parecidas (hasta 40% de diferencia) y con la separación 3 + 3: el hueco del medio tiene que ser
#    bastante más grande que los otros cuatro, y esos cuatro parecidos entre sí. Estas tolerancias
#    las ajustamos con las 12 fotos.
# 4. La iluminación cambia de una foto a otra, así que no hay un T que sirva para todas. Cada
#    placa aparece en un rango de valores de T, y nos quedamos con el del medio.
#
# En las 12 fotos los caracteres son más claros que el fondo de la placa. Con la imagen invertida
# (`cv2.bitwise_not` [U6p1:36]) no aparece ningún grupo, así que no la usamos.

# --- Candidatos a caracter para un umbral T --------------------------------------------------
def candidatos(gray, T):
    _, b = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)              # Umbralado global con T
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(b)
    c = []
    for x, y, w, h, area in stats[1:]:
        if 1.5 <= h/w <= 3.0 and area >= 20 and 10 <= h <= 45:        # Forma y tamano de un caracter
            c.append((int(x), int(y), int(w), int(h)))
    return c

# --- Seis caracteres separados en dos grupos de 3 ---------------------------------------------
def tres_y_tres(c6):
    xc = np.array([x + w/2 for x, y, w, h in c6])       # Centro horizontal de cada caracter
    d = np.diff(xc)                                     # Distancia entre caracteres vecinos
    internos = d[[0, 1, 3, 4]]                          # Las distancias dentro de cada grupo
    return bool(np.all(np.abs(internos - internos.mean()) <= 0.35*internos.mean()) and d[2] >= 1.4*internos.mean())

# --- Busco seis candidatos alineados, de altura parecida y con el patron 3 + 3 -----------------
def buscar_seis(c):
    for i in range(len(c)):
        xi, yi, wi, hi = c[i]
        for j in range(len(c)):
            xj, yj, wj, hj = c[j]
            dx = (xj + wj/2) - (xi + wi/2)
            if dx <= 0 or dx > 8*hi:                                    # j tiene que estar a la derecha y cerca
                continue
            pend = ((yj + hj/2) - (yi + hi/2)) / dx                     # Recta que pasa por los centros de i y j
            if abs(pend) > 0.4:                                         # Descarto rectas muy inclinadas
                continue
            fila = []
            for x, y, w, h in c:                                        # Candidatos que caen sobre esa recta
                y_recta = (yi + hi/2) + pend*((x + w/2) - (xi + wi/2))
                if abs(h - hi) <= 0.4*hi and abs((y + h/2) - y_recta) <= 0.3*hi:
                    fila.append((x, y, w, h))
            fila.sort()                                                 # Ordeno de izquierda a derecha
            for k in range(len(fila) - 5):
                if tres_y_tres(fila[k:k+6]):
                    return fila[k:k+6]
    return None

# --- Detecto la placa en las 12 imagenes -------------------------------------------------------
MARG = 4                    # Margen en pixeles alrededor de los caracteres
placas = {}
for i in range(1, 13):
    img_p = cv2.cvtColor(cv2.imread(str(CARPETA / f'img{i:02d}.png')), cv2.COLOR_BGR2RGB)
    gray_p = cv2.cvtColor(img_p, cv2.COLOR_RGB2GRAY)

    # Pruebo todos los umbrales y guardo en cuales aparece la placa
    hallazgos = []
    for T in range(20, 241, 5):
        c6 = buscar_seis(candidatos(gray_p, T))
        if c6:
            hallazgos.append((T, c6))
    if not hallazgos:
        print(f"img{i:02d}: no se encontró la placa")
        continue

    T, c6 = hallazgos[len(hallazgos)//2]                                # Me quedo con el T del medio del rango
    x0 = min(c[0] for c in c6) - MARG                                   # Recuadro que contiene a los 6 caracteres
    y0 = min(c[1] for c in c6) - MARG
    x1 = max(c[0] + c[2] for c in c6) + MARG
    y1 = max(c[1] + c[3] for c in c6) + MARG
    placas[i] = {"img": img_p, "gray": gray_p, "T": T, "caja": (x0, y0, x1, y1)}
    print(f"img{i:02d}: T = {T:3d} (la placa aparece con {len(hallazgos):2d} valores de T)  /  placa en x: {x0}-{x1}, y: {y0}-{y1}")

# Para ver las etapas en una imagen tomamos `img05.png`, la del enunciado. Mostramos la
# binarización con el T elegido, todos los candidatos a carácter en rojo, los seis que forman la
# placa en verde [U3S:368] y el recorte [U1L:254].

# --- Etapas del punto A sobre img05 -----------------------------------------------------------
p = placas[5]
T = p["T"]
_, bin_T = cv2.threshold(p["gray"], T, 255, cv2.THRESH_BINARY)
cand = candidatos(p["gray"], T)
c6 = buscar_seis(cand)
print(f"img05 con T = {T}: {len(cand)} candidatos a caracter, de los cuales {len(c6)} forman la placa")

img_cand = p["img"].copy()
for x, y, w, h in cand:                                                 # Candidatos en rojo
    cv2.rectangle(img_cand, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=1)
for x, y, w, h in c6:                                                   # Los 6 de la placa en verde
    cv2.rectangle(img_cand, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)

x0, y0, x1, y1 = p["caja"]
plt.figure(figsize=(14, 8))
ax1 = plt.subplot(221); imshow(p["gray"], new_fig=False, title="Imagen en gris")
plt.subplot(222, sharex=ax1, sharey=ax1); imshow(bin_T, new_fig=False, title=f"Binarizada con T = {T}")
plt.subplot(223, sharex=ax1, sharey=ax1); imshow(img_cand, new_fig=False, color_img=True, title="Candidatos (rojo) y placa (verde)")
plt.subplot(224); imshow(p["img"][y0:y1, x0:x1], new_fig=False, color_img=True, title="Placa recortada")
plt.show(block=False)

# --- Resultado del punto A en las 12 imagenes ---------------------------------------------------
plt.figure(figsize=(16, 9))
for i, p in placas.items():
    x0, y0, x1, y1 = p["caja"]
    img_rect = p["img"].copy()
    cv2.rectangle(img_rect, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)    # Placa detectada
    plt.subplot(3, 4, i); imshow(img_rect, new_fig=False, color_img=True, title=f"img{i:02d}")
plt.show(block=False)

plt.figure(figsize=(16, 7))
for i, p in placas.items():
    x0, y0, x1, y1 = p["caja"]
    plt.subplot(3, 4, i); imshow(p["img"][y0:y1, x0:x1], new_fig=False, color_img=True, title=f"img{i:02d}")
plt.show(block=False)

# ### 2.B Segmentación de los caracteres
#
# Recortamos la placa (el recuadro de los seis caracteres con 4 píxeles de margen), la
# binarizamos con el T que encontramos en 2.A y etiquetamos solo dentro del recorte. Aplicamos el
# mismo filtro de forma y tamaño, esperamos exactamente seis componentes con la separación 3 + 3
# y las ordenamos de izquierda a derecha, como la práctica ordena contornos con `sorted`
# [U3S:478]. Mostramos los caracteres de cada placa en una grilla de subplots, como en la
# práctica de letras [U1L:251-256].
#
# No usamos Otsu sobre el recorte [U3S:710] porque los caracteres son muy finos y salen partidos
# o pegados: con Otsu se separan bien los seis caracteres en solo 4 de las 12 imágenes.

# --- Segmento los caracteres dentro de cada placa ------------------------------------------------
caracteres = {}
plt.figure(figsize=(16, 7))
for i, p in placas.items():
    x0, y0, x1, y1 = p["caja"]
    placa_gray = p["gray"][y0:y1, x0:x1]                    # Recorte de la placa
    placa_img = p["img"][y0:y1, x0:x1].copy()

    c = sorted(candidatos(placa_gray, p["T"]))              # Caracteres ordenados de izquierda a derecha
    ok = len(c) == 6 and tres_y_tres(c)
    caracteres[i] = c
    print(f"img{i:02d}: {len(c)} caracteres, separados 3 + 3: {ok}")
    print(f"        (x, y, ancho, alto) en la placa: {c}")

    for x, y, w, h in c:                                    # Recuadro cada caracter
        cv2.rectangle(placa_img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)
    plt.subplot(3, 4, i); imshow(placa_img, new_fig=False, color_img=True, title=f"img{i:02d}: {len(c)} caracteres")
plt.show(block=False)

# ## Comentarios finales
#
# Problema 1: el algoritmo encuentra 17 monedas (9 de 10 centavos, 5 de 1 peso y 3 de 50
# centavos) y 2 dados, uno con 5 puntos y otro con 3. Los umbrales los elegimos mirando esta
# imagen, porque es la única que hay. El más ajustado es el factor de forma que separa dados de
# monedas: el dado más redondo da 0.792 y la moneda menos redonda 0.852. Sumar la máscara de
# saturación hizo que la segmentación dependa menos de los parámetros de Canny y del suavizado.
#
# Problema 2: en las 12 imágenes se detecta la placa y se separan seis caracteres en dos grupos
# de 3. img03 e img12 son las más frágiles, porque la placa aparece con un solo valor de T.

# --- Dejo abiertas todas las figuras hasta que se cierren -----------------------------------
plt.show()
