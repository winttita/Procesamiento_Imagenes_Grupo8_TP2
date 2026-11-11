# TP N° 2 - Procesamiento de Imágenes I (IA 4.4)
# Problema 2: detección de patentes y segmentación de sus caracteres
#
# Las imágenes img01.png ... img12.png tienen que estar en la misma carpeta que este archivo.
# Solo usamos cv2, numpy y matplotlib, como en las prácticas. Para ejecutarlo: python TP2_Problema2.py
#
# Fuentes (todas del material de la cátedra; [archivo:líneas] o [PDF p.página]):
#   [U3 PDF p.55]                       Umbralado global
#   [PDI_U3_Segmentacion.py:690]        cv2.threshold(img, thresh, maxval, THRESH_BINARY)
#   [U3 PDF p.18]                       connectedComponentsWithStats: num_labels, labels, stats, centroids
#   [PDI_U3_Segmentacion.py:333]        uso de cv2.connectedComponentsWithStats
#   [clasifico_celulas.py:44-46]        descartar objetos por área usando stats
#   [clasifico_Letras_AB.py:6-7,15]     cv2.imread + cvtColor BGR->RGB y RGB->GRAY
#   [PI_U6_Morfologico_p4_Grises.py:6-19] función imshow
#   [info_patentes.txt]                 relación alto/ancho 1.5-3.0 y dos grupos de 3 caracteres
#
# Lo que NO está en el material (lo armamos nosotros): probar varios umbrales T y agrupar los
# componentes por altura y renglón. Está explicado donde se usa.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CARPETA = Path(__file__).parent

# Función para mostrar imágenes (copiada de la práctica de U6) [PI_U6_Morfologico_p4_Grises.py:6-19]
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

# --- Candidatos a carácter para un umbral T ---------------------------------------------------
# Umbralado global [PDI_U3_Segmentacion.py:690], componentes conectadas [PDI_U3_Segmentacion.py:333]
# y filtro por área con stats [clasifico_celulas.py:44-46]. La relación alto/ancho (1.5 a 3.0) sale de
# info_patentes.txt. El área mínima (20) y el alto entre 10 y 45 píxeles los ajustamos con las 12 fotos.
def candidatos(gray, T):
    _, b = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(b)
    c = [tuple(int(v) for v in s[:4]) for s in stats[1:]
         if 1.5 <= s[3]/s[2] <= 3.0 and s[4] >= 20 and 10 <= s[3] <= 45]
    return b, c

# --- Los 6 caracteres de la patente -------------------------------------------------------------
# Tomamos el grupo más grande de candidatos con altura parecida (hasta 40 % de diferencia), en el
# mismo renglón y cercanos entre sí. Si son exactamente 6 y la separación del medio es mucho mayor
# que las otras, es el 3 + 3 de info_patentes.txt. Esta regla es nuestra, no está en el material.
def seis(c):
    mejor = []
    for a in c:
        g = [b for b in c if abs(b[3] - a[3]) <= 0.4*a[3]
             and abs((b[1] + b[3]/2) - (a[1] + a[3]/2)) <= 1.0*a[3]
             and abs((b[0] + b[2]/2) - (a[0] + a[2]/2)) <= 6*a[3]]
        if len(g) > len(mejor):
            mejor = g
    mejor = sorted(mejor)                                       # De izquierda a derecha
    if len(mejor) != 6:
        return None
    d = np.diff([x + w/2 for x, y, w, h in mejor])              # Distancia entre caracteres vecinos
    return mejor if d[2] > 1.4*d[[0, 1, 3, 4]].mean() else None

# --- 2.A Detección de la patente --------------------------------------------------------------
# La iluminación cambia de una foto a otra, así que no hay un T que sirva para todas [U3 PDF p.55-58].
# Probamos T = 20, 25, ..., 240 y nos quedamos con el T del medio de los que dan los 6 caracteres.
MARGEN = 4          # Píxeles de margen alrededor de los caracteres
placas = {}
for i in range(1, 13):
    img = cv2.cvtColor(cv2.imread(str(CARPETA / f'img{i:02d}.png')), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    Ts = [T for T in range(20, 241, 5) if seis(candidatos(gray, T)[1])]
    if not Ts:
        print(f"img{i:02d}: no se encontró la patente")
        continue
    T = Ts[len(Ts)//2]
    c6 = seis(candidatos(gray, T)[1])
    x0 = min(x for x, y, w, h in c6) - MARGEN
    y0 = min(y for x, y, w, h in c6) - MARGEN
    x1 = max(x + w for x, y, w, h in c6) + MARGEN
    y1 = max(y + h for x, y, w, h in c6) + MARGEN
    placas[i] = dict(img=img, gray=gray, T=T, caja=(x0, y0, x1, y1))
    print(f"img{i:02d}: seis caracteres con {len(Ts):2d} valores de T (de {Ts[0]} a {Ts[-1]}). "
          f"Uso T = {T}. Patente en x: {x0}-{x1}, y: {y0}-{y1}")

# Etapas sobre img05 (la del enunciado): gris, binarizada, candidatos (rojo) con los 6 elegidos (verde) y recorte
p = placas[5]
b, cand = candidatos(p["gray"], p["T"])
img_cand = p["img"].copy()
for x, y, w, h in cand:
    cv2.rectangle(img_cand, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=1)
for x, y, w, h in seis(cand):
    cv2.rectangle(img_cand, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)
x0, y0, x1, y1 = p["caja"]
print(f"img05 con T = {p['T']}: {len(cand)} candidatos a carácter, 6 forman la patente")
plt.figure(figsize=(14, 8))
ax1 = plt.subplot(221); imshow(p["gray"], new_fig=False, title="Imagen en gris")
plt.subplot(222, sharex=ax1, sharey=ax1); imshow(b, new_fig=False, title=f"Binarizada con T = {p['T']}")
plt.subplot(223, sharex=ax1, sharey=ax1); imshow(img_cand, new_fig=False, color_img=True, title="Candidatos (rojo) y patente (verde)")
plt.subplot(224); imshow(p["img"][y0:y1, x0:x1], new_fig=False, color_img=True, title="Patente recortada")
plt.show(block=False)

# Resultado en las 12 imágenes: patente marcada y patente recortada
plt.figure(figsize=(16, 9))
for i, p in placas.items():
    x0, y0, x1, y1 = p["caja"]
    marcada = p["img"].copy()
    cv2.rectangle(marcada, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
    plt.subplot(3, 4, i); imshow(marcada, new_fig=False, color_img=True, title=f"img{i:02d}")
plt.show(block=False)

plt.figure(figsize=(16, 7))
for i, p in placas.items():
    x0, y0, x1, y1 = p["caja"]
    plt.subplot(3, 4, i); imshow(p["img"][y0:y1, x0:x1], new_fig=False, color_img=True, title=f"img{i:02d}")
plt.show(block=False)

# --- 2.B Segmentación de los caracteres -------------------------------------------------------
# Trabajamos solo sobre el recorte de la patente: lo binarizamos con el mismo T, etiquetamos y nos
# quedamos con los 6 componentes. Dejamos en la máscara solo esos caracteres.
caracteres = {}
plt.figure(figsize=(16, 7))
for i, p in placas.items():
    x0, y0, x1, y1 = p["caja"]
    b, c = candidatos(p["gray"][y0:y1, x0:x1], p["T"])
    c6 = seis(c)
    if c6 is None:
        print(f"img{i:02d}: no se pudieron separar los 6 caracteres")
        continue
    mascara = np.zeros_like(b)
    for x, y, w, h in c6:
        mascara[y:y+h, x:x+w] = b[y:y+h, x:x+w]
    caracteres[i] = (mascara, c6)
    print(f"img{i:02d}: 6 caracteres (x, y, ancho, alto) en la patente: {c6}")
    plt.subplot(3, 4, i); imshow(mascara, new_fig=False, title=f"img{i:02d}")
plt.show(block=False)

# Los 6 caracteres de img05, uno por uno
mascara, c6 = caracteres[5]
plt.figure(figsize=(8, 3))
for k, (x, y, w, h) in enumerate(c6):
    plt.subplot(1, 6, k+1); imshow(mascara[y:y+h, x:x+w], new_fig=False, title=f"{k+1}")
plt.show(block=False)

plt.show()
