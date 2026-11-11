# TP N° 2 - Procesamiento de Imágenes I (IA 4.4)
# Problema 1 (monedas y dados) con k-means
#
# monedas.jpg tiene que estar en la misma carpeta que este archivo. Solo usamos cv2, numpy y
# matplotlib. Para ejecutarlo: python TP2_Problema1_kmeans.py
#
# k-means se usa en dos decisiones:
#   - 1.A  círculos (monedas) vs cuadrados (dados): K = 2 sobre el factor de forma y el relleno del recuadro.
#   - 1.B  tipo de moneda: K = 3 sobre el área y la saturación media.
#
# Fuentes (todas del material de la cátedra; [archivo:líneas] o [PDF p.página]):
#   [PDI_U5_Color_kmeans.py:16-20]            cv2.kmeans(Z, K, None, criteria, intentos, KMEANS_RANDOM_CENTERS)
#   [PDI_U5_ejemplo_kmeans.py:16-26]          k-means con 2 clusters sobre puntos 2D y su gráfico
#   [U3 PDF p.28]                             GaussianBlur + Canny
#   [PDI_U5_Color_CIELAB.py:27]               cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#   [PDI_U3_Segmentacion.py:710]              cv2.threshold con THRESH_OTSU
#   [PI_U6_Morfologico_p2_Reconstruccion.py:24-33, 84-92]   imreconstruct e imfillhole
#   [PI_U6_Morfologico_p4_Grises.py:6-19]     imshow
#   [PI_U6_Morfologia_p1_Fundamentos.py:60-61] apertura y clausura con morphologyEx
#   [PDI_U3_Segmentacion.py:333]              connectedComponentsWithStats
#   [clasifico_celulas.py:38-45, 52-55, 28]   stats, contorno externo, factor de forma y RHO_TH = 0.8
#   [Celulas_tips.txt]                        factor de forma = área / perímetro²
#
# Lo que NO está en el material (lo armamos nosotros): el relleno del recuadro (área / (ancho x alto)),
# estandarizar las variables antes del k-means, y los valores numéricos (Canny 20 y 50, tamaños de
# kernels, área mínima, 50 intentos del k-means). Cada uno está explicado donde se usa.

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

# Reconstrucción morfológica y relleno de huecos [PI_U6_Morfologico_p2_Reconstruccion.py:24-33, 84-92]
def imreconstruct(marker, mask, kernel=None):
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)
        if (marker == expanded_intersection).all():
            break
        marker = expanded_intersection
    return expanded_intersection

def imfillhole(img):
    mask = np.zeros_like(img)
    mask = cv2.copyMakeBorder(mask[1:-1, 1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255))
    marker = cv2.bitwise_not(img, mask=mask)
    img_c = cv2.bitwise_not(img)
    img_r = imreconstruct(marker=marker, mask=img_c)
    return cv2.bitwise_not(img_r)

# k-means como en la práctica [PDI_U5_Color_kmeans.py:16-20]. Probamos 50 intentos en lugar de 10 porque
# con pocos objetos y 10 intentos a veces cae en una partición peor. Devuelve la etiqueta de cada fila y los centros.
def kmeans(F, K):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(np.float32(F), K, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
    return label.flatten(), center

# --- 1.A Segmentación -------------------------------------------------------------------------
# El fondo tiene un degradé, así que un umbral global no sirve [U3 PDF p.55-58]. Segmentamos con los
# bordes (Canny [U3 PDF p.28]) y con la saturación, que marca las monedas doradas aunque su borde no cierre.
img = cv2.cvtColor(cv2.imread(str(CARPETA / 'monedas.jpg')), cv2.COLOR_BGR2RGB)
ESC = 4                                                         # La foto es de 3871 x 2681: trabajamos con 1/4
small = cv2.GaussianBlur(img, (13, 13), 0)[::ESC, ::ESC]        # Pasabajos y después achico (numpy)
gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5), 20, 50)
s = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 1]             # Canal S (saturación)
_, s_bin = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bordes = np.zeros_like(edges)
bordes[np.logical_or(edges, s_bin)] = 255                       # Unión de bordes y zonas saturadas

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cerrado = cv2.dilate(cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, se), se)        # Cierro contornos cortados
mask = cv2.morphologyEx(imfillhole(cerrado), cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # Relleno y saco restos chicos
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
print(f"Imagen de trabajo: {small.shape[1]} x {small.shape[0]}  /  Componentes encontradas: {num_labels-1}")

plt.figure(figsize=(14, 8))
ax1 = plt.subplot(221); imshow(img, new_fig=False, color_img=True, title="Imagen original")
plt.subplot(222, sharex=ax1, sharey=ax1); imshow(bordes, new_fig=False, title="Bordes (Canny) + zonas saturadas")
plt.subplot(223, sharex=ax1, sharey=ax1); imshow(cerrado, new_fig=False, title="Contornos cerrados")
plt.subplot(224, sharex=ax1, sharey=ax1); imshow(mask, new_fig=False, title="Objetos rellenos")
plt.show(block=False)

# --- Descriptores de cada objeto ------------------------------------------------------------------
# Para cada objeto de al menos 1500 píxeles calculo:
#   rho    = 4 pi A / P²  [clasifico_celulas.py:52-55]. Círculo = 1, cuadrado = pi/4 = 0.785.
#   relleno = área / (ancho x alto del recuadro). Es la idea de usar stats de [clasifico_celulas.py:38-45].
#             Un círculo da pi/4 = 0.785 sin importar cómo esté; un cuadrado girado da menos.
#   sat    = saturación media dentro del objeto.
ids, rho, relleno, area, sat = [], [], [], [], []
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < 1500:
        continue
    cnt, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    A, P = cv2.contourArea(cnt[0]), cv2.arcLength(cnt[0], True)
    ids.append(i)
    rho.append(4*np.pi*A/P**2)
    relleno.append(stats[i, cv2.CC_STAT_AREA] / (stats[i, cv2.CC_STAT_WIDTH]*stats[i, cv2.CC_STAT_HEIGHT]))
    area.append(stats[i, cv2.CC_STAT_AREA])
    sat.append(s[labels == i].mean())
rho, relleno, area, sat = map(np.array, (rho, relleno, area, sat))

# --- Círculos vs cuadrados con k-means (K = 2) ------------------------------------------------------
# Con rho sola no alcanza: el dado que tiene sombra pegada da rho = 0.50 y queda lejos del otro dado
# (0.77), así que k-means deja a ese dado solo y mezcla al otro con las monedas. Con el relleno del
# recuadro los dos dados quedan lejos de las monedas. El cluster de rho más baja son los dados.
F = np.column_stack([rho, relleno])
lab, centros = kmeans(F, 2)
es_dado = lab == np.argmin(centros[:, 0])
for k, i in enumerate(ids):
    print(f"Objeto {i:2d} --> area: {area[k]:5d}  rho: {rho[k]:.3f}  relleno: {relleno[k]:.3f}  --> {'dado' if es_dado[k] else 'moneda'}")
print(f"Monedas: {(~es_dado).sum()}  /  Dados: {es_dado.sum()}")

# Gráfico como en la práctica [PDI_U5_ejemplo_kmeans.py:24-26]
plt.figure()
plt.scatter(F[~es_dado, 0], F[~es_dado, 1], label='monedas')
plt.scatter(F[es_dado, 0], F[es_dado, 1], c='r', label='dados')
plt.scatter(centros[:, 0], centros[:, 1], s=80, c='y', marker='s', label='centros')
plt.xlabel('rho (factor de forma)'), plt.ylabel('relleno del recuadro'), plt.legend(), plt.title("k-means K = 2")
plt.show(block=False)

salida = np.zeros_like(small)
for k, i in enumerate(ids):
    salida[labels == i, 0 if es_dado[k] else 1] = 255           # Dados en rojo, monedas en verde
imshow(salida, color_img=True, title="Dados en rojo, monedas en verde")

# --- 1.B Clasificación de monedas con k-means (K = 3) -----------------------------------------------
# Variables: área y saturación media. Las estandarizo (resto la media y divido por el desvío) para que
# ninguna pese más solo por tener números más grandes. Los clusters no tienen nombre: los llamo por el
# área de su centro (la más chica, la del medio y la más grande) siguiendo cómo son estas monedas:
# 10 centavos plateadas y chicas, 1 peso bimetálicas, 50 centavos doradas y grandes.
mon = np.flatnonzero(~es_dado)
G = np.column_stack([area[mon], sat[mon]]).astype(np.float32)
lab3, c3 = kmeans((G - G.mean(0)) / G.std(0), 3)
orden = np.argsort(c3[:, 0])                                    # Centros ordenados por área
nombres = {orden[0]: '10 centavos', orden[1]: '1 peso', orden[2]: '50 centavos'}
colores = {'10 centavos': 2, '1 peso': 1, '50 centavos': 0}     # Azul, verde, rojo
cuentas = {n: 0 for n in colores}
clasif = np.zeros_like(small)
for j, k in enumerate(mon):
    tipo = nombres[lab3[j]]
    cuentas[tipo] += 1
    clasif[labels == ids[k], colores[tipo]] = 255
    print(f"Moneda {ids[k]:2d} --> area: {area[k]:5d}  saturación media: {sat[k]:5.1f}  --> {tipo}")
print(cuentas, f"/ Total de monedas: {sum(cuentas.values())}")
imshow(clasif, color_img=True, title="10 centavos: azul  /  1 peso: verde  /  50 centavos: rojo")

# --- 1.C Puntos de la cara superior de cada dado -----------------------------------------------------
# Los puntos son chicos, así que recorto cada dado de la imagen original sin achicar. Fuera del dado
# pongo blanco para que el fondo no forme objetos. Binarizo con Otsu [PDI_U3_Segmentacion.py:710],
# complemento para que los puntos queden en blanco y etiqueto. Un punto de la cara de arriba es
# redondo: rho mayor a 0.8 [clasifico_celulas.py:28]. Las sombras y las caras laterales no lo son.
RHO_TH = 0.8
gris_full = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
valores = []
plt.figure(figsize=(10, 4))
for n, k in enumerate(np.flatnonzero(es_dado)):
    x, y, w, h = stats[ids[k], :4]
    y0, x0, y1, x1 = max(0, (y-3)*ESC), max(0, (x-3)*ESC), (y+h+3)*ESC, (x+w+3)*ESC
    dado = gris_full[y0:y1, x0:x1].copy()
    dentro = np.repeat(np.repeat(labels == ids[k], ESC, 0), ESC, 1)[y0:y1, x0:x1]
    dado[~dentro] = 255
    _, bn = cv2.threshold(dado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    puntos_bin = cv2.bitwise_not(bn)
    nl, lab_p, st_p, cen_p = cv2.connectedComponentsWithStats(puntos_bin)
    marcado = cv2.cvtColor(dado, cv2.COLOR_GRAY2RGB)
    puntos = 0
    for j in range(1, nl):
        if st_p[j, cv2.CC_STAT_AREA] < 150:
            continue
        c, _ = cv2.findContours((lab_p == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if 4*np.pi*cv2.contourArea(c[0])/cv2.arcLength(c[0], True)**2 > RHO_TH:
            puntos += 1
            cv2.circle(marcado, tuple(np.int32(cen_p[j])), 25, (0, 255, 0), -1)
    valores.append(puntos)
    print(f"Dado {n+1} --> cara superior: {puntos}")
    plt.subplot(2, 2, 2*n + 1); imshow(puntos_bin, new_fig=False, title=f"Dado {n+1}: puntos (Otsu + complemento)")
    plt.subplot(2, 2, 2*n + 2); imshow(marcado, new_fig=False, color_img=True, title=f"Puntos contados: {puntos}")
plt.show(block=False)
print(f"Cantidad de dados: {len(valores)}  /  Valores: {valores}")

plt.show()
