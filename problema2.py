import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CARPETA = Path(__file__).parent
CAPTURAS = CARPETA / 'capturas' / 'problema2'
CAPTURAS.mkdir(parents=True, exist_ok=True)

# --- Parámetros (ajustados experimentando sobre las 12 imágenes) ------------------
UMBRALES = range(20, 241, 5)    # Umbrales T que se prueban en cada imagen
AREA_MIN, AREA_MAX = 15, 300    # Área (px) de un caracter. Medido: entre 18 y 161 px
RA_MIN, RA_MAX = 1.5, 3.0       # Relación de aspecto alto/ancho de un caracter (info_patentes.txt)
DIF_ALTO = 0.25                 # Cercanía: diferencia de alto entre caracteres (en altos de caracter)
DIF_Y = 0.5                     # Cercanía: diferencia vertical entre caracteres (en altos de caracter)
DIF_X = 6                       # Cercanía: distancia horizontal máxima entre caracteres (en altos de caracter)
MARGEN_X, MARGEN_Y = 0.8, 0.6   # Margen de la patente alrededor de los caracteres (en altos de caracter)


# Función para mostrar imágenes (la de la cátedra)
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


def candidatos(gray, T):
    # Umbralado + componentes conectadas + filtrado por área y por relación de aspecto.
    # Devuelve la imagen binaria y los stats (x, y, ancho, alto, área) de todos los objetos,
    # de los que pasan el filtro de área y de los que además pasan el de relación de aspecto.
    _, img_th = cv2.threshold(gray, thresh=T, maxval=255, type=cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_th, connectivity=8, ltype=cv2.CV_32S)
    stats = stats[1:]                                                           # Se descarta el fondo (etiqueta 0)
    st_area = stats[(stats[:, 4] >= AREA_MIN) & (stats[:, 4] <= AREA_MAX)]      # Filtrado por área
    ra = st_area[:, 3] / st_area[:, 2]
    st_ra = st_area[(ra >= RA_MIN) & (ra <= RA_MAX)]                            # Filtrado por relación de aspecto
    return img_th, stats, st_area, st_ra


def agrupar(st):
    # Filtrado por cercanía: para cada candidato busco los que tienen alto parecido, están en el
    # mismo renglón y a poca distancia horizontal. Devuelvo el grupo más numeroso, de izquierda a derecha.
    mejor = []
    for a in st:
        g = [b for b in st if abs(b[3] - a[3]) <= DIF_ALTO*a[3] and abs(b[1] - a[1]) <= DIF_Y*a[3] and abs(b[0] - a[0]) <= DIF_X*a[3]]
        if len(g) > len(mejor):
            mejor = g
    return sorted(mejor, key=lambda s: s[0])


def rectangulos(img, st, color):
    for x, y, w, h in [s[:4] for s in st]:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color=color, thickness=1)
    return img


resumen = []
for i in range(1, 13):
    nombre = f'img{i:02d}'
    print(f'\n=============== {nombre}.png ===============')
    img = cv2.imread(str(CARPETA / f'{nombre}.png'))
    if img is None:
        raise FileNotFoundError(f'No se pudo cargar {nombre}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    # --- Punto A - Detección de la patente ---------------------------------------
    # Primero los caracteres: pruebo varios umbrales y me quedo con los que dan el grupo más grande (hasta 6 caracteres).
    grupos = {T: agrupar(candidatos(gray, T)[3]) for T in UMBRALES}
    n_max = max(len(g) for g in grupos.values() if len(g) <= 6)
    Ts = [T for T, g in grupos.items() if len(g) == n_max]
    T = Ts[len(Ts)//2]                                  # Umbral del medio de los que funcionan
    img_th, stats, st_area, st_ra = candidatos(gray, T)
    grupo = grupos[T]
    print(f'Umbrales con {n_max} caracteres agrupados: {len(Ts)} (de T={Ts[0]} a T={Ts[-1]}). Uso T={T}')
    print(f'Componentes conectadas: {len(stats)}  -->  filtro por área: {len(st_area)}  -->  filtro por relación de aspecto: {len(st_ra)}  -->  filtro por cercanía: {len(grupo)}')

    # Después la patente: caja que encierra a los caracteres + un margen proporcional al alto de los caracteres.
    g = np.array(grupo)
    alto = g[:, 3].mean()
    x0 = max(int(g[:, 0].min() - MARGEN_X*alto), 0)
    y0 = max(int(g[:, 1].min() - MARGEN_Y*alto), 0)
    x1 = min(int((g[:, 0] + g[:, 2]).max() + MARGEN_X*alto), W)
    y1 = min(int((g[:, 1] + g[:, 3]).max() + MARGEN_Y*alto), H)
    patente = img[y0:y1, x0:x1]
    print(f'Patente detectada en x: {x0}-{x1}, y: {y0}-{y1}  (ancho/alto = {(x1-x0)/(y1-y0):.2f})')

    # --- Punto B - Segmentación de los caracteres de la patente -------------------
    # Mismo procesamiento, pero solo sobre el recorte de la patente detectada en el punto A.
    pat_th, _, _, pat_ra = candidatos(gray[y0:y1, x0:x1], T)
    caracteres = agrupar(pat_ra)
    ok = len(caracteres) == 6
    print(f'Caracteres segmentados: {len(caracteres)} de 6  -->  {"OK" if ok else "FALLA"}')
    for k, (x, y, w, h, a) in enumerate(caracteres, start=1):
        print(f'  Caracter {k}: x={x:3d}, y={y:3d}, ancho={w:2d}, alto={h:2d}, área={a:3d}, alto/ancho={h/w:.2f}')
    resumen.append((nombre, T, len(caracteres), patente, rectangulos(patente.copy(), caracteres, (0, 255, 0))))

    # --- Etapas del procesamiento ------------------------------------------------
    img_cand = rectangulos(rectangulos(img.copy(), st_area, (255, 0, 0)), st_ra, (255, 255, 0))
    img_pat = rectangulos(img.copy(), grupo, (0, 255, 0))
    cv2.rectangle(img_pat, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)
    plt.figure(figsize=(16, 7))
    plt.subplot(231); imshow(img, new_fig=False, color_img=True, title=f"{nombre} - Imagen Original")
    plt.subplot(232); imshow(img_th, new_fig=False, title=f"Umbralado (T = {T})")
    plt.subplot(233); imshow(img_cand, new_fig=False, color_img=True, title=f"Filtro por área (rojo, {len(st_area)}) y por aspecto (amarillo, {len(st_ra)})")
    plt.subplot(234); imshow(img_pat, new_fig=False, color_img=True, title=f"Filtro por cercanía (verde, {len(grupo)}) y patente (rojo)")
    plt.subplot(235); imshow(pat_th, new_fig=False, title="Patente segmentada y umbralada")
    plt.subplot(236); imshow(resumen[-1][4], new_fig=False, color_img=True, title=f"Caracteres segmentados: {len(caracteres)}")
    plt.savefig(CAPTURAS / f'{nombre}_etapas.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)

# --- Resultado en las 12 imágenes ---------------------------------------------------
plt.figure(figsize=(16, 7))
for k, (nombre, T, n, patente, pat_car) in enumerate(resumen, start=1):
    plt.subplot(3, 4, k); imshow(pat_car, new_fig=False, color_img=True, title=f"{nombre} (T = {T}): {n} caracteres")
plt.savefig(CAPTURAS / 'resumen_patentes.png', dpi=150, bbox_inches='tight')
plt.show(block=False)

print('\n=============== Resumen ===============')
fallas = [nombre for nombre, T, n, _, _ in resumen if n != 6]
print(f'Patentes detectadas: {len(resumen)} de 12')
print(f'Imágenes con los 6 caracteres segmentados: {12 - len(fallas)} de 12')
print(f'Imágenes que fallan: {fallas if fallas else "ninguna"}')
print(f'Capturas guardadas en: {CAPTURAS}')
plt.show()
