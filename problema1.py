import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CARPETA = Path(__file__).parent
CAPTURAS = CARPETA / 'capturas' / 'problema1'
CAPTURAS.mkdir(parents=True, exist_ok=True)

# --- Parámetros (ajustados experimentando sobre monedas.jpg) --------------------
K_BLUR = 15                 # Tamaño del kernel del filtro pasabajos (Gaussiano)
CANNY_TH1, CANNY_TH2 = 15, 45   # Umbrales de Canny
K_DIL = 35                  # Elemento estructural de la dilatación / erosión
K_OPEN = 41                 # Elemento estructural de la apertura final
FF_TH = 15.5                # Umbral de 1/Fp: menor --> circular (círculo ideal = 12.57, cuadrado ideal = 16)
AREA_PUNTO = 500            # Área mínima (px) de un punto de dado
TIPOS = ['10 centavos', '1 peso', '50 centavos']    # Monedas ordenadas de menor a mayor tamaño
COLORES = {'10 centavos': (0, 255, 255), '1 peso': (0, 255, 0), '50 centavos': (255, 0, 255), 'Dado': (255, 0, 0)}   # RGB


# Función para mostrar imágenes (la de la cátedra) + guardado de la captura
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False, archivo=None):
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
    if archivo:
        plt.savefig(CAPTURAS / archivo, dpi=200, bbox_inches='tight')
    if new_fig:
        plt.show(block=blocking)


# Rellenado de huecos con cv2.floodFill() (PI_U6_Morfologico_p2_Reconstruccion.py, versión 2)
def imfillhole_v2(img):
    img_flood_fill = img.copy().astype("uint8")             # Genero la imagen de salida
    h, w = img.shape[:2]                                    # Genero una máscara necesaria para cv2.floodFill()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_flood_fill, mask, (0,0), 255)         # Relleno o inundo la imagen.
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)    # Tomo el complemento de la imagen inundada --> Obtengo SOLO los huecos rellenos.
    img_fh = img | img_flood_fill_inv                       # La salida es un OR entre la imagen original y los huecos rellenos.
    return img_fh


def factor_forma(obj):
    # obj : Imagen binaria (uint8) con un único objeto.
    # Devuelve 1/Fp = Perímetro^2 / Área ("Factor de Forma.txt"). Círculo: 12.57. Cuadrado: 16.
    cnt, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.arcLength(cnt[0], True)**2 / cv2.contourArea(cnt[0])


# --------------------------------------------------------------------------------
# --- Punto A - Segmentación de monedas y dados ------------------------------------
# --------------------------------------------------------------------------------
print('=============== Punto A: segmentación ===============')

# --- Cargo imagen ---------------------------------------------------------------
img = cv2.imread(str(CARPETA / 'monedas.jpg'))
if img is None:
    raise FileNotFoundError('No se pudo cargar monedas.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(f'Imagen: {img.shape[1]}x{img.shape[0]} px')
imshow(img, color_img=True, title="Imagen Original", archivo='01_original.png')

# --- Filtro pasabajos + Canny -----------------------------------------------------
f_blur = cv2.GaussianBlur(gray, ksize=(K_BLUR, K_BLUR), sigmaX=0)
gcan = cv2.Canny(f_blur, threshold1=CANNY_TH1, threshold2=CANNY_TH2)
print(f'Canny (U1={CANNY_TH1} | U2={CANNY_TH2}): {np.count_nonzero(gcan)} píxeles de borde')
imshow(f_blur, title=f"Escala de grises + Gaussian Blur {K_BLUR}x{K_BLUR}", archivo='02_gris_blur.png')
imshow(gcan, title=f"Canny - U1={CANNY_TH1} | U2={CANNY_TH2}", archivo='03_canny.png')

# --- Morfología: cierro los bordes, relleno huecos y vuelvo al tamaño original -----
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_DIL, K_DIL))
gcan_d = cv2.dilate(gcan, se)                       # Dilatación: cierra los bordes que Canny deja abiertos
gcan_d_fh = imfillhole_v2(gcan_d)                   # Relleno huecos (con los bordes ya cerrados)
gcan_d_fh_e = cv2.erode(gcan_d_fh, se)              # Erosión con el mismo kernel: recupera el tamaño y separa objetos cercanos
img_seg = cv2.morphologyEx(gcan_d_fh_e, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_OPEN, K_OPEN)))  # Apertura: elimina restos chicos y suaviza bordes
imshow(gcan_d, title=f"Dilatación {K_DIL}x{K_DIL}", archivo='04_dilatacion.png')
imshow(gcan_d_fh, title="Rellenado de huecos", archivo='05_relleno.png')
imshow(gcan_d_fh_e, title=f"Erosión {K_DIL}x{K_DIL}", archivo='06_erosion.png')
imshow(img_seg, title=f"Apertura {K_OPEN}x{K_OPEN}", archivo='07_apertura.png')

# --- Componentes conectadas -------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_seg, connectivity=8, ltype=cv2.CV_32S)
n_antes = cv2.connectedComponentsWithStats(gcan_d_fh_e, connectivity=8, ltype=cv2.CV_32S)[0] - 1
print(f'Objetos antes de la apertura: {n_antes} / después de la apertura: {num_labels-1}')

labels_color = np.uint8(255/(num_labels-1)*labels)                  # Llevo el rango de valores a [0 255] para diferenciar mejor los colores
im_color = cv2.applyColorMap(labels_color, cv2.COLORMAP_JET)
im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)                # El mapa de color que se aplica está en BGR --> convierto a RGB
imshow(im_color, color_img=True, title=f"Componentes conectadas ({num_labels-1} objetos)", archivo='08_componentes.png')

# --- Factor de forma: monedas (circulares) vs. dados -------------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
monedas, dados = [], []
for i in range(1, num_labels):
    obj = (labels == i).astype(np.uint8)                # Genero una imagen que contenga solo el objeto "i"
    ff = factor_forma(obj)
    flag_circular = ff < FF_TH
    sat = hsv[:, :, 1][obj == 1].mean()                 # Saturación media (solo para informar)
    print(f"Objeto {i:2d} --> Área: {stats[i, cv2.CC_STAT_AREA]:6d}  /  1/Fp: {ff:5.2f}  /  Saturación media: {sat:5.1f}  /  {'Moneda' if flag_circular else 'Dado'}")
    (monedas if flag_circular else dados).append(i)
print(f'Segmentación: {len(monedas)} monedas y {len(dados)} dados')

img_ab = np.zeros_like(img)
img_ab[np.isin(labels, monedas), 1] = 255               # Monedas en verde
img_ab[np.isin(labels, dados), 0] = 255                 # Dados en rojo
imshow(img_ab, color_img=True, title=f"Factor de forma: monedas (verde, {len(monedas)}) y dados (rojo, {len(dados)})", archivo='09_factor_forma.png')

# --------------------------------------------------------------------------------
# --- Punto B - Clasificación y conteo de monedas ----------------------------------
# --------------------------------------------------------------------------------
print('\n=============== Punto B: monedas ===============')

# --- K-means sobre el área (PDI_U5_ejemplo_kmeans.py) --------------------------------
Z = np.float32(stats[monedas, cv2.CC_STAT_AREA]).reshape(-1, 1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # criteria --> ( type, max_iter, epsilon )
cv2.setRNGSeed(0)                                                           # Para que el resultado sea repetible
ret, label, center = cv2.kmeans(Z, len(TIPOS), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
orden = np.argsort(center.flatten())                    # Clusters ordenados de menor a mayor área
tipo_cluster = {int(c): TIPOS[k] for k, c in enumerate(orden)}
clase = {i: tipo_cluster[int(l)] for i, l in zip(monedas, label.flatten())}     # Objeto --> tipo de moneda
clase.update({i: 'Dado' for i in dados})

plt.figure()
for c in orden:
    A = Z[label.flatten() == c]
    plt.scatter(np.flatnonzero(label.flatten() == c), A[:, 0], label=f'{tipo_cluster[int(c)]}: {len(A)} (centro {center[c, 0]:.0f} px)')
plt.xlabel('Moneda'), plt.ylabel('Área [px]'), plt.title('K-means sobre el área de las monedas'), plt.legend()
plt.savefig(CAPTURAS / '10_kmeans_areas.png', dpi=200, bbox_inches='tight')
plt.show(block=False)

for c in orden:
    A = Z[label.flatten() == c]
    print(f'{tipo_cluster[int(c)]:12s}: {len(A)} monedas  (área entre {A.min():.0f} y {A.max():.0f} px, centro {center[c, 0]:.0f})')
print(f'Total de monedas: {len(monedas)}')

# --------------------------------------------------------------------------------
# --- Punto C - Valor de la cara superior de cada dado -----------------------------
# --------------------------------------------------------------------------------
print('\n=============== Punto C: dados ===============')
valores = {}
for n, i in enumerate(dados, start=1):
    x, y, w, h = stats[i, :4]
    dado = gray[y:y+h, x:x+w]                           # Recorte del dado
    mask = labels[y:y+h, x:x+w] == i
    T_otsu, _ = cv2.threshold(dado[mask].reshape(-1, 1), thresh=127, maxval=255, type=cv2.THRESH_OTSU)   # Umbral de Otsu solo con los píxeles del dado
    puntos = np.uint8((dado < T_otsu) & mask)           # Umbralado: los puntos son oscuros sobre el dado blanco

    # Un punto de la cara superior se ve circular. Los de las caras laterales se ven como elipses --> los descarto con el factor de forma.
    num_p, labels_p, stats_p, _ = cv2.connectedComponentsWithStats(puntos, connectivity=8, ltype=cv2.CV_32S)
    img_p = cv2.cvtColor(dado, cv2.COLOR_GRAY2RGB)
    valor = 0
    for k in range(1, num_p):
        if stats_p[k, cv2.CC_STAT_AREA] < AREA_PUNTO:    # Remuevo objetos con área chica
            continue
        ff = factor_forma((labels_p == k).astype(np.uint8))
        flag_circular = ff < FF_TH
        valor += flag_circular
        img_p[labels_p == k] = (0, 255, 0) if flag_circular else (255, 0, 0)
        print(f"Dado {n} - Mancha oscura {k:2d} --> Área: {stats_p[k, cv2.CC_STAT_AREA]:5d}  /  1/Fp: {ff:6.2f}  /  {'Punto de la cara superior' if flag_circular else 'Descartada'}")
    valores[i] = int(valor)
    print(f'Dado {n} (objeto {i}): umbral de Otsu = {T_otsu:.0f}, valor de la cara superior = {valor}')

    plt.figure(figsize=(12, 4))
    plt.subplot(131); imshow(dado, new_fig=False, title=f"Dado {n}")
    plt.subplot(132); imshow(puntos, new_fig=False, title=f"Umbralado (Otsu = {T_otsu:.0f})")
    plt.subplot(133); imshow(img_p, new_fig=False, color_img=True, title=f"Puntos circulares (verde): {valor}")
    plt.savefig(CAPTURAS / f'11_dado_{n}.png', dpi=200, bbox_inches='tight')
    plt.show(block=False)
print(f'Total de dados: {len(dados)}  /  Valores: {list(valores.values())}')

# --------------------------------------------------------------------------------
# --- Imagen resultado -------------------------------------------------------------
# --------------------------------------------------------------------------------
img_res = img.copy()
for i in range(1, num_labels):
    obj = (labels == i).astype(np.uint8)
    cnt, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_res, cnt, contourIdx=-1, color=COLORES[clase[i]], thickness=8)
    texto = f'Dado: {valores[i]}' if i in valores else clase[i]
    cv2.putText(img_res, texto, (stats[i, 0], stats[i, 1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, COLORES[clase[i]], 6)
conteo = ', '.join(f'{sum(c == t for c in clase.values())} de {t}' for t in TIPOS)
imshow(img_res, color_img=True, title=f"Monedas: {conteo}. Dados: {len(dados)} (valores {list(valores.values())})", archivo='12_resultado.png')
cv2.imwrite(str(CAPTURAS / '12_resultado_completa.jpg'), cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))

print('\n=============== Resumen ===============')
print(f'Monedas: {conteo}  (total {len(monedas)})')
print(f'Dados: {len(dados)}  /  Valores de la cara superior: {list(valores.values())}')
print(f'Capturas guardadas en: {CAPTURAS}')
plt.show()
