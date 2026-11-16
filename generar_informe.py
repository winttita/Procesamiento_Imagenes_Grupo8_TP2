# Arma Informe_TP2_PDI_Grupo8.pdf con matplotlib (PdfPages), usando las capturas que generan problema1.py y problema2.py.
# Ejecutar primero los dos scripts y después:  python generar_informe.py
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CARPETA = Path(__file__).parent
C1 = CARPETA / 'capturas' / 'problema1'
C2 = CARPETA / 'capturas' / 'problema2'
ANCHO, ALTO = 8.27, 11.69                   # A4 en pulgadas
IZQ, DER, ARR, ABA = 0.12, 0.88, 0.93, 0.07 # Márgenes (fracción de la página)
plt.rcParams['savefig.dpi'] = 200           # Resolución de las capturas dentro del PDF

# --- Contenido: lista de bloques (tipo, texto) o ('fig', [rutas], epígrafe) ------
INFORME = [
    ('h1', '1. Descripción de los ejercicios'),
    ('p', 'El trabajo práctico tiene dos problemas. Cada uno tiene su script en Python: problema1.py y problema2.py. Los dos se ejecutan sin argumentos, imprimen el resultado de cada etapa y guardan las figuras en la carpeta capturas.'),
    ('h2', '1.1 Problema 1: detección y clasificación de monedas y dados'),
    ('p', 'La imagen monedas.jpg muestra monedas de distinto valor y tamaño y dos dados sobre un fondo de intensidad no uniforme. La consigna pide:'),
    ('li', 'A. Segmentar las monedas y los dados de manera automática.'),
    ('li', 'B. Clasificar los distintos tipos de monedas y hacer un conteo automático.'),
    ('li', 'C. Determinar el valor de la cara superior de cada dado y hacer un conteo automático.'),
    ('h2', '1.2 Problema 2: detección de patentes'),
    ('p', 'Hay doce imágenes (img01.png a img12.png) con la vista anterior o posterior de un vehículo. Para cada una la consigna pide:'),
    ('li', 'A. Detectar automáticamente la placa patente y segmentarla.'),
    ('li', 'B. Segmentar los caracteres de la patente detectada en el punto A.'),
    ('p', 'En los dos problemas el script debe informar y mostrar los resultados de cada etapa del procesamiento.'),

    ('h1', '2. Análisis de los problemas'),
    ('p', 'Todos los números de este informe salen de la ejecución de los scripts o de medidas que hicimos con las mismas funciones sobre las imágenes de entrada.'),
    ('h2', '2.1 Problema 1'),
    ('p', 'La imagen mide 3871x2681 píxeles. Contando a mano hay 17 monedas (9 de 10 centavos, 5 de 1 peso y 3 de 50 centavos) y 2 dados, con un 5 y un 3 en la cara superior. Usamos ese conteo para verificar la salida del script.'),
    ('p', 'El fondo no es uniforme. En gris, la franja izquierda de la imagen tiene una media de 53 y la franja derecha una media de 170. Hay monedas más oscuras que el fondo claro y más claras que el fondo oscuro, así que un umbral global no separa los objetos. Por eso segmentamos con detección de bordes, que depende del cambio local de intensidad.'),
    ('p', 'El fondo es una tela con textura. Canny marca esa textura como borde si la imagen no se suaviza lo suficiente. Con un filtro de 5x5 y umbrales 15 y 45 el relleno de huecos tomó la imagen completa como un único objeto. En el otro extremo, con un filtro de 15x15 y umbrales 30 y 90 varios contornos no cerraron y solo quedaron 14 objetos grandes.'),
    ('p', 'Los contornos de Canny tienen cortes. En la primera prueba el dado de la derecha y una moneda de 50 centavos quedaron abiertos y no se rellenaron. Ese dado es blanco y está sobre la parte clara del fondo, con poco contraste en su borde derecho.'),
    ('p', 'La dilatación que cierra los contornos también une objetos cercanos. Después del relleno contamos 18 objetos grandes en lugar de 19.'),
    ('p', 'Para clasificar las monedas medimos el área y la saturación media en HSV. Las de 50 centavos tienen saturación entre 123.9 y 141.6 y se distinguen bien, pero una moneda de 10 centavos gastada da 62.9 y una de 1 peso da 65.0. El color solo no alcanza. El área separa los tres tipos con más de 11000 píxeles entre grupos.'),
    ('p', 'En los dados se ven puntos de las caras laterales y sombras, que no hay que contar.'),
    ('h2', '2.2 Problema 2'),
    ('p', 'Las imágenes miden 640x360 píxeles y la patente ocupa una parte chica. Los caracteres miden entre 10 y 21 píxeles de alto y entre 5 y 13 de ancho, con trazos de 1 o 2 píxeles.'),
    ('p', 'La iluminación cambia de una foto a otra. Probamos umbrales de 20 a 240 de a 5 y en cada imagen funciona un rango distinto: img01 necesita entre 145 y 180 e img12 entre 80 y 110. No hay un umbral que sirva para las doce. Otsu sobre la imagen completa da entre 92 y 134 y en varias imágenes queda fuera del rango que funciona.'),
    ('p', 'Después de umbralar aparecen cientos de objetos (hasta 648 en img01). Muchos tienen el tamaño y la forma de un caracter: reflejos, adoquines, partes de la parrilla. Hace falta un criterio que use la relación entre los caracteres.'),
    ('p', 'En img03 la letra D está pegada al marco de la patente. Entre el trazo de la letra y el marco hay una o dos columnas de píxeles de un gris intermedio. Con ningún umbral la D queda como un objeto separado y entero.'),

    ('h1', '3. Técnicas utilizadas'),
    ('p', 'Trabajamos con Python 3.14.6, numpy 2.5.3, OpenCV 5.0.0.93 y matplotlib 3.11.2. Entre paréntesis va la unidad de la materia de donde sale cada técnica.'),
    ('h2', '3.1 Problema 1'),
    ('li', 'Filtro pasabajos Gaussiano de 15x15 (U3). Atenúa la textura del fondo antes de buscar bordes. Barrimos de 5 a 21.'),
    ('li', 'Canny con umbrales 15 y 45 (U3). Segmenta por bordes sin depender del nivel del fondo. Barrimos el umbral bajo de 8 a 50 con el alto igual al triple.'),
    ('li', 'Dilatación con elipse de 35x35 (U6). Cierra los cortes de los contornos. Con 27, 35 y 45 se obtienen 19 objetos y elegimos el valor del medio.'),
    ('li', 'Rellenado de huecos con cv2.floodFill, función imfillhole_v2 de la cátedra (U6). Se aplica con los contornos ya cerrados.'),
    ('li', 'Erosión con la misma elipse de 35x35 (U6). Devuelve el tamaño original y separa los objetos que la dilatación unió.'),
    ('li', 'Apertura con elipse de 41x41 (U6). Elimina 5 restos de 231 píxeles como máximo y suaviza los contornos.'),
    ('li', 'Componentes conectadas con cv2.connectedComponentsWithStats (U3). Da el área y el bounding box de cada objeto.'),
    ('li', 'Factor de forma 1/Fp = Perímetro² / Área, con findContours, arcLength y contourArea (U7 y Factor de Forma.txt). Separa monedas de dados. Umbral 15.5: las monedas dan entre 14.16 y 14.83 y los dados 16.13 y 17.71.'),
    ('li', 'K-means con K = 3 sobre el área (U5). Agrupa las monedas por tamaño sin escribir rangos a mano. Los grupos se nombran por orden de tamaño: 10 centavos, 1 peso y 50 centavos.'),
    ('li', 'Umbral de Otsu sobre los píxeles de cada dado (U3). Separa los puntos negros del dado blanco. Dio 116 y 121.'),
    ('li', 'Factor de forma sobre las manchas oscuras del dado, con el mismo umbral 15.5. Los puntos de la cara superior dan entre 13.90 y 14.74. Los laterales y las sombras dan más de 32.'),
    ('h2', '3.2 Problema 2'),
    ('li', 'Umbralado global con cv2.threshold (U3), repetido con T de 20 a 240 de a 5.'),
    ('li', 'Componentes conectadas con conectividad 8 (U3).'),
    ('li', 'Filtrado por área, entre 15 y 300 píxeles (U6, clasifico_celulas.py). Los caracteres midieron entre 18 y 161. Con el límite del 10 % de la imagen que sugirió la cátedra el resultado es el mismo en las doce imágenes.'),
    ('li', 'Filtrado por relación de aspecto alto/ancho entre 1.5 y 3.0 (info_patentes.txt).'),
    ('li', 'Filtrado por cercanía. Dos candidatos son vecinos si su alto difiere menos del 25 %, su posición vertical menos de medio alto y su distancia horizontal menos de 6 altos. Se conserva el grupo más numeroso. Esta regla la armamos nosotros a partir de los datos de info_patentes.txt.'),
    ('li', 'Elección del umbral. De todos los T que dan el grupo más grande (hasta 6 caracteres) se usa el del medio. La idea viene de un intento anterior nuestro.'),
    ('li', 'Patente: rectángulo que encierra a los caracteres más un margen de 0.8 altos de caracter en horizontal y 0.6 en vertical.'),
    ('li', 'Caracteres: el mismo procesamiento aplicado solo al recorte de la patente.'),

    ('h1', '4. Desarrollo paso a paso'),
    ('h2', '4.1 Problema 1'),
    ('fig', [C1 / '01_original.png', C1 / '02_gris_blur.png'], 'Figura 1. Imagen original (izquierda) y en escala de grises con filtro Gaussiano de 15x15 (derecha). El fondo es más oscuro a la izquierda que a la derecha.'),
    ('fig', [C1 / '03_canny.png', C1 / '04_dilatacion.png'], 'Figura 2. Bordes de Canny con umbrales 15 y 45 (izquierda): 135953 píxeles de borde en 3716 trozos. Después de dilatar con una elipse de 35x35 (derecha) los contornos quedan cerrados.'),
    ('fig', [C1 / '05_relleno.png', C1 / '06_erosion.png'], 'Figura 3. Rellenado de huecos (izquierda): 18 objetos grandes, porque dos quedaron unidos. Erosión de 35x35 (derecha): 19 objetos grandes y 5 restos chicos.'),
    ('fig', [C1 / '07_apertura.png', C1 / '08_componentes.png'], 'Figura 4. Apertura de 41x41 (izquierda), que elimina los restos. Componentes conectadas (derecha): 19 objetos, un color por etiqueta.'),
    ('fig', [C1 / '09_factor_forma.png', C1 / '10_kmeans_areas.png'], 'Figura 5. Clasificación por factor de forma (izquierda): 17 monedas en verde y 2 dados en rojo. K-means sobre el área de las monedas (derecha): tres grupos con centros en 62053, 89547 y 105214 píxeles.'),
    ('fig', [C1 / '11_dado_1.png'], 'Figura 6. Dado 1: recorte, umbralado con Otsu (116) y manchas clasificadas. Las 5 manchas circulares (verde) son los puntos de la cara superior. Las 3 rojas son puntos de una cara lateral.'),
    ('fig', [C1 / '11_dado_2.png'], 'Figura 7. Dado 2: recorte, umbralado con Otsu (121) y manchas clasificadas. Hay 3 puntos circulares. Las manchas rojas son sombra y puntos laterales.'),
    ('fig', [C1 / '12_resultado.png'], 'Figura 8. Imagen resultado. Cada objeto lleva su contorno y su clase: 10 centavos (celeste), 1 peso (verde), 50 centavos (magenta) y dado con su valor (rojo).'),
    ('h2', '4.2 Problema 2'),
    ('fig', [C2 / 'img05_etapas.png'], 'Figura 9. Etapas en img05, la imagen del enunciado, con T = 150. De 188 componentes pasan 32 el filtro de área (rojo), 9 el de aspecto (amarillo) y 6 el de cercanía (verde). Abajo: patente detectada, recorte umbralado y caracteres segmentados.'),
    ('fig', [C2 / 'img12_etapas.png'], 'Figura 10. Etapas en img12, la patente más chica, con T = 95. Los caracteres miden 10 u 11 píxeles de alto. Se segmentan los 6.'),
    ('fig', [C2 / 'img03_etapas.png'], 'Figura 11. Etapas en img03, la única que falla, con T = 115. La patente se detecta, pero en el recorte umbralado la D está unida al marco y solo se segmentan 5 caracteres.'),
    ('fig', [C2 / 'resumen_patentes.png'], 'Figura 12. Patente detectada y caracteres segmentados en las doce imágenes, con el umbral usado en cada una. Las etapas de las demás imágenes están en capturas/problema2.'),

    ('h1', '5. Resultados finales'),
    ('h2', '5.1 Problema 1'),
    ('p', 'Salida del script: 9 monedas de 10 centavos, 5 de 1 peso y 3 de 50 centavos (17 en total), y 2 dados con valores 5 y 3. Coincide con el conteo manual.'),
    ('pre', 'Tipo          Cantidad   Área mín.   Área máx.   Centro K-means\n'
            '10 centavos       9        59829       64660        62053\n'
            '1 peso            5        87279       93021        89547\n'
            '50 centavos       3       104861      105438       105214'),
    ('h2', '5.2 Problema 2'),
    ('p', 'La patente se detecta en las 12 imágenes. Los 6 caracteres se segmentan en 11 de las 12. Falla img03, con 5 de 6. La tabla muestra, para cada imagen, el rango de umbrales que da el grupo más grande, el umbral usado y cuántos objetos quedan después de cada filtro.'),
    ('pre', 'Imagen   Umbrales     T    Compon.   Área   Aspecto   Cercanía   Caracteres\n'
            'img01    145 a 180   165     648     104      12        6          6\n'
            'img02     90 a 120   105     159      38       9        6          6\n'
            'img03    105 a 125   115     363      53       6        5          5  (falla)\n'
            'img04    125 a 175   150     337      50       7        6          6\n'
            'img05    110 a 190   150     188      32       9        6          6\n'
            'img06     55 a 120    95      67      18       7        6          6\n'
            'img07     75 a 120   100      68      19       6        6          6\n'
            'img08    130 a 190   160     414      91      14        6          6\n'
            'img09     80 a 120   100     294      39       9        6          6\n'
            'img10    115 a 140   130     488     101       9        6          6\n'
            'img11    135 a 150   145     602     108      15        6          6\n'
            'img12     80 a 110    95      72      18      10        6          6'),
    ('p', 'Causa de la falla en img03: la D toca el marco de la patente. Con T = 120 la letra y un pedazo del marco forman una componente de 16x24 píxeles, que el filtro por cercanía descarta por ser más alta que los otros caracteres. Con 125 o más la D se parte en pedazos. Probamos umbrales de a 1, conectividad 4, agrandar la imagen, top-hat en grises y umbral adaptativo, y img03 siguió en 5 caracteres. Una erosión de 2x1 la llevó a 6 con un solo umbral, pero img01 e img11 dejaron de funcionar, porque corta los trazos de 1 o 2 píxeles.'),

    ('h1', '6. Conclusiones'),
    ('p', 'En el Problema 1 la segmentación por bordes resolvió el fondo no uniforme, que un umbral global no puede separar. El costo es la sensibilidad a los parámetros: el suavizado y los umbrales de Canny tienen un rango angosto en el que se obtienen los 19 objetos. Los valores que usamos están ajustados para esta imagen y habría que revisarlos con otra foto.'),
    ('p', 'El orden de las operaciones morfológicas importa. Hay que cerrar los contornos antes de rellenar, y erosionar después para separar lo que la dilatación unió.'),
    ('p', 'El factor de forma nos sirvió dos veces con el mismo umbral: para separar monedas de dados y para quedarnos con los puntos de la cara superior. El valor medido para un círculo (cerca de 14.2) es mayor que el teórico (12.57) porque el perímetro se mide sobre píxeles. Por eso fijamos el umbral con los valores medidos.'),
    ('p', 'Para clasificar las monedas el área alcanzó. K-means encontró los tres grupos sin que fijáramos rangos. El color quedó descartado como único criterio por las monedas gastadas.'),
    ('p', 'En el Problema 2 ningún umbral fijo sirve para las doce imágenes. Recorrer umbrales y usar la cantidad de caracteres agrupados como criterio de elección resolvió 11 de 12. Los filtros de área y de aspecto dejan pasar entre 6 y 15 objetos por imagen. El filtro por cercanía es el que deja solo los caracteres.'),
    ('p', 'El límite del método está en la resolución. Cuando un caracter toca el marco, como en img03, las componentes conectadas no lo separan y la erosión que podría hacerlo rompe los caracteres de otras imágenes. La patente igual se detecta con los 5 caracteres restantes.'),
]


def nueva_pagina(pdf, fig, n):
    if fig is not None:
        fig.text(0.5, 0.035, str(n), ha='center', fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
    return plt.figure(figsize=(ANCHO, ALTO)), ARR


def portada(pdf):
    fig = plt.figure(figsize=(ANCHO, ALTO))
    lineas = [(0.86, 'Universidad Nacional de Rosario', 14, 'bold'),
              (0.83, 'Facultad de Ciencias Exactas, Ingeniería y Agrimensura', 12, 'normal'),
              (0.80, 'Tecnicatura Universitaria en Inteligencia Artificial', 12, 'normal'),
              (0.74, 'Asignatura: IA 4.4 - Procesamiento de Imágenes I', 12, 'normal'),
              (0.71, 'Año 2026, 2° semestre', 12, 'normal'),
              (0.58, 'TRABAJO PRÁCTICO N° 2', 20, 'bold'),
              (0.53, '"Detección y clasificación de monedas y dados', 14, 'normal'),
              (0.50, 'y detección de patentes"', 14, 'normal'),
              (0.40, 'GRUPO 8', 14, 'bold'),
              (0.36, 'Integrantes:', 12, 'normal'),
              (0.33, 'CIVETTA, Valentino', 12, 'normal'),
              (0.305, 'FRANK, Maximiliano', 12, 'normal'),
              (0.28, 'FUCCI, Milagros', 12, 'normal'),
              (0.255, 'WINTER, Federico', 12, 'normal'),
              (0.16, 'Fecha: 14/11/2026', 12, 'normal')]
    for y, s, size, weight in lineas:
        fig.text(0.5, y, s, ha='center', fontsize=size, fontweight=weight)
    pdf.savefig(fig)
    plt.close(fig)


def alto_fig(rutas):
    # Ancho y alto (fracción de página) de una fila de imágenes, respetando el aspecto.
    w = (DER - IZQ - 0.02*(len(rutas) - 1)) / len(rutas)
    imgs = [plt.imread(r) for r in rutas]
    h = max(w*ANCHO*im.shape[0]/im.shape[1] for im in imgs) / ALTO
    return w, h, imgs


def alto_minimo(i):
    # Alto que necesita el bloque i para no quedar cortado: una figura o una tabla enteras,
    # un título junto con lo que le sigue, o el comienzo de un párrafo.
    tipo = INFORME[i][0]
    if tipo == 'fig':
        return alto_fig(INFORME[i][1])[1] + 0.016*len(textwrap.wrap(INFORME[i][2], 96)) + 0.02
    if tipo == 'pre':
        return 0.015*(INFORME[i][1].count('\n') + 1)
    if tipo in ('h1', 'h2'):
        return 0.06 + alto_minimo(i + 1)
    return 0.05


with PdfPages(CARPETA / 'Informe_TP2_PDI_Grupo8.pdf') as pdf:
    portada(pdf)
    n_pag = 2
    fig, y = nueva_pagina(pdf, None, n_pag)
    for i, bloque in enumerate(INFORME):
        tipo = bloque[0]
        if y - alto_minimo(i) < ABA:                    # El bloque no entra en lo que queda de la página
            n_pag += 1
            fig, y = nueva_pagina(pdf, fig, n_pag - 1)
        if tipo == 'fig':
            w, h, imgs = alto_fig(bloque[1])
            lineas = textwrap.wrap(bloque[2], 96)
            for k, im in enumerate(imgs):
                ax = fig.add_axes([IZQ + k*(w + 0.02), y - h, w, h])
                ax.imshow(im), ax.axis('off')
            y -= h + 0.012
            for linea in lineas:
                fig.text(IZQ, y, linea, fontsize=8.5, style='italic', va='top')
                y -= 0.016
            y -= 0.018
            continue
        size, weight, ancho, familia, antes, despues = {'h1': (14, 'bold', 60, 'sans-serif', 0.022, 0.012),
                                                        'h2': (11.5, 'bold', 75, 'sans-serif', 0.012, 0.008),
                                                        'p': (10, 'normal', 90, 'sans-serif', 0, 0.010),
                                                        'li': (10, 'normal', 86, 'sans-serif', 0, 0.006),
                                                        'pre': (8, 'normal', 200, 'monospace', 0, 0.012)}[tipo]
        lineas = bloque[1].split('\n') if tipo == 'pre' else textwrap.wrap(bloque[1], ancho)
        dy = size*1.5/72/ALTO                                                   # Alto de una línea (fracción de página)
        y -= antes
        for k, linea in enumerate(lineas):
            if y - dy < ABA:
                n_pag += 1
                fig, y = nueva_pagina(pdf, fig, n_pag - 1)
            if tipo == 'li':
                linea = ('•  ' if k == 0 else '    ') + linea
            fig.text(IZQ + (0.015 if tipo == 'li' else 0), y, linea, fontsize=size, fontweight=weight, family=familia, va='top')
            y -= dy
        y -= despues
    nueva_pagina(pdf, fig, n_pag)
    plt.close('all')
print(f'Informe generado: {CARPETA / "Informe_TP2_PDI_Grupo8.pdf"} ({n_pag} páginas)')
