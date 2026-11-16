# TP2: monedas, dados y patentes

Materia: Procesamiento de Imágenes I (IA 4.4), Tecnicatura Universitaria en Inteligencia Artificial, FCEIA, Universidad Nacional de Rosario. Año 2026, 2° semestre. Grupo 8.

## Autores

- Civetta, Valentino
- Frank, Maximiliano
- Fucci, Milagros
- Winter, Federico

## Descripción

El trabajo tiene dos problemas. La consigna completa está en `TUIA_PDI_TP2_2026_C2.pdf`.

El Problema 1 segmenta las monedas y los dados de `monedas.jpg`, clasifica y cuenta las monedas por tipo e informa el valor de la cara superior de cada dado.

El Problema 2 detecta la patente en cada una de las doce imágenes `img<id>.png` y segmenta sus seis caracteres.

## Estructura de la carpeta

```
.
├── README.md                       Este archivo.
├── Informe_TP2_PDI_Grupo8.pdf      Informe del trabajo práctico.
├── TUIA_PDI_TP2_2026_C2.pdf        Consigna del TP.
├── requirements.txt                Dependencias del proyecto.
├── problema1.py                    Problema 1: monedas y dados.
├── problema2.py                    Problema 2: patentes.
├── generar_informe.py              Arma el PDF del informe con las capturas.
├── Problema1_explicacion.md        Resolución paso a paso del Problema 1.
├── Problema2_explicacion.md        Resolución paso a paso del Problema 2.
├── monedas.jpg                     Entrada del Problema 1.
├── img01.png ... img12.png         Entradas del Problema 2.
├── Factor de Forma.txt             Material de la cátedra (Problema 1).
├── info_patentes.txt               Material de la cátedra (Problema 2).
└── capturas/
    ├── problema1/                  Figuras de cada etapa del Problema 1.
    └── problema2/                  Figuras de cada etapa del Problema 2.
```

Los dos scripts crean la carpeta `capturas/` si no existe y reescriben las figuras en cada ejecución.

## Requisitos

Python 3.14.6. Es la versión con la que ejecutamos y verificamos los comandos de este README.

| Librería | Versión | Uso |
|---|---|---|
| numpy | 2.5.3 | Arrays y operaciones sobre `stats` |
| opencv-python | 5.0.0.93 | Canny, morfología, umbralado, componentes conectadas, contornos, K-means |
| matplotlib | 3.11.2 | Figuras de cada etapa y PDF del informe |

El archivo `requirements.txt` no fija versiones. En nuestro entorno estaba instalado `opencv-contrib-python` 5.0.0.93, que incluye todo lo de `opencv-python`.

## Instalación

Desde esta carpeta, en Linux o macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows cambia la activación del entorno:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecución

Ejecutá los scripts desde esta carpeta, con el entorno virtual activado. Ninguno recibe argumentos.

### problema1.py

```bash
python problema1.py
```

- Entrada: `monedas.jpg`.
- Salida por consola: el resultado de cada etapa, una línea por objeto con su área y su factor de forma, la cantidad de monedas de cada tipo y el valor de cada dado.
- Salida gráfica: 13 ventanas de matplotlib, una por etapa. Cuando cerrás las ventanas el script termina.
- Archivos que genera: las mismas figuras en `capturas/problema1/` y la imagen resultado en tamaño completo (`12_resultado_completa.jpg`).

La ejecución tardó unos 26 segundos en la máquina donde la medimos.

### problema2.py

```bash
python problema2.py
```

- Entrada: `img01.png` a `img12.png`.
- Salida por consola, para cada imagen: el umbral elegido, cuántos objetos quedan después de cada filtro, la posición de la patente y los datos de cada caracter. Al final, un resumen con las imágenes que fallan.
- Salida gráfica: una ventana por imagen con las seis etapas y una ventana con las doce patentes.
- Archivos que genera: `capturas/problema2/img<id>_etapas.png` y `capturas/problema2/resumen_patentes.png`.

La ejecución tardó unos 11 segundos.

### Sin pantalla

Para correr sin abrir ventanas (por ejemplo en un servidor), anteponé `MPLBACKEND=Agg`. Los scripts imprimen y guardan las capturas igual:

```bash
MPLBACKEND=Agg python problema1.py
MPLBACKEND=Agg python problema2.py
```

## Informe

El informe está en [Informe_TP2_PDI_Grupo8.pdf](Informe_TP2_PDI_Grupo8.pdf). Se arma con `python generar_informe.py`, que usa las capturas ya generadas por los dos scripts.

La resolución paso a paso de cada problema está en [Problema1_explicacion.md](Problema1_explicacion.md) y [Problema2_explicacion.md](Problema2_explicacion.md).
