# Rusty Bargain — Estimación del valor de mercado de vehículos usados

![Python](https://img.shields.io/badge/Python-3.11-1A4A4F?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-9A6A8C?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-3A6878?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-4A88A4?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-1E3130?style=flat-square&logo=scikitlearn&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-Completado-A8D8CC?style=flat-square)

Modelo de valuación automática para un servicio de venta de coches de segunda mano. El encargo declara
tres criterios simultáneos de interés: **calidad de la predicción, velocidad de la predicción y tiempo
de entrenamiento**. El trabajo demuestra que los tres no admiten optimización conjunta y construye la
frontera de alternativas defendibles.

---

## Resultado

| | |
|---|---|
| **Modelo recomendado** | LightGBM — 600 árboles, η 0.05, 127 hojas |
| **RECM en prueba** | **1 517.36 €** |
| **Proceso actual (tabla de referencia)** | 2 218.66 € |
| **Reducción del error** | **701.30 € — 31.6 %** |
| **Latencia de predicción individual** | 0.659 ms |
| **Tiempo de entrenamiento** | 10.975 s |
| **Exponente de escalamiento** | k = 0.058 |

---

## Comparación completa

| Modelo | RECM (€) | σ pliegues | Entren. (s) | Lote (s) | Latencia (ms) | k |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM** | **1 527.02** | 19.13 | 10.975 | 0.242 | 0.659 | 0.058 |
| XGBoost | 1 556.75 | 18.98 | 7.203 | 0.088 | 5.376 | 0.589 |
| CatBoost | 1 567.63 | 20.40 | 41.538 | 0.055 | 0.221 | 0.782 |
| Bosque aleatorio | 1 628.43 | 18.22 | 8.048 | 0.065 | 14.622 | 1.292 |
| Árbol de decisión | 1 915.65 | 12.53 | 1.521 | 0.017 | 0.388 | 1.184 |
| *Tabla de referencia* | *2 218.66* | *29.55* | *0.018* | *0.009* | *0.925* | — |
| *Regresión lineal* | *2 718.05* | *37.35* | *1.226* | *0.022* | *0.361* | — |
| *Constante (media)* | *4 546.83* | *29.94* | *0.000* | *0.000* | *0.005* | — |

Validación cruzada de 5 pliegues sobre 245 106 registros de entrenamiento. Las líneas base van en cursiva.

---

## Hallazgos

**Uno. El rival verdadero no es la regresión lineal.** Una tabla de medianas por marca, modelo y
antigüedad alcanza 2 218.66 € y supera al modelo lineal en 499 €. El precio de un vehículo no es
aditivo en sus características, y la forma lineal no puede representar las interacciones que la
agrupación captura de forma implícita.

**Dos. Los tres criterios del cliente son incompatibles entre sí.** El modelo de mayor calidad es el
más lento por lote. El más rápido por lote es el más lento de entrenar. El más rápido de entrenar es
el de peor calidad entre los métodos de potenciación. La frontera reduce las alternativas a dos.

**Tres. La latencia individual no se deduce del tiempo por lote.** El sobrecosto de predecir un
registro aislado va de 133 a 11 040 veces el costo por registro dentro del lote, según el modelo. El
bosque aleatorio pasa de 1.3 µs por registro en lote a 14 622 µs aislado: el reparto entre núcleos
cuesta más que el cálculo.

**Cuatro. LightGBM no escala con el número de registros.** Su exponente empírico es k = 0.058. El
tiempo crece 16 % ante un aumento de 900 % en los datos, porque el método de histogramas busca cortes
sobre 255 contenedores fijos y no sobre los registros. A diez millones de registros la extrapolación
sitúa el reentrenamiento en 13 segundos, frente a 22 minutos del bosque aleatorio.

**Cinco. La expansión de columnas abarata la búsqueda de cortes.** XGBoost procesa 314 columnas
indicadoras y entrena más rápido que LightGBM sobre 12 columnas nativas. Una columna binaria admite un
único punto de corte; una categórica de 250 niveles admite muchos más.

**Seis. La RECM oculta el fallo en el segmento económico.** El error relativo es de 166.8 % en el
rango de 100 a 999 €, que agrupa el 20.4 % del conjunto de prueba. El error cuadrático medio penaliza
el cuadrado de la desviación y concentra la capacidad del modelo en el extremo superior.

---

## Estructura

```
proyecto_15MetodosNumericos/
├── data/                  car_data.csv (no versionado, 41 MB)
├── notebooks/
│   ├── rusty_bargain.ipynb        análisis completo, 18 secciones
│   └── plantilla_tripleten.ipynb  plantilla original, referencia
├── src/                   funciones reutilizables
├── figures/               gráficas exportadas
├── docs/                  sitio del caso de estudio
├── environment.yml
└── README.md
```

---

## Metodología

Ocho secciones de análisis con protocolo declarado antes de ver resultados.

- **Auditoría de calidad** — 5 hallazgos documentados: columna constante, ausencias codificadas como
  ceros, valores fuera de rango físico, precio nulo, duplicación.
- **Lista de verificación de fuga** — 4 columnas excluidas por información posterior al momento de la
  consulta: `DateCrawled`, `DateCreated`, `LastSeen`, `NumberOfPictures`.
- **Tres líneas base** — constante, tabla de referencia por grupo, regresión lineal. La segunda es el
  proceso actual y constituye el umbral que todo modelo debe superar.
- **Regla de decisión fijada de antemano** — si la diferencia de RECM entre dos modelos es menor que la
  desviación entre pliegues del mejor, se declaran equivalentes y decide la velocidad.
- **Cronometraje con reloj monótono** — mediana sobre 5 repeticiones, con separación entre
  entrenamiento, predicción por lote y latencia individual.
- **Escalamiento empírico** — ajuste de T(n) = c · n^k sobre 5 tamaños de muestra, con R² reportado.

---

## Reproducción

```bash
conda env create -f environment.yml
conda activate rusty-bargain
python -m ipykernel install --user --name rusty-bargain --display-name "Python (rusty-bargain)"
```

Coloca `car_data.csv` en `data/` y ejecuta el notebook completo desde `notebooks/`.

Semilla fija: `RANDOM_STATE = 12345`.

---

## Salvedades

Las cifras absolutas de RECM son una estimación optimista: se conservaron 27 543 registros coincidentes
en características y precio, por considerarse señal de mercado. El efecto opera por igual sobre todos
los modelos y no altera el orden relativo.

La extrapolación a diez millones de registros cubre un factor de 41 sobre el rango medido. Es
orientación de magnitud, no predicción.

Las particiones son aleatorias, no temporales. El conjunto abarca dos meses de captura de 2016.

---

*Proyecto desarrollado como parte del bootcamp de Data Science de TripleTen. Sprint 15 — Métodos
numéricos. Métrica oficial: RECM sobre `Price`.*
