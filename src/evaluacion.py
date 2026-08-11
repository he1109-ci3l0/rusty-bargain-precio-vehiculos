"""Evaluación y cronometraje de modelos.

Instrumento único de medición, empleado sin excepción por todos los modelos
comparados en el proyecto. Concentrarlo aquí garantiza que la comparación sea
válida y evita la duplicación de código entre secciones.
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 12345
N_PLIEGUES = 5


def recm(y_real, y_predicho):
    """Raíz del error cuadrático medio, en la unidad de la variable objetivo.

    Se calcula de forma explícita y no mediante el argumento squared de
    mean_squared_error, retirado en scikit-learn a partir de la versión 1.6.
    """
    return float(np.sqrt(mean_squared_error(y_real, y_predicho)))


def esquema_validacion(n_pliegues=N_PLIEGUES, semilla=RANDOM_STATE):
    """Validación cruzada con barajado previo.

    Una partición única produce una estimación con varianza propia. Lo que
    aporta la validación cruzada no es solo un promedio más estable, sino la
    desviación entre pliegues, que es la vara para juzgar si una diferencia
    observada entre modelos es real.
    """
    return KFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)


def evaluar_modelo(nombre, constructor, X, y, cv=None, n_latencia=100):
    """Ajusta y cronometra un modelo sobre todos los pliegues.

    Devuelve nueve magnitudes: calidad con su dispersión y rango, y tres
    medidas de tiempo distintas.

    Se emplea time.perf_counter, reloj monótono de alta resolución destinado a
    medir intervalos, y no time.time, que puede retroceder si el sistema ajusta
    la hora. Se reporta la mediana y no el promedio, porque un pico del sistema
    operativo durante una medición contamina el segundo y no la primera.

    La latencia individual se mide aparte, prediciendo un registro a la vez. No
    equivale a dividir el tiempo del lote entre el número de filas: predecir una
    sola observación paga costos fijos que se amortizan en el lote, y es el caso
    real de una aplicación que responde consulta por consulta.
    """
    if cv is None:
        cv = esquema_validacion()

    recms = []
    tiempos_entrenamiento = []
    tiempos_prediccion = []
    modelo = None
    X_val = None

    for idx_tr, idx_val in cv.split(X):
        X_tr, X_val = X.iloc[idx_tr], X.iloc[idx_val]
        y_tr, y_val = y.iloc[idx_tr], y.iloc[idx_val]

        modelo = constructor()

        inicio = time.perf_counter()
        modelo.fit(X_tr, y_tr)
        tiempos_entrenamiento.append(time.perf_counter() - inicio)

        inicio = time.perf_counter()
        prediccion = modelo.predict(X_val)
        tiempos_prediccion.append(time.perf_counter() - inicio)

        recms.append(recm(y_val, prediccion))

    muestra = X.iloc[[0]]
    latencias = []
    for _ in range(n_latencia):
        inicio = time.perf_counter()
        modelo.predict(muestra)
        latencias.append(time.perf_counter() - inicio)

    return {
        "modelo": nombre,
        "recm_media": float(np.mean(recms)),
        "recm_desv": float(np.std(recms, ddof=1)),
        "recm_min": float(np.min(recms)),
        "recm_max": float(np.max(recms)),
        "t_entrenamiento_s": float(np.median(tiempos_entrenamiento)),
        "t_prediccion_lote_s": float(np.median(tiempos_prediccion)),
        "registros_lote": len(X_val),
        "latencia_individual_ms": float(np.median(latencias)) * 1000,
    }


def son_equivalentes(fila_a, fila_b):
    """Regla de decisión fijada antes de conocer resultados.

    Dos modelos se declaran equivalentes en calidad cuando la diferencia de
    RECM es menor que la desviación entre pliegues del mejor de ellos. En ese
    caso la elección se resuelve por velocidad, y entre modelos equivalentes en
    ambas magnitudes se prefiere el de menor complejidad.
    """
    diferencia = abs(fila_a["recm_media"] - fila_b["recm_media"])
    umbral = min(fila_a["recm_desv"], fila_b["recm_desv"])
    return diferencia < umbral


def frontera(tabla, col_calidad="recm_media", col_velocidad="latencia_individual_ms"):
    """Marca los modelos no dominados en calidad y velocidad simultáneamente.

    Un modelo queda dominado cuando existe otro con menor error y menor latencia
    a la vez. Los dominados se descartan sin argumentación adicional.
    """
    tabla = tabla.copy()

    def no_dominado(fila):
        peor_o_igual = (
            (tabla[col_calidad] <= fila[col_calidad])
            & (tabla[col_velocidad] <= fila[col_velocidad])
            & (
                (tabla[col_calidad] < fila[col_calidad])
                | (tabla[col_velocidad] < fila[col_velocidad])
            )
        )
        return not peor_o_igual.any()

    tabla["en_frontera"] = tabla.apply(no_dominado, axis=1)
    return tabla


def exponente_escalamiento(n, tiempos):
    """Estima k en T(n) = c · n^k mediante ajuste sobre los logaritmos.

    Tomando logaritmos, log T = log c + k · log n, de modo que k es la pendiente
    de la recta en escala logarítmica doble. Se devuelve también el coeficiente
    de determinación: un valor bajo indica que la forma de potencia describe mal
    la curva, situación que se presenta cuando el costo está dominado por un
    término independiente de n.
    """
    log_n = np.log(np.asarray(n, dtype=float))
    log_t = np.log(np.asarray(tiempos, dtype=float))

    k, log_c = np.polyfit(log_n, log_t, 1)

    ajustado = k * log_n + log_c
    ss_res = float(np.sum((log_t - ajustado) ** 2))
    ss_tot = float(np.sum((log_t - log_t.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    return {"exponente_k": float(k), "constante_c": float(np.exp(log_c)), "r2": r2}
