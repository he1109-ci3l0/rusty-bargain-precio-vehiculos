"""Preparación del conjunto de vehículos usados de Rusty Bargain.

Reproduce las decisiones documentadas en las secciones 3, 4 y 5 del notebook:
auditoría de calidad, exclusión por riesgo de fuga, limpieza, imputación,
ingeniería de características y las dos representaciones de las categóricas.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 12345
ANIO_REFERENCIA = 2016

COLUMNAS_CATEGORICAS = [
    "VehicleType", "Gearbox", "Model", "FuelType", "Brand", "NotRepaired",
]

COLUMNAS_EXCLUIDAS_POR_FUGA = [
    "DateCrawled", "DateCreated", "LastSeen", "NumberOfPictures",
]

CARACTERISTICAS = [
    "VehicleType", "Gearbox", "Power", "Model", "Mileage",
    "RegistrationMonth", "FuelType", "Brand", "NotRepaired",
    "antiguedad", "potencia_ausente", "mes_ausente",
]

PRECIO_MINIMO = 100
ANIO_MINIMO = 1900
POTENCIA_MAXIMA = 1000


def cargar(ruta):
    """Lee el archivo de origen sin transformación alguna."""
    return pd.read_csv(ruta)


def limpiar(df):
    """Aplica las reglas de limpieza y devuelve el marco resultante con su bitácora.

    El orden es deliberado. Los ceros de Power y RegistrationMonth se convierten
    a valor ausente antes del filtro por rango; de lo contrario se perderían
    registros cuyo único defecto está en una variable.
    """
    datos = df.copy()
    registros_inicio = len(datos)
    bitacora = []

    datos = datos.drop_duplicates()
    bitacora.append(("Duplicados completos", registros_inicio - len(datos), len(datos)))

    datos = datos.drop(columns=COLUMNAS_EXCLUIDAS_POR_FUGA)

    datos.loc[datos["Power"] == 0, "Power"] = np.nan
    datos.loc[datos["RegistrationMonth"] == 0, "RegistrationMonth"] = np.nan
    bitacora.append(("Ceros convertidos a ausente", 0, len(datos)))

    previo = len(datos)
    datos = datos[datos["Price"] >= PRECIO_MINIMO]
    bitacora.append((f"Precio menor a {PRECIO_MINIMO}", previo - len(datos), len(datos)))

    previo = len(datos)
    datos = datos[
        (datos["RegistrationYear"] >= ANIO_MINIMO)
        & (datos["RegistrationYear"] <= ANIO_REFERENCIA)
    ]
    bitacora.append(("Ano de matriculacion fuera de rango", previo - len(datos), len(datos)))

    previo = len(datos)
    datos = datos[(datos["Power"].isna()) | (datos["Power"] <= POTENCIA_MAXIMA)]
    bitacora.append((f"Potencia superior a {POTENCIA_MAXIMA}", previo - len(datos), len(datos)))

    datos = datos.reset_index(drop=True)

    resumen = pd.DataFrame(
        bitacora, columns=["regla", "registros_eliminados", "registros_restantes"]
    )
    return datos, resumen


def imputar(datos):
    """Imputa las categóricas con etiqueta explícita.

    La ausencia en NotRepaired probablemente indica que el vendedor evitó
    declarar daños, circunstancia informativa por sí misma. Imputar con la moda
    la borraría. Power y RegistrationMonth conservan sus ausentes, que las
    librerías de potenciación manejan de forma nativa.
    """
    datos = datos.copy()
    for col in COLUMNAS_CATEGORICAS:
        datos[col] = datos[col].fillna("desconocido")
    return datos


def derivar(datos):
    """Construye antigüedad e indicadores de ausencia.

    Los indicadores no son adorno. Los anuncios sin potencia declarada
    presentan mediana de 1 300 euros contra 3 200 de los que la declaran.
    """
    datos = datos.copy()
    datos["antiguedad"] = ANIO_REFERENCIA - datos["RegistrationYear"]
    datos["potencia_ausente"] = datos["Power"].isna().astype(int)
    datos["mes_ausente"] = datos["RegistrationMonth"].isna().astype(int)
    return datos


def representar(datos):
    """Devuelve las dos representaciones de las variables categóricas.

    La nativa, de doce columnas, para LightGBM y CatBoost.
    La indicadora, de más de trescientas, para XGBoost y scikit-learn.

    El argumento drop_first evita que la suma de las columnas de una misma
    variable reproduzca el intercepto y haga singular a la matriz XtX.
    """
    X = datos[CARACTERISTICAS].copy()
    y = datos["Price"].copy()

    X_nativa = X.copy()
    for col in COLUMNAS_CATEGORICAS:
        X_nativa[col] = X_nativa[col].astype("category")

    X_indicadora = X.copy()
    X_indicadora["Power"] = X_indicadora["Power"].fillna(X_indicadora["Power"].median())
    X_indicadora["RegistrationMonth"] = X_indicadora["RegistrationMonth"].fillna(
        X_indicadora["RegistrationMonth"].median()
    )
    X_indicadora = pd.get_dummies(
        X_indicadora, columns=COLUMNAS_CATEGORICAS, drop_first=True
    )

    X_catboost = X_nativa.copy()
    for col in COLUMNAS_CATEGORICAS:
        X_catboost[col] = X_catboost[col].astype(str)

    return X_nativa, X_indicadora, X_catboost, y


def preparar(ruta):
    """Ejecuta la preparación completa desde el archivo de origen."""
    df = cargar(ruta)
    datos, resumen = limpiar(df)
    datos = imputar(datos)
    datos = derivar(datos)
    X_nativa, X_indicadora, X_catboost, y = representar(datos)
    return {
        "datos": datos,
        "resumen_limpieza": resumen,
        "X_nativa": X_nativa,
        "X_indicadora": X_indicadora,
        "X_catboost": X_catboost,
        "y": y,
    }
