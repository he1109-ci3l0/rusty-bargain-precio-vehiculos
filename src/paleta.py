"""Paleta y configuración visual del proyecto.

Paleta bibliografia_BoneID, con roles fijos por familia de modelo. Aplicar el
mismo color a la misma familia en todas las gráficas evita que el lector tenga
que releer la leyenda en cada figura.
"""

import matplotlib.pyplot as plt

COLORES = {
    "fondo": "#D5DBE7",
    "rejilla": "#5C5468",
    "texto": "#1F2224",
    "base": "#3A6878",
    "arboles": "#1A4A4F",
    "potenciacion": "#9A6A8C",
    "ganador": "#A8D8CC",
    "alerta": "#D4A8B4",
    "neutro": "#1E3130",
    "apoyo": "#4A88A4",
    "malva": "#B08AAE",
    "teal": "#3D7A8A",
    "blanco": "#FFFFFF",
}

POR_MODELO = {
    "Arbol": COLORES["arboles"],
    "Bosque": COLORES["teal"],
    "LightGBM": COLORES["potenciacion"],
    "XGBoost": COLORES["malva"],
    "CatBoost": COLORES["apoyo"],
}


def aplicar():
    """Fija la paleta en matplotlib para toda la sesion."""
    plt.rcParams["figure.facecolor"] = COLORES["blanco"]
    plt.rcParams["axes.facecolor"] = COLORES["fondo"]
    plt.rcParams["axes.edgecolor"] = COLORES["rejilla"]
    plt.rcParams["axes.labelcolor"] = COLORES["texto"]
    plt.rcParams["text.color"] = COLORES["texto"]
    plt.rcParams["xtick.color"] = COLORES["texto"]
    plt.rcParams["ytick.color"] = COLORES["texto"]
    plt.rcParams["grid.color"] = COLORES["rejilla"]
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    return COLORES
