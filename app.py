# =============================================================================
# app.py  –  Analizador Interactivo de Aceros al Carbono
# Ejecutar con:  streamlit run app.py
# Requiere:      steels.csv en el mismo directorio
# =============================================================================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analizador de Aceros al Carbono",
    page_icon="⚙️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FUNCIÓN DE TRUNCAMIENTO A 4 DECIMALES (Restricción de Ingeniería)
# ─────────────────────────────────────────────────────────────────────────────
def truncar4(valor: float) -> float:
    """Truncamiento estricto a 4 decimales sin redondeo."""
    return np.trunc(valor * 10_000) / 10_000


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CARGA Y LIMPIEZA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos(ruta: str = "steels.csv") -> pd.DataFrame:
    """
    Carga steels.csv, aplica limpieza y genera columnas calculadas.
    Todas las transformaciones están documentadas paso a paso.
    """
    # 2.1  Lectura del CSV con codificación UTF-8
    df = pd.read_csv(ruta, sep=",", encoding="utf-8", dtype={"SAE Grade": str})

    # 2.2  Limpieza de la columna Conditions
    #      – Eliminar espacios dobles y aplicar formato Title Case
    df["Conditions"] = (
        df["Conditions"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)   # colapsa espacios múltiples
        .str.strip()
        .str.title()                             # mayúsculas iniciales
    )

    # 2.3  Conversión de propiedades mecánicas a valores numéricos
    #      (algunas celdas pueden contener texto o errores de OCR)
    for col in ["UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2.4  Cálculo de Carbono_Promedio
    #      Si C (Min) es NaN, se usa únicamente C (Max)
    df["C (Min)"]  = pd.to_numeric(df["C (Min)"],  errors="coerce")
    df["C (Max)"]  = pd.to_numeric(df["C (Max)"],  errors="coerce")
    df["Carbono_Promedio"] = np.where(
        df["C (Min)"].isna(),
        df["C (Max)"],
        (df["C (Min)"] + df["C (Max)"]) / 2,
    )

    # 2.5  Eliminación de filas sin UTS ni Dureza (no graficables)
    df = df.dropna(subset=["UTS (MPa)", "Hardness (HB)"]).reset_index(drop=True)

    # 2.6  Agrupación de tratamientos térmicos en categorías generales
    df["Tratamiento"] = df["Conditions"].apply(clasificar_tratamiento)

    return df


def clasificar_tratamiento(condicion: str) -> str:
    """
    Clasifica la descripción detallada de un tratamiento en una categoría
    general para facilitar comparaciones en los gráficos.
    Orden de prioridad: Quenched & Tempered > combinados > simples.
    """
    c = condicion.lower()

    # Tratamientos combinados con temple y revenido
    if ("quench" in c or "oil quench" in c or "water quench" in c) and "temper" in c:
        return "Quenched & Tempered"

    # Combinaciones dobles (el tratamiento final define la categoría)
    if "normalized" in c and "cold draw" in c:
        return "Normalized + Cold Drawn"
    if "anneal" in c and "cold draw" in c:
        return "Annealed + Cold Drawn"
    if "spheroidiz" in c:
        return "Spheroidized Annealed"

    # Tratamientos simples
    if "normalized" in c:
        return "Normalized"
    if "anneal" in c:
        return "Annealed"
    if "cold draw" in c or "cold orawn" in c:   # tolera error tipográfico del dataset
        return "Cold Drawn"
    if "hot roll" in c or "hot rott" in c:      # tolera error tipográfico del dataset
        return "Hot Rolled"

    return "Other"


# ─── Carga efectiva ───────────────────────────────────────────────────────────
try:
    aceros = cargar_datos("steels.csv")
except FileNotFoundError:
    st.error("❌  No se encontró **steels.csv**. Colócalo en el mismo directorio que app.py.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BARRA LATERAL  –  Controles de filtrado
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Warning.svg/240px-Warning.svg.png",
        width=40,
    )
    st.title("⚙️ Panel de Control")
    st.markdown("---")

    # 3.1  Slider: rango de carbono promedio
    c_min_global = float(aceros["Carbono_Promedio"].min())
    c_max_global = float(aceros["Carbono_Promedio"].max())
    rango_carbono = st.slider(
        "🔬 Rango de Carbono Promedio (%C)",
        min_value=round(c_min_global, 3),
        max_value=round(c_max_global, 3),
        value=(round(c_min_global, 3), round(c_max_global, 3)),
        step=0.005,
        help="Filtra los aceros según su contenido de carbono promedio.",
    )

    # 3.2  Multiselect: tratamiento térmico
    tratamientos_disponibles = sorted(aceros["Tratamiento"].unique())
    tratamientos_sel = st.multiselect(
        "🔥 Tratamiento Térmico",
        options=tratamientos_disponibles,
        default=tratamientos_disponibles,
        help="Selecciona uno o más tratamientos para filtrar el dataset.",
    )

    # 3.3  Selectbox: propiedad en eje Y
    propiedad_y = st.selectbox(
        "📊 Propiedad mecánica (eje Y)",
        options=["UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"],
        index=0,
        help="Elige la propiedad que aparecerá en el eje Y del gráfico de dispersión.",
    )

    st.markdown("---")
    st.caption("Datos: AISI/SAE Carbon Steels · Fuente: steels.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FILTRADO DEL DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
mascara = (
    aceros["Carbono_Promedio"].between(rango_carbono[0], rango_carbono[1])
    & aceros["Tratamiento"].isin(tratamientos_sel if tratamientos_sel else tratamientos_disponibles)
)
df_filtrado = aceros[mascara].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚙️ Analizador Interactivo de Aceros al Carbono")
st.markdown(
    """
    Base de datos **AISI/SAE** de aceros al carbono – propiedades mecánicas y composición química.  
    Usa los controles de la barra lateral para filtrar y explorar el dataset de forma dinámica.
    """
)
st.markdown("---")

# ─── KPIs rápidos ─────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Aceros filtrados",  f"{len(df_filtrado)}")
k2.metric("UTS promedio (MPa)", f"{truncar4(df_filtrado['UTS (MPa)'].mean()):.4f}" if len(df_filtrado) else "–")
k3.metric("Dureza promedio (HB)", f"{truncar4(df_filtrado['Hardness (HB)'].mean()):.4f}" if len(df_filtrado) else "–")
k4.metric("%C promedio", f"{truncar4(df_filtrado['Carbono_Promedio'].mean()):.4f}" if len(df_filtrado) else "–")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SECCIÓN 1  –  GRÁFICO DE DISPERSIÓN INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────
st.subheader(f"📈  {propiedad_y} vs. Contenido de Carbono (%C)")

st.info(
    """
    **Cómo interpretar este gráfico:**  
    Cada punto representa un acero SAE/AISI bajo un tratamiento térmico específico.  
    Al aumentar el %C las propiedades de resistencia (UTS, YS, Dureza) tienden a crecer, 
    mientras que la ductilidad (Elongación) disminuye. Pasa el cursor sobre un punto para 
    ver el grado SAE y los valores exactos.
    """
)

if df_filtrado.empty:
    st.warning("⚠️  Sin datos para los filtros actuales. Amplía el rango o selecciona más tratamientos.")
else:
    fig_scatter = px.scatter(
        df_filtrado.dropna(subset=["Carbono_Promedio", propiedad_y]),
        x="Carbono_Promedio",
        y=propiedad_y,
        color="Tratamiento",
        symbol="Tratamiento",
        hover_name="SAE Grade",
        hover_data={
            "SAE Grade": True,
            propiedad_y: True,
            "Carbono_Promedio": ":.4f",
            "Tratamiento": True,
            "Conditions": True,
        },
        title=f"<b>{propiedad_y}</b> en función del <b>%C promedio</b> por Tratamiento Térmico",
        labels={
            "Carbono_Promedio": "%C Promedio",
            propiedad_y: propiedad_y,
            "Tratamiento": "Tratamiento Térmico",
        },
        template="plotly_white",
        height=520,
    )
    fig_scatter.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
    fig_scatter.update_layout(
        legend=dict(title="Tratamiento", orientation="v", x=1.01, y=1),
        font=dict(family="Arial", size=13),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SECCIÓN 2  –  GRÁFICO COMBINADO (TODAS LAS PROPIEDADES vs %C)
#     Escala normalizada: Elongación × 10 para comparar en la misma curva.
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊  Comparativa de Propiedades Mecánicas vs. %C  (curvas superpuestas)")

st.info(
    """
    **Cómo interpretar este gráfico:**  
    Las cuatro propiedades se grafican juntas usando sus valores reales, excepto la 
    **Elongación (%) que se multiplica × 10** para llevarla a una escala comparable con 
    UTS, YS y Dureza. Las tendencias relativas entre propiedades son válidas; 
    los valores absolutos de la elongación deben dividirse entre 10.
    """
)

if not df_filtrado.empty:
    # Preparar DataFrame «largo» para trazado múltiple
    df_curvas = df_filtrado.dropna(subset=["Carbono_Promedio"]).copy()
    df_curvas["Elongation_x10"] = df_curvas["Elongation (%)"] * 10  # escala comparable

    propiedades_curvas = {
        "UTS (MPa)":         ("UTS (MPa)",       "#E63946"),
        "YS (MPa)":          ("YS (MPa)",         "#457B9D"),
        "Hardness (HB)":     ("Hardness (HB)",    "#2A9D8F"),
        "Elongation ×10 (%)":("Elongation_x10",   "#F4A261"),
    }

    fig_multi = go.Figure()
    for nombre_display, (col, color) in propiedades_curvas.items():
        df_tmp = df_curvas[["Carbono_Promedio", col, "SAE Grade", "Tratamiento"]].dropna()
        df_tmp = df_tmp.sort_values("Carbono_Promedio")
        fig_multi.add_trace(
            go.Scatter(
                x=df_tmp["Carbono_Promedio"],
                y=df_tmp[col],
                mode="markers+lines",
                name=nombre_display,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color, opacity=0.8),
                hovertemplate=(
                    f"<b>{nombre_display}</b><br>"
                    "%C Promedio: %{x:.4f}<br>"
                    f"Valor: %{{y}}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig_multi.update_layout(
        title="<b>Propiedades Mecánicas vs. %C</b>  (Elongación escalada ×10)",
        xaxis_title="%C Promedio",
        yaxis_title="Valor de propiedad (MPa · HB · %×10)",
        template="plotly_white",
        height=520,
        legend=dict(title="Propiedad", orientation="v", x=1.01, y=1),
        font=dict(family="Arial", size=13),
    )
    st.plotly_chart(fig_multi, use_container_width=True)
else:
    st.warning("⚠️  Sin datos suficientes para trazar las curvas.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SECCIÓN 3  –  GRÁFICO DE RADAR (Comparativa de Ashby entre aceros)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🕸️  Diagrama de Radar  –  Comparativa tipo Ashby entre aceros")

st.info(
    """
    **Cómo interpretar el diagrama de radar:**  
    Cada eje representa una propiedad mecánica normalizada al máximo del dataset filtrado.  
    Un área mayor indica un acero con mejor desempeño global. Compara directamente 
    la "huella" de cada acero para identificar cuál equilibra mejor todas las propiedades.
    """
)

# Selección de aceros para comparar
grades_disponibles = sorted(aceros["SAE Grade"].unique())
grades_sel = st.multiselect(
    "🔩 Selecciona 2 o 3 grados SAE para comparar",
    options=grades_disponibles,
    default=grades_disponibles[:3],
    max_selections=3,
    help="Elige exactamente 2 o 3 grados SAE. El radar mostrará sus propiedades normalizadas.",
)

if len(grades_sel) < 2:
    st.warning("⚠️  Selecciona al menos **2 grados SAE** para activar el radar.")
else:
    propiedades_radar = ["UTS (MPa)", "YS (MPa)", "Hardness (HB)", "Elongation (%)"]

    # Calcular el promedio por grado SAE (sin filtrar por tratamiento aquí)
    df_radar = (
        aceros[aceros["SAE Grade"].isin(grades_sel)]
        .groupby("SAE Grade")[propiedades_radar]
        .mean()
        .reset_index()
    )

    # Normalización min-max (0–1) respecto al dataset completo
    for prop in propiedades_radar:
        global_max = aceros[prop].max()
        global_min = aceros[prop].min()
        rango = global_max - global_min if global_max != global_min else 1
        df_radar[prop + "_norm"] = (df_radar[prop] - global_min) / rango

    categorias = propiedades_radar + [propiedades_radar[0]]  # cierra el polígono
    COLORES_RADAR = ["#E63946", "#457B9D", "#2A9D8F"]

    fig_radar = go.Figure()
    for i, (_, fila) in enumerate(df_radar.iterrows()):
        valores = [fila[p + "_norm"] for p in propiedades_radar]
        valores += [valores[0]]  # cierra el polígono
        fig_radar.add_trace(
            go.Scatterpolar(
                r=valores,
                theta=categorias,
                fill="toself",
                name=f"SAE {fila['SAE Grade']}",
                line=dict(color=COLORES_RADAR[i % len(COLORES_RADAR)], width=2),
                opacity=0.6,
            )
        )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        title="<b>Comparativa de Propiedades Mecánicas</b> (valores normalizados 0–1)",
        template="plotly_white",
        height=520,
        legend=dict(title="Grado SAE"),
        font=dict(family="Arial", size=13),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Tabla de valores reales con truncamiento a 4 decimales
    st.markdown("#### Valores promedio reales por grado SAE")
    tabla_display = df_radar[["SAE Grade"] + propiedades_radar].copy()
    for col in propiedades_radar:
        tabla_display[col] = tabla_display[col].apply(
            lambda v: truncar4(v) if pd.notna(v) else v
        )
    st.dataframe(tabla_display.set_index("SAE Grade"), use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  SECCIÓN 4  –  TABLA DE DATOS FILTRADOS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋  Dataset Filtrado")
cols_mostrar = ["SAE Grade", "Tratamiento", "Carbono_Promedio",
                "UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"]
st.dataframe(
    df_filtrado[cols_mostrar].reset_index(drop=True),
    use_container_width=True,
    height=300,
)
st.caption(f"Filas mostradas: {len(df_filtrado)}  |  Total en dataset: {len(aceros)}")

# ─────────────────────────────────────────────────────────────────────────────
# FIN
# ─────────────────────────────────────────────────────────────────────────────
