# IMPORTACIÓN DE LIBRERÍAS PARA PREPROCESAMIENTO, ANÁLISIS, VISUALIZACIÓN Y CREACIÓN DEL DASHBOARD

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px  
import plotly.graph_objects as go
from scipy.stats import linregress
import dash
from dash import Dash, dcc, html, Input, Output, State
import pandasql as ps
from dash import dash_table
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import multiprocessing as mp   

# CARGA DE DATOS
df = pd.read_excel('owid-energy-data.xlsx')  # DataFrame principal con los datos energéticos
df_metadatos = pd.read_excel('owid-energy-data.xlsx', sheet_name=1)  # DataFrame con los metadatos: definiciones, unidades y fuentes de datos
world = gpd.read_file("ne_110m_admin_0_countries.geojson")  # GeoDataFrame con los polígonos de los países
world = world.rename(columns={"POP_EST": "value", "NAME": "name"}) 

#PREPROCESADO DE DATOS

#Añado pib de la UE (27 miembros)
iso_ue27 = [
    "DEU", "AUT", "BEL", "BGR", "CZE", "CYP", "HRV", "DNK", "SVK",
    "SVN", "ESP", "EST", "FIN", "FRA", "GRC", "HUN", "IRL", "ITA",
    "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SWE"
]# Lista de códigos ISO de los países de la Unión Europea (27)

# Filtrar los países de la UE por código ISO
df_ue27 = df[df["iso_code"].isin(iso_ue27)]

# Sumar el PIB por año EXCLUYENDO 2023
pib_ue27 = df_ue27[df_ue27["año"] != 2023].groupby("año")["pib"].sum().reset_index()

# Añadir columna 'país' para fusionar con 'Unión Europea(27)'
pib_ue27["país"] = "Unión Europea(27)"

# Fusiona df con pib_ue27 (PIB agregado de UE27 por año) usando un left join en 'país' y 'año'. 
# 'pib_y' contiene el PIB de pib_ue27 para filas donde país="Unión Europea(27)" y años coincidentes, y NaN en el resto. 
# combine_first prioriza los valores no nulos de pib_y (UE27) sobre los de df["pib"], actualizando el PIB de la UE27 mientras preserva los valores originales para otros países.
df["pib"] = df.merge(pib_ue27, on=["país", "año"], how="left")["pib_y"].combine_first(df["pib"])

# Asegurar que el PIB de "Unión Europea(27)" en 2023 sea NaN porque no hay datos en algunos de los 27 países
df.loc[(df["país"] == "Unión Europea(27)") & (df["año"] == 2023), "pib"] = np.nan

# Creación de nuevas variables derivadas en el DataFrame principal

## PIB per cápita
df["pib per capita"] = df["pib"] / df["población"]
df["pib per capita"] = df["pib per capita"].round(2)

## Intensidad energética (energía consumida por unidad de PIB)
df["intensidad_energetica"] = df["primaria_energía_consumo"] / df["pib"] * 10**9
df["intensidad_energetica"] = df["intensidad_energetica"].round(2)

## Emisiones de gases de efecto invernadero per cápita
df["GEI_per_capita"] = df["efecto_invernadero_gas_emisiones"] / df["población"] * 10**6
df["GEI_per_capita"] = df["GEI_per_capita"].round(2)

# Adición de metadatos para las nuevas variables
nueva_fila_pib_pc = pd.DataFrame({
    "columna": ["pib per capita"],
    "descripción": ["PIB per cápita - Producto Interno Bruto dividido entre la población"],
    "unidades": ["$/persona"],
    "Fuente": ["Derivado de los atributos 'pib' y 'población'"]
})

nueva_fila_intensidad = pd.DataFrame({
    "columna": ["intensidad_energetica"],
    "descripción": ["Intensidad energética - Relación entre el PIB y el consumo de energía primaria"],
    "unidades": ["kWh/$"],
    "Fuente": ["Derivado de los atributos 'primaria_energía_consumo' y 'pib'"]
})

nueva_fila_GEIpc = pd.DataFrame({
    "columna": ["GEI_per_capita"],
    "descripción": ["Emisiones per cápita de gases de efecto invernadero"],
    "unidades": ["toneladas CO2eq/persona"],
    "Fuente": ["Derivado de los atributos 'efecto_invernadero_gas_emisiones' y 'población'"]
})

df_metadatos = pd.concat([df_metadatos, nueva_fila_pib_pc, nueva_fila_intensidad, nueva_fila_GEIpc], ignore_index=True)
df_metadatos = df_metadatos[~df_metadatos["columna"].str.contains("cambio", case=False, na=False)]



# Creación de listas de países y atributos para los menús desplegables (dropdowns) del dashboard
df.loc[df["país"] == "URSS", "iso_code"] = "USR"  #Asigno a la URSS un código ISO para considerarlo como país en el filtrado siguiente
countries = list(set(df[df['iso_code'].notnull()]['país'])) # Lista de países: se seleccionan aquellas filas con código ISO (las filas que no tienen código ISO no pertenecen a países) 
countries.append("Mundo")    # Se añade "Mundo" para permitir la visualización de valores agregados globales en la sección de Series Temporales
countries.append("Unión Europea(27)")
countries.append("Kosovo")   # Se añade manualmente Kosovo para que la lista de países coincida con los del GeoDataFrame

# Lista de códigos ISO de las repúblicas ex-soviéticas
ex_soviet_iso = ["ARM", "AZE", "BLR", "EST", "GEO", "KAZ", "KGZ", "LTU", "LVA", "MDA", "RUS", "TJK", "TKM", "UKR", "UZB"]

# Eliminación de datos de la URSS a partir de 1991 y de ex-sovieticos antes de 1991 (owid presenta datos de estos países por separado antes de la  fecha oficial de disolución de la URSS)
df = df[~((df["país"] == "URSS") & (df["año"] > 1991))]
df = df[~((df["iso_code"].isin(ex_soviet_iso)) & (df["año"] <= 1991))]  

atributos = list(df.columns) # Lista de atributos del DataFrame principal
atributos_num = [atributo for atributo in atributos if df[atributo].dtypes == "float64" and "cambio" not in atributo]# Filtro de atributos numéricos de interés (no interesan los que sean de cambio para los gráficos escogidos
atributos_num_None = ["None"] + atributos_num # Lista de atributos numéricos con opción "None" (para color/tamaño opcional en gráficos de dispersión)
atributos_num_absolutos = [
    atributo for atributo in atributos_num
    if all(x not in atributo for x in ["per", "cambio", "proporción", "importaciones", "carbono_intensidad_elec", "intensidad_energetica"])
] # Atributos numéricos absolutos, excluyendo proporciones, cambios, intensidades y otros que no aplican en un treemap

# Creación de un diccionario de unidades a partir de los metadatos
# Se utilizará en las visualizaciones para mostrar las unidades correspondientes a cada atributo

dict_unidades=dict()

for columna in atributos:
    try: dict_unidades[columna]=df_metadatos[df_metadatos.columna==columna].unidades.values[0]
    except: dict_unidades[columna]=df_metadatos[df_metadatos.columna==columna].unidades.values

# Corrección de códigos ISO para mejorar la coherencia entre el DataFrame principal y el GeoDataFrame

world.ISO_A3[world.name=='Kosovo']='KOS'
world.ISO_A3[world.name=='Somaliland']='SOM'
world.name[world.name=='Somaliland']='Somalia'
df.iso_code[df["país"]=='Kosovo']='KOS'
world.ISO_A3[world.name=='France']='FRA'
df.iso_code[df["país"]=='France']='FRA'
world.ISO_A3[world.name=='Norway']='NOR'
df.iso_code[df["país"]=='Norway']='NOR'

# Diccionario que mapea los códigos ISO a los nombres de países en español (extraídos del DataFrame principal)
iso_to_spanish = dict(zip(df.iso_code, df['país']))

# Actualización de los nombres de los países en el GeoDataFrame para mostrarlos en español
# Se usa la columna ISO_A3 como clave de mapeo y se conserva el nombre original si no se encuentra equivalencia
world['name'] = world['ISO_A3'].map(iso_to_spanish).fillna(world['name'])

# Creación de un DataFrame para series temporales, que incluye tanto países como el agregado mundial
df_conmundo = df[~df.iso_code.isnull() | (df["país"] == "Mundo")| (df["país"] == "Unión Europea(27)")]

# Eliminación de filas sin código ISO (es decir, que no representan países) del DataFrame principal
df = df[~df.iso_code.isnull()]

# Se añade la columna 'continente' al DataFrame principal usando los códigos ISO como clave. Esta columna 'continente' se usará para agregar por continentes en los treemap

## Crear un diccionario {ISO_A3: CONTINENT} a partir del GeoDataFrame
dict_countries = world.set_index('ISO_A3')['CONTINENT'].to_dict()

## Mapear los códigos ISO del DataFrame principal al continente correspondiente
df['continente'] = df['iso_code'].map(dict_countries)

## Traducir los nombres de los continentes al español
traducciones_continentes = {
    'Africa': 'África',
    'Asia': 'Asia',
    'Europe': 'Europa',
    'North America': 'América del Norte',
    'South America': 'América del Sur',
    'Oceania': 'Oceanía',
    'Antarctica': 'Antártida'
}

df['continente'] = df['continente'].map(traducciones_continentes)




# DEFINICIÓN DE LA ESTRUCTURA HTML DEL DASHBOARD 

# Inicialización de la aplicación Dash con hoja de estilo externa de Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])   #Estilo BOOTSTRAP: https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/

# Inicialización del layout 
app.layout = dbc.Container([  #sistema de rejilla de Bootstrap
    # Encabezado con logo y título
    html.Div([
    # Primera fila: logo + título
    dbc.Row([
        dbc.Col(html.Img(src="http://materials.cv.uoc.edu/cdocent/common/img/logo-uoc.png",
                         alt="Logo UOC",
                         style={'margin': '20px 0px 0px', 'maxWidth': '100%'}),
                width=2),
        dbc.Col(html.H1("Dashboard con visualizaciones interactivas para el análisis de la evolución del consumo energético global",
                        style={'margin': '0px 0px 0px', 'fontSize': '40px', 'fontWeight': 'bold', 'textAlign': 'center'}),
                width=10)
    ], style={'color': '#002a77',  'paddingBottom': '20px'}),

    # Segunda fila: asignatura y autor
    dbc.Row([
        dbc.Col([
            html.P("M2.882 - Trabajo Fin de Máster", style={'margin': '0px 20px 0px'}),
            html.P("Autor: Jonay González Guerra", style={'margin': '0px 20px 0px'})
        ], width=12)  # 12 es el ancho máximo de columnas en Dash Bootstrap Components
    ], style={'color': '#002a77', 'fontSize': '15px', 'padding': '0px'}),
    ],style={'backgroundColor': '#ADD8E6'}),
    html.Br(),

    # Párrafo introductorio con enlace a la fuente de datos
    dbc.Row([
        dbc.Col(html.P([
            "El uso de la energía es un factor determinante en el desarrollo económico y en la sostenibilidad ambiental. "
            "A medida que los países buscan equilibrar crecimiento económico, eficiencia energética y reducción de emisiones "
            "de carbono, disponer de herramientas interactivas que faciliten el análisis de datos se vuelve cada vez más relevante. "
            "Este Trabajo Fin de Máster presenta un dashboard con visualizaciones interactivas que permite explorar la relación "
            "entre consumo energético a partir de distintas fuentes, datos demográficos y económicos, así como de emisiones de "
            "gases de efecto invernadero en los distintos países del mundo a lo largo de las últimas décadas. "
            "Los datos empleados para la visualización han sido obtenidos de la web: ",
            html.A("Energy-Our World in Data", href="https://ourworldindata.org/energy")
        ], style={'color': '#002a77','textAlign': 'justify'}), width=12)
    ]),

    #html.Br(),

    # Sección: Distribución mundial
    html.Div([
        html.Button([
            html.Div("Distribución mundial", style={'flex': 1, 'textAlign': 'left'}), # 'flex: 1' hace que este div crezca proporcionalmente dentro de un contenedor con display: flex.  útil para distribuir elementos horizontalmente de forma equilibrada
            html.Span(" \u25BC", style={'fontSize': '24px', 'marginLeft': 'auto'})  #flecha hacia abajo a la derecha del Button
        ], id="toggle-geo", n_clicks=0, # n_clicks=0 propiedad numérica del html.Button  toggle-geo que inicializa el contador de clics del botón (para callbacks). 
            style={'fontSize': '20px', 'width': '100%', 'height': '80px',
                   'textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77',
                   'border': 'none', 'outline': 'none', 'display': 'flex', 'alignItems': 'center'}),
        html.Div([  #División con el contenido del bloque a mostrar u ocultar
            html.Br(),
            dbc.Row([
                dbc.Col(html.P(
                    "En este apartado puedes observar cómo se distribuye un atributo a tu elección a lo largo del mundo y para un año concreto. "
                    "Para visualizar la distribución mundial de dicho atributo puedes escoger entre 3 tipos de gráficos: Mapamundi, Treemap o un diagrama de barras. "
                    "En el caso de optar por el diagrama de barras, podrás escoger un número de países a mostrar, los cuales aparecerán ordenados según el valor del atributo de manera descendiente.",
                    style={'textAlign': 'justify','color': '#002a77' }
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    dcc.RadioItems(
                        id="radio-geo",
                        options=[
                            {'label': 'Mapamundi', 'value': 'world'},
                            {'label': 'Treemap', 'value': 'treemap'},
                            {'label': 'Diagrama de barras', 'value': 'barras'}
                        ],
                        value='world',
                        labelStyle={'display': 'block', 'margin': '5px', 'color': '#002a77'}
                    ),  #Botones de opciones mutuamente excluyentes para seleccionar un tipo de gráfico. Por defecto se muestra Mapamundi.
                    html.P("Atributo:"),
                    dcc.Dropdown(id="dropdown-geo", options=atributos_num, value='población', clearable=False,
                                 style={'color': '#002a77'}), #Dropdown para escoger el atributo numérico a mostrar en los gráficos de la sección.
                    html.P("Año:"),
                    dcc.Slider(df.año.min(), df.año.max(), step=1, value=2010, marks=None, id='slider-geo',
                               tooltip={"placement": "bottom", "always_visible": True}),   #Deslizador para seleccionar año para el que visibilizar los gráficos de la sección.
                    html.Div([
                        html.P("Número de países a mostrar:"),
                        dcc.Slider(5, 30, step=1, value=20, marks=None, id='slider-top-n',
                                   tooltip={"placement": "bottom", "always_visible": True})  #Deslizador para seleccionar el número de países a mostrar en el diagrama de barras. 
                    ], id="slider-container", style={'display': 'none'})  #El contenedor de este deslizador solo aparecerá si se selecciona el diagrama de barras. Gestionado mediante un callback.
                ], width=2, style={'padding': '20px'}),  #Columna con controles estrecha a la izquierda. Relleno (espacio interior) de 20 píxeles 
                dbc.Col(dcc.Graph(
                    id="graph-geo",
                    style={"width": "100%", "height": "80vh", "maxHeight": "800px", "minHeight": "400px"},   # El gráfico ocupa el 100% del ancho y el 80% de la altura de la ventana (80vh), con una altura máxima de 800px y mínima de 400px.
                    config={"responsive": True}   # Asegura que el gráfico se redimensione proporcionalmente según el tamaño disponible, manteniendo una apariencia adecuada en pantallas grandes y pequeñas.
                ), width=10, style={'padding': '0px'})   #Columna con gráficos más ancha, a la derecha.
            ])
        ], id="geo-content", style={'display': 'none'})  #'display': 'none' para que el bloque esté inicialmente oculto
    ], style={'border': 'none', 'padding': '10px', 'margin': '10px'}),  # Espacio interno (padding) y externo (margin) de 10px, sin borde (border: 'none')

    # Sección: Clustering
    html.Div([
        html.Button([
            html.Div("Clustering", style={'flex': 1, 'textAlign': 'left'}),
            html.Span(" \u25BC", style={'fontSize': '24px', 'marginLeft': 'auto'})
        ], id="toggle-clusters", n_clicks=0,
            style={'fontSize': '20px', 'width': '100%', 'height': '80px',
                   'textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77',
                   'border': 'none', 'outline': 'none', 'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col(html.P(
                    "En este apartado los países se agrupan en clusters según los atributos seleccionados por el usuario mediante el algoritmo K-Means, "
                    "configurable en número de grupos, año de análisis y variables consideradas. Si se seleccionan dos o más atributos se aplica una reducción "
                    "de dimensiones con PCA y se muestra un gráfico de dispersión, mientras que con un solo atributo se presenta un histograma "
                    "con la distribución por grupo. En ambos casos se incluye un mapa mundial donde cada país se colorea según el grupo al que pertenece.",
                    style={'textAlign': 'justify','color': '#002a77'}
                ), width=12)
            ]),
            dbc.Row([  #Primera fila con controles en columna de la izquierda y gráfico de clusters diferenciados por color a la derecha, en un diagrama de dispersión en el plano de las 2 componentes principales, o bien un histograma si solo se elige un atributo
                dbc.Col([
                    html.Br(),
                    html.P("Atributos:"),
                    dcc.Dropdown(
                        id="dropdown-atributos-cluster",
                        options=[{"label": a, "value": a} for a in atributos_num],  #equivalente a options=atributos_num, pero así permite cambiar etiquetas y filtrar lista inicial.
                        multi=True,  #dropdown con opciones múltiples.
                        value=["fósil_proporción_energía", "energía_per_capita", "GEI_per_capita"], #opciones por defecto
                        clearable=False,
                        style={'color': '#002a77'}
                    ),
                    html.P("Año:"),
                    dcc.Slider(df.año.min(), df.año.max(), step=1, value=2010, marks=None, id='slider-año-clusters',
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.P("Número de clusters:"),
                    dcc.Slider(1, 10, step=1, value=3, marks=None, id='slider-num-clusters',  #Deslizador para seleccionar entre 1 y 10 clusters.
                               tooltip={"placement": "bottom", "always_visible": True})
                ], width=2, style={'padding': '20px','color': '#002a77'}),
                dbc.Col(dcc.Graph(
                    id="graph-clusters-1",
                    style={"width": "100%", "height": "100%"}, config={"responsive": True}
                ), width=10, style={'padding': '0px'})
            ]),
            dbc.Row([  #Segunda fila con Mapamundi con mismos colores de clusters, ocupa todo el ancho.
                dbc.Col(dcc.Graph(
                    id="graph-clusters-2",
                    style={"width": "100%", "height": "80vh", "maxHeight": "800px", "minHeight": "400px"},
                    config={"responsive": True}
                ), width=12)
            ], style={'padding': '0px', 'margin': '0 auto'})
        ], id="clusters-content", style={'display': 'none'})
    ], style={'border': 'none', 'padding': '10px', 'margin': '10px'}),

    # Sección: Diagrama de dispersión
    html.Div([
        html.Button([
            html.Div("Diagrama de dispersión", style={'flex': 1, 'textAlign': 'left'}),
            html.Span(" \u25BC", style={'fontSize': '24px', 'marginLeft': 'auto'})
        ], id="toggle-dispersion", n_clicks=0,
            style={'fontSize': '20px', 'width': '100%', 'height': '80px',
                   'textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77',
                   'border': 'none', 'outline': 'none', 'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col(html.P(
                    "En este apartado se muestra un diagrama de dispersión bidimensional para dos atributos y un año a tu elección, así como la recta de regresión "
                    "lineal correspondiente con datos estadísticos relevantes. La visualización también permite escoger dos atributos adicionales que "
                    "se observarán en el tamaño de los puntos y su color.",
                    style={'color': '#002a77','textAlign': 'justify'}
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.P("Atributo eje X:"),
                    dcc.Dropdown(id="dropdown21", options=atributos_num, value='pib per capita', clearable=False, style={'color': '#002a77'}),
                    html.P("Atributo eje Y:"),
                    dcc.Dropdown(id="dropdown22", options=atributos_num, value='energía_per_capita', clearable=False, style={'color': '#002a77'}),
                    html.P("Atributo tamaño:"),
                    dcc.Dropdown(id="dropdown23", options=atributos_num_None, value='None', clearable=False, style={'color': '#002a77'}),
                    html.P("Atributo color:"),
                    dcc.Dropdown(id="dropdown24", options=atributos_num_None, value='None', clearable=False, style={'color': '#002a77'}),
                    html.P("Año:"),
                    dcc.Slider(df.año.min(), df.año.max(), step=1, value=2010, marks=None, id='my-slider2',
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.P("Escala de ejes:"),
                    dcc.RadioItems(
                        id="scale-type",
                        options=[
                            {"label": "Escala lineal", "value": "linear"},
                            {"label": "Escala logarítmica", "value": "log"}
                        ], #Botones de opciones mutuamente excluyentes para seleccionar escala.
                        value="linear",
                        labelStyle={'display': 'block', 'margin': '5px', 'color': '#002a77'}
                    )
                ], width=2, style={'color': '#002a77','padding': '20px'}),
                dbc.Col(dcc.Graph(
                    id="graph2",
                    style={"width": "100%", "height": "100%"}
                ), width=10, style={'padding': '20px'})
            ])
        ], id="dispersion-content", style={'display': 'none'})
    ], style={'color': '#002a77','border': 'none', 'padding': '10px', 'margin': '10px'}),

    # Sección: Series temporales
    html.Div([
        html.Button([
            html.Div("Series Temporales", style={'flex': 1, 'textAlign': 'left'}),
            html.Span(" \u25BC", style={'fontSize': '24px', 'marginLeft': 'auto'})
        ], id="toggle-ts", n_clicks=0,
            style={'fontSize': '20px', 'width': '100%', 'height': '80px',
                   'textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77',
                   'border': 'none', 'outline': 'none', 'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col(html.P(
                    "En este apartado se puede observar las series temporales correspondientes a un atributo y una serie de países a elección del usuario. "
                    "Puedes escoger entre visualizar el valor del atributo o su tasa de variación porcentual anual.",
                    style={'textAlign': 'justify'}
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.P("Atributo a mostrar:"),
                    dcc.Dropdown(id="dropdown11", options=atributos_num, value='población', clearable=False, style={'color': '#002a77'}),
                    html.P("Países o regiones:"),
                    dcc.Dropdown(
                        id="dropdown-countries",
                        options=[{"label": c, "value": c} for c in countries],
                        value=['España'],
                        multi=True,
                        clearable=False,
                        style={'color': '#002a77'}
                    ),  #Desplegable de opciones múltiples para seleccionar países a mostrar en el gráfico de series temporales.
                    html.Br(),
                    dcc.RadioItems(
                        id="radio-ts",
                        options=[
                            {'label': 'Atributo', 'value': 'ts'},
                            {'label': 'Tasa de Variación Anual', 'value': 'rate'}
                        ], #Botones de opciones mutuamente excluyentes para seleccionar tipo de representación.
                        value='ts',
                        labelStyle={'display': 'block', 'margin': '5px', 'color': '#002a77'}
                    )
                ], width=2, style={'padding': '20px'}),
                dbc.Col(dcc.Graph(
                    id="graph1",
                    style={"width": "100%", "height": "100%"}
                ), width=10, style={'padding': '20px'})
            ])
        ], id="ts-content", style={'display': 'none'})
    ], style={'color': '#002a77','border': 'none', 'padding': '10px', 'margin': '10px'}),

    # Sección: Consultas SQL
    html.Div([
        html.Button([
            html.Div("Consultas SQL", style={'flex': 1, 'textAlign': 'left'}),
            html.Span(" \u25BC", style={'fontSize': '24px', 'marginLeft': 'auto'})
        ], id="toggle-SQL", n_clicks=0,
            style={'fontSize': '20px', 'width': '100%', 'height': '80px',
                   'textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77',
                   'border': 'none', 'outline': 'none', 'display': 'flex', 'alignItems': 'center'}),
        dcc.Store(id="query-result-store", data={}),  # Estado que almacena los resultados de la consulta, para poderlos descargar si el usuario lo desea
        html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col(html.P(
                    'En este apartado puedes realizar consultas SQL sobre el conjunto de datos. Para ello ten en cuenta que el nombre de la tabla es "tabla".',
                    style={'textAlign': 'justify'}
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.P('Ejemplo de consulta:', style={'textAlign': 'justify'}), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.P(
                    """SELECT año, país, población, pib AS PIB, "pib per capita" AS "PIB per capita" FROM tabla WHERE país = "España" AND año BETWEEN 1980 AND 2022;""",
                    style={'textAlign': 'justify'}
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.P(
                    'Puedes descargar los resultados de la consulta en formato CSV si lo deseas.',
                    style={'textAlign': 'justify'}
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H4("Escribe tu consulta SQL a continuación:"),
                    dcc.Textarea(id='sql-input', style={'width': '100%', 'height': '100px', 'fontFamily': 'monospace'}),   #Recuadro donde introducir la consulta SQL. 
                    html.Button('Ejecutar consulta', id='execute-query', n_clicks=0),  #Botón para lanzar la consulta SQL
                    html.Button("Descargar CSV", id="download-csv", n_clicks=0, style={'marginLeft': '10px'}),  #Botón para descargar la consulta, guardada en el estado "query-result-store", en formato CSV.
                    dcc.Download(id="download-link"),  #Descarga los datos si el usuario ha pulsado en "Descargar CSV"
                    html.Div(id='query-results', style={'marginTop': '20px'}) #Muestra los resultados del query
                ], style={'padding': '20px'}), width=12)
            ])
        ], id="SQL-content", style={'display': 'none'})
    ], style={'color': '#002a77','border': 'none', 'padding': '10px', 'margin': '10px'}),

    # Sección: Metadatos
    html.Div([
        html.Button([
            html.Div("Metadatos", style={'flex': 1, 'textAlign': 'left'}),
            html.Span(" \u25BC", style={'fontSize': '24px', 'marginLeft': 'auto'})
        ], id="toggle-meta", n_clicks=0,
            style={'fontSize': '20px', 'width': '100%', 'height': '80px',
                   'textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77',
                   'border': 'none', 'outline': 'none', 'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col(html.P(
                    'A continuación se muestra una tabla con la descripción de cada uno de los atributos representables en el dashboard, '
                    'las unidades correspondientes y las fuentes de donde han sido obtenidos los datos:',
                    style={'textAlign': 'justify'}
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='meta-results', style={'marginTop': '20px'}), width=12)   #Muestra la tabla de Metadatos de la misma forma que se mostraban los resultados del query en la sección de "Consultas SQL"
            ])
        ], id="meta-content", style={'display': 'none'})
    ], style={'color': '#002a77','border': 'none', 'padding': '10px', 'margin': '10px'}),

    html.Br(),

    # Pie de página (footer) con el logo de la UOC
    dbc.Row(
        dbc.Col(html.Img(src="http://materials.cv.uoc.edu/cdocent/common/img/logo-uoc-bottom.png",
                         alt="Logo UOC",
                         style={'margin': '0 auto', 'display': 'block', 'maxWidth': '95%'}),
                width=12),
        style={'backgroundColor': '#002a77', 'padding': '35px 0px', 'marginTop': '25px', 'height': '140px'}
    )

], fluid=True, style={'width': '100%', 'margin': '0 auto'})   #El parámetro fluid=True en dbc.Container hace que el contenedor se ajuste automáticamente al 100% del ancho de la pantalla disponible, haciéndolo responsive.






# CALLBACKS Y FUNCIONES CON GRÁFICOS Y CONSOLA SQL

# Callback para mostrar u ocultar la sección "Distribución mundial" al hacer clic en el botón
@app.callback(
    Output("geo-content", "style"), # Cambia el estilo (visible/oculto) del contenido
    Input("toggle-geo", "n_clicks"),# Entrada: número de clics en el botón 
)
def toggle_geo(n_clicks):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'} 
    return {'display': 'none'}  # Si es par -->ocultar contenido
    
# Callback que genera la figura de distribución mundial (mapamundi, treemap o barras)
# según el tipo de visualización, el atributo seleccionado, el año y el número de países a mostrar (en el diagrama de barras)

@app.callback(
    Output("graph-geo", "figure"), # Salida: gráfico que se muestra en la sección
    Input("radio-geo", "value"),   # Entrada: tipo de gráfico seleccionado (world, treemap, barras)
    Input("dropdown-geo", "value"),# Entrada: atributo numérico seleccionado
    Input("slider-geo", "value"),  # Entrada: año seleccionado
    Input("slider-top-n", "value") # Entrada: número de países a mostrar (solo para gráfico de barras)
)    

# Función que genera la figura correspondiente a la sección "Distribución mundial"
# según el tipo de visualización seleccionada, el atributo, el año y el número de países (solo para gráfico de barras)

def actualizar_geo(vista, atributo, año, num_paises):
    # Mapamundi
    if vista == 'world':
        # Crear una serie de Pandas con los valores del atributo por país en ese año, para luego asignársela a la columna value del GeoDataFrame "world"
        nueva_column = pd.Series(data=np.repeat(np.nan, world.shape[0]))
        for i in range(world.shape[0]):
            try:
                nueva_column.iloc[i] = df[(df.año == año) & (df.iso_code == world.iloc[i].ISO_A3)][atributo].values[0]
            except:
                nueva_column.iloc[i] = np.nan
        world.value = nueva_column
        # Si el año es <= 1991, asignar valor y nombre de "URSS" a los países ex-soviéticos
        if año <= 1991:
            try:
                valor_URSS = df[(df.año == año) & (df["país"] == "URSS")][atributo].values[0]
            except:
                valor_URSS = np.nan
            world.loc[world['ISO_A3'].isin(ex_soviet_iso), 'value'] = valor_URSS
            world.loc[world['ISO_A3'].isin(ex_soviet_iso), 'name'] = 'URSS'
        # Capa base para mostrar países sin datos
        world_base = world.copy()
        world_base["value"] = world["value"].fillna(-1).apply(lambda x: "Sin datos" if x == -1 else "Con datos")  # Usa -1 para distinguir sin datos
        world["indice"] = world.index
        world_base["indice"] = world_base.index
        # Se crea la figura de base (gris/blanco según datos) y la figura de datos (escala continua)
        fig = px.choropleth(world_base, geojson=world_base.geometry, locations="indice", 
                             color="value",
                             color_discrete_map={"Sin datos": "lightgray", "Con datos": "white"},
                             hover_name="name",  
                             hover_data={"value": False, "indice": False}, 
                             title=f"Mapa coroplético mundial de {atributo} ({dict_unidades[atributo]})  en el año {año}"                             
                             )
        fig_data = px.choropleth(world, geojson=world.geometry, locations="indice",  
                             color="value", 
                             hover_name="name", 
                             hover_data={"value": True, "indice": False},  
                             color_continuous_scale=px.colors.sequential.Plasma,
                             labels={"value": f"{atributo} ({dict_unidades[atributo]})"},
                             title=f"Mapa coroplético mundial de {atributo} ({dict_unidades[atributo]}) en el año {año}"
                             )
        # Añadir capa de datos sobre el mapa base
        fig.add_trace(fig_data.data[0])  
        # Mejoras estéticas de la figura
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_coloraxes(colorbar_title="valor")
        fig.update_traces(showlegend=False)
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0), 
            coloraxis_colorbar=dict(
                title="valor",
                x=0.975,  
                len=0.75 )
            )  

        # Revertir nombre de países ex-soviéticos a sus nombres originales (para futuras interacciones)
        if año <= 1991:
            world.loc[world['ISO_A3'].isin(ex_soviet_iso), 'name'] = world.loc[world['ISO_A3'].isin(ex_soviet_iso), 'ISO_A3'].map(iso_to_spanish).fillna(world.loc[world['ISO_A3'].isin(ex_soviet_iso), 'name'])
        return fig
    # Treemap            
    elif vista == 'treemap':
        # Verificar si el atributo es adecuado para un treemap (magnitud absoluta y positiva)
        if atributo in atributos_num_absolutos:
            df_año = df[df.año == año].copy()  #filtar los datos para el año de interés
            # Aplicar lógica histórica: URSS hasta 1991, repúblicas ex-soviéticas desde 1991
            if año <= 1991:
                df_año = df_año[~(df_año['iso_code'].isin(ex_soviet_iso) & (df_año['país'] != 'URSS'))]
                df_año.loc[df_año['país'] == 'URSS', 'continente'] = 'Europa'
            elif año > 1991:
                df_año = df_año[df_año['país'] != 'URSS']
            df_año = df_año.dropna(subset=['continente'])
            total_mundial = df_año[atributo].sum()
            # Generar treemap jerárquico: Mundo → Continente → País
            fig = px.treemap(
                df_año, 
                path=[px.Constant("Mundo"), 'continente', 'país'], 
                values=atributo, 
                labels={atributo: f"{atributo} ({dict_unidades[atributo]})"},    #En la etiqueta emergente o tooltip, sustituye el nombre del atributo por: atributo (unidades).
                title=f"Treemap de {atributo} ({dict_unidades[atributo]}) en el año {año}"
            )
            # Personalización de etiquetas y porcentaje sobre el total
            for trace in fig.data:
                customdata = []
                text_values = []
                for value, label in zip(trace.values, trace.labels):
                    percent = (value / total_mundial) * 100 if total_mundial > 0 else None
                    customdata.append([percent])
                    text_values.append(f"{label}<br>{value:,.2f} {dict_unidades[atributo]}<br>{percent:.2f}%")  #<br> para salto de línea
                trace.customdata = np.array(customdata)
                trace.text = text_values  
                trace.texttemplate = "%{text}"    # Muestra el contenido personalizado de 'trace.text' directamente en las celdas del treemap
            # Tooltip personalizado y color raíz
            fig.update_traces(
                root_color='#ADD8E6', #para el recuadro del mundo mismo color que buttons y encabezado
                hovertemplate='<b>%{label}</b><br>' + atributo + f' ({dict_unidades[atributo]}): ' + '%{value} <br>' + 'Porcentaje: %{customdata[0]:.2f}%<extra></extra>' #Etiqueta emergente.  <extra></extra> oculta el rastro por defecto que Plotly añade al final del tooltip https://plotly.com/python/hover-text-and-formatting/
            )
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            return fig
        else: 
           # Si el atributo no es apto para treemap (por ser relativo o contener valores negativos)
           fig = go.Figure()
           fig.add_annotation(
                text="Un treemap no es una visualización apropiada para este tipo de atributo.",
                x=0.5, y=0.5,
                xref="paper", yref="paper", #coordenadas x e y relacionadas con el área del gráfico entero ("paper"), no con los ejes de datos.
                showarrow=False,
                font=dict(size=20)
            )
           fig.update_layout(
                xaxis={'visible': False},
                yaxis={'visible': False},
                plot_bgcolor="#f0f0f0",
                margin=dict(t=50, l=25, r=25, b=25)
            )
           return fig
    # Diagrama de barras       
    else:
        df_año = df[df.año == año].copy()
        # Aplicar lógica histórica: URSS hasta 1991, repúblicas ex-soviéticas desde 1991
        if año <= 1991:
            df_año = df_año[~(df_año['iso_code'].isin(ex_soviet_iso) & (df_año['país'] != 'URSS'))]
            df_año.loc[df_año['país'] == 'URSS', 'continente'] = 'Europa'
        elif año > 1991:
            df_año = df_año[df_año['país'] != 'URSS']
        df_año = df_año.dropna(subset=['continente'])
        df_top = df_año.nlargest(num_paises, atributo)  # se seleccionan los N mayores, los devuelve ordenados de mayor a menor
        df_top["rank"] = range(1, len(df_top) + 1)  # Se crea una nueva columna con el número en el ranking
    
        # Creación de la figura con Plotly-express bar    
        fig = px.bar(
            df_top, 
            x=atributo,  
            y="país",  
            color="país",
            orientation="h",  
            labels={atributo: f"{atributo} ({dict_unidades[atributo]})"},
            hover_data={"país": True, "rank": True, atributo: True},
            title=f"Top {num_paises} países según el atributo {atributo} en el año {año}",
            
        )
        # Mejoras estéticas de la figura
        fig.update_layout(plot_bgcolor="white", showlegend=False)
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_layout(
            height=max(200, 20 * num_paises),  
            plot_bgcolor="white", 
            showlegend=False,
            margin=dict(t=50, l=100, r=30, b=40)  
        )
        return fig

# Callback para mostrar u ocultar el slider de número de países según el tipo de gráfico seleccionado
@app.callback(
    Output('slider-container', 'style'),   #Salida: estilo del contenedor del slider
    Input('radio-geo', 'value')            # Entrada: tipo de gráfico seleccionado (mapa, treemap o barras)
)
def toggle_slider(selected_graph):
    if selected_graph == "barras":         
        return {'display': 'block'}        # Mostrar el slider si se ha seleccionado "Diagrama de barras"
    return {'display': 'none'}             # Ocultar el slider en cualquier otro caso



# Callback para mostrar u ocultar la sección "Clustering" al hacer clic en el botón
@app.callback(
    Output("clusters-content", "style"), # Cambia el estilo (visible/oculto) del contenido
    Input("toggle-clusters", "n_clicks"),# Entrada: número de clics en el botón 
)
def toggle_clusters(n_clicks):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}
    return {'display': 'none'} # Si es par -->ocultar contenido
    
# Callback que solo deja seleccionar 4 atributos como máximo
@app.callback(
    Output("dropdown-atributos-cluster", "value"),
    Input("dropdown-atributos-cluster", "value")
)
def limitar_atributos_seleccionados(valores):
    if valores and len(valores) > 4:
        return valores[:4]  
    return valores
    
# Callback para las figuras de la sección de Clustering
@app.callback(
    Output("graph-clusters-1", "figure"), # Salida: Diagrama de dispersión con clusters
    Output("graph-clusters-2", "figure"),  # Salida: Mapamundi con clusters
    Input("dropdown-atributos-cluster", "value"),# Entrada: atributos numéricos seleccionados para el clustering
    Input("slider-año-clusters", "value"),  # Entrada: año seleccionado
    Input("slider-num-clusters", "value"),  # Entrada: número de clusters seleccionados
)    

# Función que genera las dos figuras correspondientes a la sección "clustering"
def actualizar_clusters(atributos, año, num_clusters):
    atributos = list(atributos)

    df_año = df[df.año == año].copy()
    df_cluster = df_año[["país", "año", "iso_code"] + atributos].dropna() #Filtro de las columnas a utilizar en esta sección
    
    # Si no hay datos suficientes para clustering para los atributos y año escogidos, o si no hay atributos seleccionados, se asigna "Sin datos"
    if df_cluster.empty or len(df_cluster) < num_clusters or len(atributos)==0 :
        world["grupo"] = "Sin datos"
        world["indice"] = world.index

        hover_dict = {"grupo": True, "indice": False, "name": False}
        for col in atributos:
            col_label = f"{col} ({dict_unidades[col]})" if col in dict_unidades else col
            world[col_label] = None
            hover_dict[col_label] = True

        fig1 = go.Figure().update_layout(
            title=f"No hay datos suficientes para agrupar en {num_clusters} clusters en el año {año}"
        )

        fig2 = px.choropleth(
            world,
            geojson=world.geometry,
            locations="indice",
            color="grupo",
            color_discrete_map={"Sin datos": "lightgray"},
            hover_name="name",
            hover_data=hover_dict,

        )
        fig2.update_geos(fitbounds="locations", visible=False)
        fig2.update_coloraxes(showscale=False)
        fig2.update_traces(showlegend=False)
        fig2.update_layout(
            title_text=f"No hay datos suficientes para agrupar en {num_clusters} clusters en el año {año}",
            plot_bgcolor="white"
        )
        
        fig1.update_layout(plot_bgcolor="white")
        fig2.update_layout(plot_bgcolor="white")

        return fig1, fig2

    # Procedimiento normal si hay datos y atributos suficientes
    #Normalización de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[atributos])
    
    #Clustering mediante k-means a partir de datos normalizados
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_cluster["cluster"] = kmeans.fit_predict(X_scaled).astype(str)
    
    # Generar diccionario de colores para los clusters presentes, se usan mismos colores en ambas gráficas
    clusters_unicos = sorted(df_cluster["cluster"].unique())
    paleta_base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    colores = {str(i): paleta_base[i] for i in range(len(clusters_unicos))}
    colores["Sin datos"] = "lightgray"

    # Gráfico 1 (Reducción de dimensionalidad a las 2 componentes principales más representativas para visualizar la dispersión de los datos en el plano, solo si hay más de 1 atributo)
    if len(atributos) >= 2:
        #Análisis de componentes principales
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        df_cluster["PCA1"] = components[:, 0]
        df_cluster["PCA2"] = components[:, 1]
        #Diagrama de dispersión en el plano de las 2 componentes principales
        fig1 = px.scatter(
            df_cluster,
            x="PCA1",
            y="PCA2",
            color="cluster",
            color_discrete_map=colores,  
            hover_name="país",
            hover_data={"año": True, **{col: True for col in atributos}},  ## ** para desempaquetar un diccionario dentro del otro
            title=f"Clustering de países para el año {año} (PCA)",
            labels={
                **{col: f"{col} ({dict_unidades[col]})" for col in atributos if col in dict_unidades},
                "PCA1": "Componente principal 1",
                "PCA2": "Componente principal 2",
                "cluster": "Grupo"
            }
        )

        fig1.update_layout(legend_title_text="Grupo")
        fig1.update_layout(plot_bgcolor="white")
        fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    elif len(atributos) == 1: #Si solo se escoge un atributo, se muestra un histograma en lugar de diagrama de dispersión
        atributo = atributos[0]
        fig1 = px.histogram(
            df_cluster,
            x=atributo,
            color="cluster",
            color_discrete_map=colores,
            nbins= min(50, max(10, num_clusters * 4, len(df_cluster) // 6)),
            opacity=0.8,
            title=f"Distribución del atributo '{atributo}' por grupo para el año {año}",
            labels={
                atributo: f"{atributo} ({dict_unidades.get(atributo, '')})",
                "cluster": "Grupo"
            }
        )
        fig1.update_layout(barmode='overlay')
        fig1.update_layout(plot_bgcolor="white")
        fig1.update_layout(legend_title_text="Grupo")

    else:
        fig1 = go.Figure().update_layout(
            title="No se muestra gráfico porque no se ha seleccionado ningún atributo"
        )
        fig1.update_layout(plot_bgcolor="white")


    # Gráfico 2 (mapamundi con clustering)
    df_map = df_cluster.set_index("iso_code").copy()

    # Asignar cluster de la URSS a repúblicas ex-soviéticas si el año es <= 1991
    if año <= 1991:
        try:
            cluster_URSS = df_cluster[(df_cluster["país"] == "URSS") & (df_cluster["año"] == año)]["cluster"].values[0]
        except:
            cluster_URSS = "Sin datos"
        for iso in ex_soviet_iso:
            df_map.loc[iso, "cluster"] = cluster_URSS
            df_map.loc[iso, "país"] = "URSS"  # Para que el hover muestre URSS
            # Copiar los atributos a mostrar
            for col in atributos:
                try:
                    val = df_cluster[(df_cluster["país"] == "URSS") & (df_cluster["año"] == año)][col].values[0]
                    df_map.loc[iso, col] = val
                except:
                    df_map.loc[iso, col] = np.nan

    # Asignar los valores al GeoDataFrame
    world["grupo"] = world["ISO_A3"].map(df_map["cluster"])
    world["grupo"] = world["grupo"].fillna("Sin datos").astype(str)
    world["indice"] = world.index

    # Asignar valores de atributos como columnas (para hover)
    hover_dict = {"grupo": True, "indice": False, "name": False}
    for col in atributos:
        if col in df_map.columns:
            col_label = f"{col} ({dict_unidades[col]})" if col in dict_unidades else col
            world[col_label] = world["ISO_A3"].map(df_map[col])
            hover_dict[col_label] = True

    # Si es año ≤ 1991, poner nombre "URSS" a los ex-soviéticos
    if año <= 1991:
        world.loc[world["ISO_A3"].isin(ex_soviet_iso), "name"] = "URSS"

    fig2 = px.choropleth(
        world,
        geojson=world.geometry,
        locations="indice",
        color="grupo",
        color_discrete_map=colores,
        hover_name="name",
        hover_data=hover_dict,
        
    )
    fig2.update_geos(fitbounds="locations", visible=False)
    fig2.update_coloraxes(showscale=False)
    fig2.update_traces(showlegend=False)
    fig2.update_layout(
        title_text=f"Clustering de países para el año {año} (Mapamundi)",
        plot_bgcolor="white"
    )
    fig2.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0), 
            coloraxis_colorbar=dict(
                title="valor",
                x=0.95,  
                len=0.75 )
            )  
    # Revertir nombre para mantener nombres originales luego
    if año <= 1991:
        world.loc[world['ISO_A3'].isin(ex_soviet_iso), 'name'] = world.loc[world['ISO_A3'].isin(ex_soviet_iso), 'ISO_A3'].map(iso_to_spanish).fillna(world.loc[world['ISO_A3'].isin(ex_soviet_iso), 'name'])

    return fig1, fig2





# Callback para mostrar u ocultar la sección "Diagrama de dispersión" al hacer clic en el botón

@app.callback(
    Output("dispersion-content", "style"),   # Cambia el estilo (visible/oculto) del contenido
    Input("toggle-dispersion", "n_clicks"),  # Entrada: número de clics en el botón 
)
def toggle_dispersion(n_clicks):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}
    return {'display': 'none'} # Si es par -->ocultar contenido

# Callback que actualiza el gráfico de dispersión (graph2) en función del año,
# los atributos seleccionados para X, Y, tamaño, color y la escala de los ejes 

@app.callback(
    Output("graph2", "figure"),                     # Salida: figura del gráfico de dispersión
    Input("my-slider2", "value"),                   # Entrada: año seleccionado
    Input("dropdown21", "value"),                   # Entrada:Atributo para el eje X
    Input("dropdown22", "value"),                   # Entrada:Atributo para el eje Y
    Input("dropdown23", "value"),                   # Entrada:Atributo para el tamaño de los puntos
    Input("dropdown24", "value"),                   # Entrada:Atributo para el color de los puntos
    Input("scale-type", "value")                    # Entrada: Tipo de escala (lineal o logarítmica)
)
# Función que genera el gráfico de dispersión según el año y los atributos seleccionados,
# con opción de escala logarítmica, color, tamaño y recta de regresión
def grafica2(año, atributo_x, atributo_y, atributo_tamaño, atributo_color, escala):
    # Filtrar el DataFrame para el año seleccionado 
    df_año = df[df.año == año].copy()  
    # Ajuste para el atributo de tamaño, si está activo
    if atributo_tamaño != "None":
        minimo_at_tam = df_año[atributo_tamaño].min()
        # Evita tamaños nulos o muy pequeños en la visualización
        if pd.isna(minimo_at_tam) or minimo_at_tam <= 1e-3:
            minimo_at_tam = 1e-3
        # Rellenar valores nulos con el mínimo permitido
        df_año[atributo_tamaño].fillna(minimo_at_tam, inplace=True)
        # Crear etiqueta para mostrar los valores verdaderos sin afectar el tamaño del punto
        df_año["etiqueta_tamaño"] = df_año[atributo_tamaño].map(lambda x: "null" if x == minimo_at_tam  else x)
        # Asegurar que no haya valores negativos o nulos en tamaño
        df_año[atributo_tamaño] = df_año[atributo_tamaño].map(lambda x: minimo_at_tam if x <= 0 else x)
        
    # Regresión lineal con ajuste según escala
    df_año_clean = df_año.dropna(subset=[atributo_x, atributo_y]) # Crear subconjunto sin nulos en X e Y para la regresión
    
    # Cambio de escala de las variables si se usa escala logarítmica
    if escala == "log":
        df_año_clean = df_año_clean[(df_año_clean[atributo_x] > 0) & (df_año_clean[atributo_y] > 0)]  
        x = np.log10(df_año_clean[atributo_x])  
        y = np.log10(df_año_clean[atributo_y])
    else:
        x = df_año_clean[atributo_x]
        y = df_año_clean[atributo_y]
        
    # Calcular regresión lineal
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        # Si se usó logaritmo, deshacer la transformación para graficar
        if escala == "log":
            x_line = 10**x_line  
            y_line = 10**y_line

        linea_reg = True
    except:
        linea_reg = False  

    # Diagrama de dispersión con plotly-express scatter. Selección de configuración según atributos opcionales
    if atributo_tamaño == "None" and atributo_color == "None":
        fig2 = px.scatter(df_año, x=atributo_x, y=atributo_y, 
                      hover_data={"país": True, atributo_x: True, atributo_y: True},
                      labels={atributo_x: f"{atributo_x} ({dict_unidades[atributo_x]})",
                              atributo_y: f"{atributo_y} ({dict_unidades[atributo_y]})"},
                      )
    
    elif atributo_tamaño == "None":
        fig2 = px.scatter(df_año, x=atributo_x, y=atributo_y, color=atributo_color, 
                      hover_data={"país": True, atributo_x: True, atributo_y: True, atributo_color: True}, 
                      labels={atributo_x: f"{atributo_x} ({dict_unidades[atributo_x]})",
                              atributo_y: f"{atributo_y} ({dict_unidades[atributo_y]})",
                              atributo_color: f"{atributo_color} ({dict_unidades[atributo_color]})"},
                     )
    
    elif atributo_color == "None":
        fig2 = px.scatter(df_año, x=atributo_x, y=atributo_y, size=atributo_tamaño, 
                      hover_data={"país": True, atributo_x: True, atributo_y: True, 
                                  atributo_tamaño: False, "etiqueta_tamaño": True},
                      labels={atributo_x: f"{atributo_x} ({dict_unidades[atributo_x]})",
                              atributo_y: f"{atributo_y} ({dict_unidades[atributo_y]})",
                              "etiqueta_tamaño": f"{atributo_tamaño} ({dict_unidades[atributo_tamaño]})"},
                      )
    
    else: 
        fig2 = px.scatter(df_año, x=atributo_x, y=atributo_y, color=atributo_color, size=atributo_tamaño, 
                      hover_data={"país": True, atributo_x: True, atributo_y: True, 
                                  atributo_color: True, atributo_tamaño: False, "etiqueta_tamaño": True},
                      labels={atributo_x: f"{atributo_x} ({dict_unidades[atributo_x]})",
                              atributo_y: f"{atributo_y} ({dict_unidades[atributo_y]})",
                              atributo_color: f"{atributo_color} ({dict_unidades[atributo_color]})",
                              "etiqueta_tamaño": f"{atributo_tamaño} ({dict_unidades[atributo_tamaño]})"},
                      )
    
    #Añadir recta de regresión,  con etiquetas dinámicas adaptadas a cada escala
    if linea_reg:
        if escala == "log":
            equation_text = f"y = 10^{intercept:.6f} * x^{slope:.6f}<br>R<sup>2</sup> = {r_value**2:.3f}, p = {p_value:.3g}"
        else:
            equation_text = f"y = {slope:.6f}x + {intercept:.6f}<br>R<sup>2</sup> = {r_value**2:.3f}, p = {p_value:.3g}"
    
    try:
        fig2.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Recta de regresión', 
                              line=dict(color='red'), hoverinfo='text', text=equation_text))
    except: linea_reg = False
    
    #Mejoras estéticas de la figura
    fig2.update_layout(
        plot_bgcolor="white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=0, t=20, b=20)
    )
    
    #Configuración de ejes, incluida la escala escogida
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', type=escala)
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', type=escala)

    return fig2




# Callback para mostrar u ocultar la sección "Serie Temporal" al hacer clic en el botón

@app.callback(
    Output("ts-content", "style"),   # Cambia el estilo (visible/oculto) del contenido
    Input("toggle-ts", "n_clicks"),  # Entrada: número de clics en el botón 
)
def toggle_ts(n_clicks):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'} 
    return {'display': 'none'} # Si es par -->ocultar contenido   
    
# Callback para generar la figura de la serie temporal o tasa de variación según lo seleccionado
@app.callback(
    Output("graph1", "figure"),             # Devuelve la figura a mostrar en el componente "graph1"
    Input("dropdown11", "value"),           # Atributo seleccionado 
    Input("dropdown-countries", "value"),   # Lista de países seleccionados
    Input("radio-ts", "value")              # Opción entre serie temporal absoluta o tasa de variación
)
# Función que genera el gráfico de series temporales según el atributo y los países seleccionados,
# con opción de mostrar la tasa de variación porcentual a elección del usuario
def grafica1(atributo, countries, option):
    # Se filtra el dataframe, que incluye también los datos agregados mundiales, para quedarnos solo con los países seleccionados y sin valores nulos en el atributo
    df_countries = df_conmundo[df_conmundo["país"].isin(countries)].dropna(subset=[atributo])
    
    if option == 'rate': # Se calcula y se muestra la tasa de variación porcentual anual del atributo si así lo ha elegido el usuario
        df_countries["rate"] = df_countries.groupby("país")[atributo].pct_change() * 100
        y_axis = "rate"
        title = f'Tasa de variación anual de {atributo}'
        label_y = f'{atributo} (tasa anual %)'
    else: # En caso contrario (si option=='ts'), se muestra directamente la serie temporal del atributo
        y_axis = atributo
        title = f'Serie temporal de {atributo}'
        label_y = f'{atributo} ({dict_unidades[atributo]})'

    # Creación de la figura con Plotly-express line 
    fig = px.line(
        df_countries, x="año", y=y_axis, color='país',
        labels={"año": "Año", y_axis: label_y},
        title=title, 
    )
    # Mejoras estéticas de la figura  
    fig.update_layout(plot_bgcolor="white", margin=dict(l=40, r=0, t=30, b=20))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    return fig





# Callback para mostrar u ocultar la sección "Consultas SQL" al hacer clic en el botón
@app.callback(
    Output("SQL-content", "style"),  # Cambia el estilo (visible/oculto) del contenido
    Input("toggle-SQL", "n_clicks"), # Entrada: número de clics en el botón 
) 
def toggle_SQL(n_clicks):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}
    return {'display': 'none'} # Si es par -->ocultar contenido    
    
    
# Función para realizar las consultas mediante pandas SQL y controlar que solo se permitan consultas de lectura (SELECT)
def run_sql(query_text, df, q):
    try:
        if not query_text.strip().lower().startswith("select"):  #Si no empieza por "SELECT", se introduce en la cola de multiproceso el siguiente mensaje de error y se termina la ejecución de la función (return)
            q.put(("error", "Solo se permiten consultas de lectura (instrucciones SELECT)."))
            return  
        result = ps.sqldf(query_text, {"tabla": df})   #En caso contrario, realiza la consulta con pandas sql sobre el dataframe df (referido como tabla en la consulta) y se introduce el resultado de la consulta en la cola.
        q.put(("result", result))
    except Exception as e:
        q.put(("error", str(e)))  #Si hay algún otro error, se introduce en la cola para mostrarlo en pantalla
             
    
# Callback que ejecuta la consulta SQL escrita por el usuario al hacer clic en el botón de "ejecutar consulta" ('execute-query')
@app.callback(
    Output('query-results', 'children'),   # Salida donde se mostrará la tabla de resultados o los mensajes de error
    Output('query-result-store', 'data'),  # Guarda los resultados en formato lista de diccionarios de Python para poderlos descargar si así lo desea el usuario(ver siguiente callback)
    Input('execute-query', 'n_clicks'),    # Clic en el botón para ejecutar la consulta
    State('sql-input', 'value')            # Consulta SQL escrita por el usuario (estado actual del input)
)
def run_query(n_clicks, query):
    if n_clicks == 0 or not query:
        return "", {}


    q = mp.Queue()   #Se crea la cola de multiprocesos
    p = mp.Process(target=run_sql, args=(query, df, q)) # Se lanza un proceso independiente (multiprocessing.Process) que ejecuta la consulta mediante la función run_sql
    p.start()  
    p.join(timeout=40)   #Se le da 40 segundos para resolver la consulta, en caso de que no de respuesta, se cancela la consulta para evitar que se cuelgue el sistema

    if p.is_alive():  #si tras 40 segundos la consulta sigue realizandose, se termina y  se muestra en pantalla un mensaje explicativo
        p.terminate()
        p.join()
        return html.Div([
            " La consulta SQL ha tardado demasiado tiempo y ha sido cancelada automáticamente (timeout de 40 segundos)."
        ], style={'color': 'red'}), {}

    if not q.empty(): #si la cola no está vacía, se muestran los resultados de la consulta, ya sea la tabla con los resultados, o un error de consulta
        status, content = q.get()
        if status == "error":
            return html.Div([
                " Error al ejecutar la consulta: ", content
            ], style={'color': 'red'}), {}

        result = content  #data frame con los resultados
        data_dict = result.to_dict('records') #Lista de diccionarios con los resultados (cada diccionario corresponde a una fila).   ‘records’ : list like [{column -> value}, … , {column -> value}]   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html
        return html.Div([
            html.H4("Resultados de la consulta"),
            dash_table.DataTable(
                columns=[{"name": col, "id": col} for col in result.columns],  #tiene que ser una lista de diccionarios  con nombre e id https://dash.plotly.com/datatable
                data=data_dict,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                page_size=10,
                sort_action='native',
            )
        ]), data_dict  #devuelve la tabla html y la lista de diccionarios con los resultados para poderlos descargar en formato csv (siguiente callback)

    else: #Si la cola está vacía, se muestra mensaje de error.
        return html.Div([
            " La consulta no pudo completarse (proceso sin respuesta)."
        ], style={'color': 'red'}), {}



# Callback que permite descargar en CSV los resultados de la consulta SQL ejecutada previamente(almacenados como lista de diccionarios en el output query-result-store del callback anterior).
@app.callback(
    Output("download-link", "data"),       # Devuelve los datos preparados para descarga
    Input("download-csv", "n_clicks"),     # Clic en el botón de descarga
    State("query-result-store", "data"),   # Se usa State (no Input) para acceder a los datos sin disparar el callback cuando cambian. Resultados almacenados de la consulta previa
    prevent_initial_call=True              # Evita que se dispare al cargar la app por primera vez
)
def download_csv(n_clicks, query_data):  
    if query_data:
        # Se convierten los datos almacenados en una lista de diccionarios a un DataFrame para exportarlos como CSV usando el método .to_csv de Pandas
        df_result = pd.DataFrame(query_data)
        return dcc.send_data_frame(df_result.to_csv, "query_result.csv", index=False)    #https://dash.plotly.com/dash-core-components/download
    return dash.no_update # Si no hay datos, no se actualiza nada







# Callback para mostrar u ocultar la sección "Metadatos" al hacer clic en el botón
@app.callback(
    Output("meta-content", "style"),   # Cambia el estilo (visible/oculto) del contenido
    Output('meta-results', 'children'),  # Contenedor donde se inserta la tabla de metadatos
    Input("toggle-meta", "n_clicks"),  # Entrada: número de clics en el botón  
)
def toggle_meta(n_clicks):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        
        # Se muestra el contenido del DataFrame de metadatos, omitiendo la primera columna (índice)
        result = df_metadatos.reset_index(drop=True).iloc[:, 1:]
        # Se muestran los metadatos en una tabla interactiva de Dash
        return {'display': 'block'}, html.Div([
            dash_table.DataTable(
                columns=[{"name": col, "id": col} for col in result.columns],
                data=result.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                page_size=10,  
                sort_action='native',    
            )
        ])  
    return {'display': 'none'}, None # Si es par -->ocultar contenido    


# Ejecuta la app con el servidor de desarrollo de Dash (debug desactivado), accesible desde cualquier IP en el puerto 7082.  La IP pública está configurada desde AWS para permitir el acceso externo.
if __name__ == '__main__': app.run_server(debug=False, host='0.0.0.0', port=7082)
                                        
































