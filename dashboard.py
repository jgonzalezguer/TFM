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

# CARGA DE DATOS
df = pd.read_excel('owid-energy-data.xlsx')  # DataFrame principal con los datos energéticos
df_metadatos = pd.read_excel('owid-energy-data.xlsx', sheet_name=1)  # DataFrame con los metadatos: definiciones, unidades y fuentes de datos
world = gpd.read_file("ne_110m_admin_0_countries.geojson")  # GeoDataFrame con los polígonos de los países
world = world.rename(columns={"POP_EST": "value", "NAME": "name"}) 


#PREPROCESADO DE DATOS

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


# Creación de listas de países y atributos para los menús desplegables (dropdowns) del dashboard

countries = list(set(df[df['iso_code'].notnull()]['país'])) # Lista de países: se seleccionan aquellas filas con código ISO (las filas que no tienen código ISO no pertenecen a países) 
countries.append("Mundo")    # Se añade "Mundo" para permitir la visualización de valores agregados globales en la sección de Series Temporales
countries.append("Kosovo")   # Se añade manualmente Kosovo para que la lista de países coincida con los del GeoDataFrame

atributos = list(df.columns) # Lista de atributos del DataFrame principal
atributos_num = [atributo for atributo in atributos if df[atributo].dtypes == "float64"] # Filtro de atributos numéricos
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
df_conmundo = df[~df.iso_code.isnull() | (df["país"] == "Mundo")]

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
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Inicialización del layout 
app.layout = html.Div([
    # Encabezado con logo, título y autor
    html.Div([
        # Primera fila: logo + título
        dbc.Row([
            # Logo UOC (columna izquierda)
            dbc.Col(html.Img(src="http://materials.cv.uoc.edu/cdocent/common/img/logo-uoc.png",
                             alt="Logo UOC",
                             style={'margin': '20px 0px 0px', 'maxWidth': '100%'}), width=2),
            # Título principal (columna derecha)
            dbc.Col(html.H1("Dashboard con visualizaciones interactivas para el análisis de la evolución del consumo energético global",
                            style={'margin': '0px 0px 0px', 'fontSize': '40px','font-weight': 'bold','textAlign': 'center'}), width=9)
        ], style={'color': '#002a77', 'fontSize': '20px', 'paddingBottom': '20px','maxWidth': '95%'}),
        
        # Segunda fila: asignatura y autor
        dbc.Row([
            dbc.Col(html.Div([
            html.P(["  M2.882 - Trabajo Fin de Máster"],style={'margin': '0px 20px 0px'}),
            html.P(["  Autor: Jonay González Guerra"],style={'margin': '0px 20px 0px'}),
            ]), width=2),
        ], style={'color': '#002a77', 'fontSize': '15px', 'padding': '0px','maxWidth': '95%'})
    ],style={'backgroundColor': '#ADD8E6'}),
    
    html.Br(), # Separador visual
    # Párrafo introductorio con enlace a la fuente de datos
    html.P([
        "El uso de la energía es un factor determinante en el desarrollo económico y en la sostenibilidad ambiental. A medida que los países buscan equilibrar crecimiento económico, \
        eficiencia energética y reducción de emisiones de carbono, disponer de herramientas interactivas que faciliten el análisis de datos se vuelve cada vez más relevante. \
        Este Trabajo Fin de Máster presenta un dashboard con visualizaciones interactivas que permite explorar la relación entre consumo energético a partir de distintas fuentes,  \
        datos demográficos y económicos, así como de emisiones de gases de efecto invernadero en los distintos países del mundo a lo largo de las últimas décadas. \
        Los datos empleados para la visualización han sido obtenidos de la web: ",
        html.A("Energy-Our World in Data", href="https://ourworldindata.org/energy")
    ],style={'maxWidth': '95%', 'margin': '0 auto', 'textAlign': 'justify'}), 
    
    html.Br(),
    
    # Sección: Distribución mundial
    
    html.Div([
        # Botón para desplegar/plegar la sección interactiva
        html.Button([html.Div("Distribución mundial", style={'flex': 1, 'textAlign': 'left'}), html.Span(" ▼", style={'font-size': '24px', 'margin-left': 'auto'})  ], id="toggle-geo", n_clicks=0, 
        style={'font-size': '20px','width': '100%','height': '80px','textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77','border': 'none', 'outline': 'none','display': 'flex',  'alignItems': 'center'}),
        # Almacén local para guardar el estado de visibilidad del contenido (por defecto oculto)
        dcc.Store(id="store-geo", data=False),
        # Contenido oculto/visible según el estado del botón
        html.Div([html.Br(),
            # Texto explicativo de la sección
            html.P('En este apartado puedes observar cómo se distribuye un atributo a tu elección a lo largo del mundo y para un año concreto. \
            Para visualizar la distribución mundial de dicho atributo puedes escoger entre 3 tipos de gráficos:  Mapamundi, Treemap \
            o un diagrama de barras. En el caso de optar por el diagrama de barras, podrás escoger un número de países a mostrar, los cuales aparecerán ordenados según el valor del atributo de manera descendiente.', 
            style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
            # Contenedor con los controles y el gráfico
            html.Div([
                # Columna izquierda: controles de usuario
                html.Div([
                html.Br(),  
                # Selector de tipo de gráfico
                dcc.RadioItems(
                    id="radio-geo",
                    options=[
                        {'label': ' Mapamundi', 'value': 'world'},
                        {'label': ' Treemap', 'value': 'treemap'},
                        {'label': ' Diagrama de barras', 'value': 'barras'}
                    ],
                    value='world',
                    labelStyle={'display': 'block', 'margin': '5px', 'color': '#002a77'}
                ),
                # Dropdown de atributo
                html.P("Atributo:"),
                dcc.Dropdown(id="dropdown-geo", options=atributos_num, value='población', clearable=False, style={'color': '#002a77'}),
                html.P("Año:"),
                # Slider de año
                dcc.Slider(df.año.min(), df.año.max(), step=1, value=2010, marks=None, id='slider-geo', tooltip={"placement": "bottom", "always_visible": True}),
                # Slider de número de países (solo para el caso de diagrama de barras)
                html.Div([
                html.P("Número de países a mostrar:"),
                dcc.Slider(5, len(countries), step=1, value=20,marks=None, id='slider-top-n',tooltip={"placement": "bottom", "always_visible": True})
                ], id="slider-container", style={'display': 'none'})
            ], style={'width': '15%', 'padding': '20px'}),
            # Columna derecha: gráfico
            html.Div([
                dcc.Graph(id="graph-geo", style={"width": "70%"}),
            ], style={'width': '80%', 'padding': '0px'})
            ], style={'display': 'flex'}),
        ],  id="geo-content", style={'display': 'block'}) 
    ], style={'border': 'none', 'padding': '10px', 'margin': '10px'}),
    
    # Sección: Clustering
    html.Div([
        # Botón para desplegar/plegar la sección interactiva
        html.Button([html.Div("Clustering", style={'flex': 1, 'textAlign': 'left'}), html.Span(" ▼", style={'font-size': '24px', 'margin-left': 'auto'})  ], id="toggle-clusters", n_clicks=0, 
        style={'font-size': '20px','width': '100%','height': '80px','textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77','border': 'none', 'outline': 'none','display': 'flex',  'alignItems': 'center'}),
        # Almacén local para guardar el estado de visibilidad del contenido (por defecto oculto)
        dcc.Store(id="store-clusters", data=False),
        # Contenido oculto/visible según el estado del botón
        html.Div([html.Br(),
            # Texto explicativo de la sección
            html.P(
            "En este apartado los países se agrupan en clusters según los atributos seleccionados por el usuario mediante el algoritmo K-Means, "
            "configurable en número de grupos, año de análisis y variables consideradas. "
            "Si se seleccionan dos o más atributos se aplica una reducción de dimensiones con PCA y se muestra un gráfico de dispersión, "
            "mientras que con un solo atributo se presenta un histograma con la distribución por grupo. "
            "En ambos casos se incluye un mapa mundial donde cada país se colorea según el grupo al que pertenece.",
            style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
            # Contenedor con los controles y el gráfico de dispersión
            html.Div([
                # Columna izquierda: controles de usuario
                html.Div([
                html.Br(),  
                # Dropdown de atributos
                html.P("Atributos:"),
                dcc.Dropdown(
                        id="dropdown-atributos-cluster",
                        options=[{"label": a, "value": a} for a in atributos_num],   
                        multi=True,
                        value= ["fósil_proporción_energía", "energía_per_capita", "GEI_per_capita"],                         
                        clearable=False,
                        style={'color': '#002a77'}
                    ),
                # Slider de año
                html.P("Año:"),
                dcc.Slider(df.año.min(), df.año.max(), step=1, value=2010, marks=None, id='slider-año-clusters', tooltip={"placement": "bottom", "always_visible": True}),
                # Slider con el número de clusters
                html.P("Número de clusters:"),
                dcc.Slider(1, 10, step=1, value=3,marks=None, id='slider-num-clusters',tooltip={"placement": "bottom", "always_visible": True}),
            ], style={'width': '15%', 'padding': '20px'}),
            # Columna derecha: gráfico de dispersión
            html.Div([
                dcc.Graph(id="graph-clusters-1", style={"width": "100%"}),
            ], style={'width': '85%', 'padding': '0px'})
            ], style={'display': 'flex'}),
            # Gráfico Mapamundi con clusters
            html.Div([
                html.Div([
                    html.Div([], style={'width': '10%'}),
                    html.Div([
                        dcc.Graph(id="graph-clusters-2", style={"width": "100%"})
                    ], style={'width': '100%'})
                ], style={'display': 'flex'})
            ])
        ],  id="clusters-content", style={'display': 'block'}) 
    ], style={'border': 'none', 'padding': '10px', 'margin': '10px'}),

   
    # Sección: Diagrama de dispersión
    html.Div([
        # Botón para mostrar u ocultar el contenido de la sección
        html.Button([html.Div("Diagrama de dispersión", style={'flex': 1, 'textAlign': 'left'}), 
                 html.Span(" ▼", style={'font-size': '24px', 'margin-left': 'auto'})], 
                id="toggle-dispersion", n_clicks=0, 
                style={'font-size': '20px','width': '100%','height': '80px','textAlign': 'center', 
                       'backgroundColor': '#ADD8E6', 'color': '#002a77','border': 'none', 
                       'outline': 'none','display': 'flex',  'alignItems': 'center'}),
        # Almacena el estado de visibilidad de la sección (por defecto oculto)
        dcc.Store(id="store-dispersion", data=False),  
        # Contenido que se despliega al hacer clic en el botón
        html.Div([html.Br(),
            # Descripción de la funcionalidad
            html.P('En este apartado se muestra un diagrama de dispersión bidimensional para dos atributos y un año a tu elección, así como la recta de regresión lineal correspondiente con datos estadísticos relevantes. \
            La visualización también permite escoger dos atributos adicionales que se observarán en el tamaño de los puntos y su color.', 
            style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
            # Contenedor general con controles y gráfico
            html.Div([
                # Columna izquierda: controles de usuario
                html.Div([
                html.Br(),
                # Dropdowns de atributos
                html.P("Atributo eje X:"),
                dcc.Dropdown(id="dropdown21", options=atributos_num, value='pib per capita', clearable=False, style={'color': '#002a77'}),
                
                html.P("Atributo eje Y:"),
                dcc.Dropdown(id="dropdown22", options=atributos_num, value='energía_per_capita', clearable=False, style={'color': '#002a77'}),
                
                html.P("Atributo tamaño:"),
                dcc.Dropdown(id="dropdown23", options=atributos_num_None, value='None', clearable=False, style={'color': '#002a77'}),
                
                html.P("Atributo color:"),
                dcc.Dropdown(id="dropdown24", options=atributos_num_None, value='None', clearable=False, style={'color': '#002a77'}),
                
                # Slider de año
                html.P("Año:"),
                dcc.Slider(df.año.min(), df.año.max(), step=1, value=2010, marks=None, id='my-slider2',
                           tooltip={"placement": "bottom", "always_visible": True}),
                
                html.P("Escala de ejes:"),
                dcc.RadioItems(
                    id="scale-type",
                    options=[
                        {"label": "Escala real", "value": "linear"},
                        {"label": "Escala logarítmica", "value": "log"}
                    ],
                    value="linear",
                    labelStyle={'display': 'block', 'margin': '5px', 'color': '#002a77'}
                ),
                
            ], style={'width': '15%', 'padding': '20px'}),
            
            # Columna derecha con el diagrama de dispersión
            html.Div([
                dcc.Graph(id="graph2", style={"width": "100%"}),
            ], style={'width': '70%', 'padding': '20px'})
            
        ], style={'display': 'flex'}), 
        
    ], id="dispersion-content", style={'display': 'block'}) 
], style={'border': 'none', 'padding': '10px', 'margin': '10px'}),

    
    
    
    # Sección: Series temporales
    html.Div([
        # Botón para mostrar/ocultar la sección
        html.Button([html.Div("Series Temporales", style={'flex': 1, 'textAlign': 'left'}), html.Span(" ▼", style={'font-size': '24px', 'margin-left': 'auto'})  ], id="toggle-ts", n_clicks=0, 
        style={'font-size': '20px','width': '100%','height': '80px','textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77','border': 'none', 'outline': 'none','display': 'flex',  'alignItems': 'center'}),
        # Almacén del estado visible/oculto de la sección (por defecto oculto)
        dcc.Store(id="store-ts", data=False),
        # Contenido que se despliega al hacer clic en el botón
        html.Div([html.Br(),
        html.P('En este apartado se puede observar las series temporales correspondientes a un atributo y una serie de países a elección del usuario. \
        Puedes escoger entre visualizar el valor del atributo o su tasa de variación porcentual anual.', 
        style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
            # Contenedor con los controles y el gráfico
            html.Div([
                # Controles (columna izquierda)
                html.Div([
                    html.Br(),
                    html.P("Atributo a mostrar:"),
                    # Dropdown con el atributo a mostrar
                    dcc.Dropdown(id="dropdown11", options=atributos_num, value='población', clearable=False, style={'color': '#002a77'}),
                    html.P("Países:"),
                    # Dropdown con los países a mostrar
                    dcc.Dropdown(
                        id="dropdown-countries",
                        options=[{"label": c, "value": c} for c in countries],  
                        value=['España'],  
                        multi=True,  
                        clearable=False,
                        style={'color': '#002a77'}
                    ),
                    html.Br(),
                    # Selector del tipo de visualización (valor o tasa de variación anual)
                    dcc.RadioItems(
                        id="radio-ts",
                        options=[
                            {'label': ' Atributo', 'value': 'ts'},
                            {'label': ' Tasa de Variación Anual', 'value': 'rate'}
                        ],
                        value='ts',
                        labelStyle={'display': 'block', 'margin': '5px', 'color': '#002a77'}
                    )
                ], style={'width': '15%', 'padding': '20px'}),
                # Gráfico (columna derecha)
                html.Div([
                    dcc.Graph(id="graph1", style={"width": "100%"}),
                ], style={'width': '70%', 'padding': '20px'})
            ], style={'display': 'flex'}),
          ],  id="ts-content", style={'display': 'block'})
    ], style={'border': 'none', 'padding': '10px', 'margin': '10px'}),


    # Sección: Consultas SQL
    html.Div([
    # Botón para mostrar/ocultar el contenido
    html.Button([html.Div("Consultas SQL", style={'flex': 1, 'textAlign': 'left'}), html.Span(" ▼", style={'font-size': '24px', 'margin-left': 'auto'})  ], id="toggle-SQL", n_clicks=0, 
        style={'font-size': '20px','width': '100%','height': '80px','textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77','border': 'none', 'outline': 'none','display': 'flex',  'alignItems': 'center'}),
    # Almacenes del estado de visibilidad (inicialmente oculto) y de los resultados de la consulta
    dcc.Store(id="store-SQL", data=False),  dcc.Store(id="query-result-store", data={}),
    # Contenido de la sección
    html.Div([html.Br(),
        # Instrucciones de uso
        html.P('En este apartado puedes realizar consultas SQL sobre el conjunto de datos. Para ello ten en cuenta que el nombre de la tabla es "tabla".', 
        style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
        html.P('Ejemplo de consulta: ', 
        style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
        html.P('SELECT año, país, población, pib AS PIB, "pib per capita" AS "PIB per capita" FROM tabla WHERE país ="España" AND año BETWEEN 1980 AND 2022; ', 
        style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
        html.P('Puedes descargar los resultados de la consulta en formato CSV si lo deseas.', 
        style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
        # Área de entrada de la consulta y botones
        html.Div([
            html.H4("Escribe tu consulta SQL a continuación:"),
            # Entrada de texto para escribir la consulta
            dcc.Textarea(id='sql-input', style={'width': '100%', 'height': '100px', 'font-family': 'monospace'}),
            # Botones para ejecutar y descargar
            html.Button('Ejecutar consulta', id='execute-query', n_clicks=0),
            html.Button("Descargar CSV", id="download-csv", n_clicks=0, style={'margin-left': '10px'}),
            dcc.Download(id="download-link"),
            # Resultados de la consulta
            html.Div(id='query-results', style={'marginTop': '20px'})
        ], style={'padding': '20px'}),
    ],  id="SQL-content", style={'display': 'block'})  # Inicialmente visible
    ], style={'border': 'none', 'padding': '10px', 'margin': '10px'}),
    
    # Sección: Metadatos
    html.Div([
    # Botón para mostrar/ocultar la sección (inicialmente oculto)
    html.Button([html.Div("Metadatos", style={'flex': 1, 'textAlign': 'left'}), html.Span(" ▼", style={'font-size': '24px', 'margin-left': 'auto'})  ], id="toggle-meta", n_clicks=0, 
        style={'font-size': '20px','width': '100%','height': '80px','textAlign': 'center', 'backgroundColor': '#ADD8E6', 'color': '#002a77','border': 'none', 'outline': 'none','display': 'flex',  'alignItems': 'center'}),
    # Almacén para controlar la visibilidad de la sección (inicialmente oculto)   
    dcc.Store(id="store-meta", data=False),  
    # Contenido desplegable 
    html.Div([
        html.Div([
            # Descripción introductoria del contenido
            html.P('A continuación se muestra una tabla con la descripción de cada uno de los atributos representables en el dashboard, las unidades correspondientes y las fuentes de donde han sido obtenidos los datos:', 
        style={'maxWidth': '97%', 'margin': '0 auto', 'textAlign': 'justify'}),
            # Aquí se mostrará dinámicamente la tabla con los metadatos
            html.Div(id='meta-results', style={'marginTop': '20px'})
        ], style={'padding': '20px'}),
    ],  id="meta-content", style={'display': 'block'})  # Inicialmente visible
    ], style={'border': 'none', 'padding': '10px', 'margin': '10px'}),    
    

    html.Br(),
    # Pie de página (footer) con el logo de la UOC
    html.Div(style={'background': '#002a77', 'padding': '35px 0px', 'marginTop': '25px','height':'140px'}, children=[
        html.Div(className='row', children=[
            html.Div(className='col-sm-12', children=[
                html.Img(src="http://materials.cv.uoc.edu/cdocent/common/img/logo-uoc-bottom.png",
                         alt="Logo UOC",
                         style={'margin': '0 auto', 'display': 'block', 'maxWidth': '95%'})
            ])
        ])
    ])
            
    
    ],  style={'color': '#002a77', 'maxWidth': '2000px', 'margin': '0 auto', 'overflowX': 'hidden'})
    
    
    
  
# CALLBACKS Y FUNCIONES CON GRÁFICOS Y CONSOLA SQL

# Callback para mostrar u ocultar la sección "Distribución mundial" al hacer clic en el botón
@app.callback(
    Output("geo-content", "style"), # Cambia el estilo (visible/oculto) del contenido
    Output("store-geo", "data"),    # Actualiza el valor almacenado en el Store (True/False)
    Input("toggle-geo", "n_clicks"),# Entrada: número de clics en el botón 
    Input("store-geo", "data")      # Entrada: valor actual de visibilidad (True o False)
)
def toggle_geo(n_clicks, is_visible):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}, True  
    return {'display': 'none'}, False # Si es par -->ocultar contenido
    
# Callback que genera la figura de distribución mundial (mapamundi, treemap o barras)
# según el tipo de visualización, el atributo seleccionado, el año y el número de países a mostrar (si aplica)

@app.callback(
    Output("graph-geo", "figure"), # Salida: gráfico que se muestra en la sección
    Input("radio-geo", "value"),   # Entrada: tipo de gráfico seleccionado (world, treemap, barras)
    Input("dropdown-geo", "value"),# Entrada: atributo numérico seleccionado
    Input("slider-geo", "value"),  # Entrada: año seleccionado
    Input("slider-top-n", "value") # Entrada: número de países a mostrar (solo para gráfico de barras)
)    

# Función que genera la figura correspondiente a la sección "Distribución mundial"
# según el tipo de visualización seleccionada, el atributo, el año y el número de países (solo para gráfico de barras)

def actualizar_geo(vista, atributo, año,num_paises):
    # Mapamundi
    if vista == 'world':
        # Crear una serie de Pandas con los valores del atributo por país en ese año, para luego asignarsela a la columna value del GeoDataFrame "world"
        nueva_column=pd.Series(data=np.repeat(np.nan,world.shape[0]))
        for i in range(world.shape[0]):
            try: nueva_column.iloc[i]=df[(df.año==año)&(df.iso_code==world.iloc[i].ISO_A3)][atributo].values[0]
            except: nueva_column.iloc[i]=np.nan
        
        world.value=nueva_column
        
        # Capa base para mostrar países sin datos
        world_base=world.copy()
        world_base["value"] = world["value"].fillna(-1).apply(lambda x: "Sin datos" if x == -1 else "Con datos")  # Usa -1 para distinguir sin datos
        
               
        world["indice"] = world.index
        world_base["indice"] = world_base.index
        
        # Se crea la figura, con plotly-express cloropleth, de la capa base en blanco/gris según disponibilidad de datos
        fig = px.choropleth(world_base, geojson=world_base.geometry, locations="indice", 
                             color="value",
                             color_discrete_map={"Sin datos": "lightgray", "Con datos": "white"},
                             hover_name="name",  
                             hover_data={"value": False,"indice":False},  
                             width=1500, height=600)

        # Se crea la figura, con plotly-express cloropleth, de la capa con los valores del atributo y escala de color continua
        fig_data = px.choropleth(world, geojson=world.geometry, locations="indice",  
                             color="value", 
                             hover_name="name", 
                             hover_data={"value": True,"indice":False},  
                             color_continuous_scale=px.colors.sequential.Plasma,
                             labels={"value": f"{atributo} ({dict_unidades[atributo]})"},
                             width=1500, height=600)

        # Añadir capa de datos sobre el mapa base
        fig.add_trace(fig_data.data[0])  

        # Mejoras estéticas de la figura
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_coloraxes(colorbar_title=f"{atributo} ({dict_unidades[atributo]})")
        fig.update_traces(showlegend=False)

        return fig
        
        
    # Treemap            
    elif vista=='treemap':
        # Verificar si el atributo es adecuado para un treemap (magnitud absoluta y positiva)
        if atributo in atributos_num_absolutos:
            df_año = df[df.año == año].copy().dropna(subset=["continente"]) #Se filtran los datos para el año escogido
            total_mundial = df_año[atributo].sum()
            
            # Generar treemap jerárquico: Mundo → Continente → País con plotly-express treemap
            fig = px.treemap(
                df_año, 
                path=[px.Constant("Mundo"), 'continente', 'país'], 
                values=atributo, 
                labels={atributo: f"{atributo} ({dict_unidades[atributo]})"},
                width=1300, height=500
            )
            
            # Personalización de etiquetas y porcentaje sobre el total
            for trace in fig.data:
                customdata = []
                text_values = []
                
                for value, label in zip(trace.values, trace.labels):
                    percent = (value / total_mundial) * 100 if total_mundial > 0 else None
                    customdata.append([percent])
                    

                    text_values.append(f"{label}<br>{value:,.2f} {dict_unidades[atributo]}<br>{percent:.2f}%")

                trace.customdata = np.array(customdata)
                trace.text = text_values  
                trace.texttemplate = "%{text}"  
                
         
            # Tooltip personalizado y color raíz
            fig.update_traces(
                root_color='#ADD8E6',
                hovertemplate='<b>%{label}</b><br>' + atributo +
                              f' ({dict_unidades[atributo]}): ' + '%{value} <br>' + 
                              'Porcentaje: %{customdata[0]:.2f}%<extra></extra>'
            )

            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            return fig
        else: 
           # Si el atributo no es apto para treemap (por ser relativo o contener valores negativos)
           fig = go.Figure()

           fig.add_annotation(
                text="Un treemap no es una visualización apropiada para este tipo de atributo.",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=20)
            )

           fig.update_layout(
                width=1300, height=500,
                xaxis={'visible': False},
                yaxis={'visible': False},
                plot_bgcolor="#f0f0f0",
                margin=dict(t=50, l=25, r=25, b=25)
            )

           return fig
           
    # Diagrama de barras       
    else:
    
            df_año = df[df.año == año].copy().dropna(subset=["continente"])  #Se filtran los datos para el año escogido
            df_top = df_año.nlargest(num_paises, atributo)  #se seleccionan los N mayores
            df_top["rank"] = range(1, len(df_top) + 1)  #Se crea una nueva columna con el número en el ranking
            df_top=df_top.sort_values(atributo, ascending=False)  #Se ordenan los valores del atributo de mayor a menor
            
            # Creación de la figura con Plotly-express bar    
            fig = px.bar(
                df_top, 
                x=atributo,  
                y="país",  
                color="país",
                orientation="h",  
                labels={atributo: f"{atributo} ({dict_unidades[atributo]})"},hover_data={"país":True,'rank':True,atributo:True},
                title=f"Top {num_paises} países según el atributo {atributo} en el año {año}",
                width=1300, height=500
                )
            # Mejoras estéticas de la figura
            fig.update_layout(plot_bgcolor="white",showlegend=False)
            fig.update_yaxes(showgrid=False)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            
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
    Output("store-clusters", "data"),    # Actualiza el valor almacenado en el Store (True/False)
    Input("toggle-clusters", "n_clicks"),# Entrada: número de clics en el botón 
    Input("store-clusters", "data")      # Entrada: valor actual de visibilidad (True o False)
)
def toggle_clusters(n_clicks, is_visible):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}, True  
    return {'display': 'none'}, False # Si es par -->ocultar contenido
    
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
    df_cluster = df_año[["país", "año", "iso_code"] + atributos].dropna()
    
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
            width=1500,
            height=600
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
    
    # Generar colores dinámicos para los clusters presentes, se usan mismos colores en ambas gráficas
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
            hover_data={"año": True, **{col: True for col in atributos}},
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


    # Gráfico 2 (mapamundi)
    df_map = df_cluster.set_index("iso_code")
    world["grupo"] = world["ISO_A3"].map(df_map["cluster"])
    world["grupo"] = world["grupo"].fillna("Sin datos").astype(str)
    world["indice"] = world.index


    hover_dict = {"grupo": True, "indice": False, "name": False}
    for col in atributos:
        if col in df_map.columns:
            col_label = f"{col} ({dict_unidades[col]})" if col in dict_unidades else col
            world[col_label] = world["ISO_A3"].map(df_map[col])
            hover_dict[col_label] = True

    fig2 = px.choropleth(
        world,
        geojson=world.geometry,
        locations="indice",
        color="grupo",
        color_discrete_map=colores,
        hover_name="name",
        hover_data=hover_dict,
        width=1500,
        height=600
    )
    fig2.update_geos(fitbounds="locations", visible=False)
    fig2.update_coloraxes(showscale=False)
    fig2.update_traces(showlegend=False)
    fig2.update_layout(
        title_text=f"Clustering de países para el año {año} (Mapamundi)",
        plot_bgcolor="white"
    )

    return fig1, fig2




# Callback para mostrar u ocultar la sección "Diagrama de dispersión" al hacer clic en el botón

@app.callback(
    Output("dispersion-content", "style"),   # Cambia el estilo (visible/oculto) del contenido
    Output("store-dispersion", "data"),      # Actualiza el valor almacenado en el Store (True/False)
    Input("toggle-dispersion", "n_clicks"),  # Entrada: número de clics en el botón 
    Input("store-dispersion", "data")        # Entrada: valor actual de visibilidad (True o False)
)
def toggle_dispersion(n_clicks, is_visible):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}, True  
    return {'display': 'none'}, False # Si es par -->ocultar contenido

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
                      width=1400, height=500)
    
    elif atributo_tamaño == "None":
        fig2 = px.scatter(df_año, x=atributo_x, y=atributo_y, color=atributo_color, 
                      hover_data={"país": True, atributo_x: True, atributo_y: True, atributo_color: True}, 
                      labels={atributo_x: f"{atributo_x} ({dict_unidades[atributo_x]})",
                              atributo_y: f"{atributo_y} ({dict_unidades[atributo_y]})",
                              atributo_color: f"{atributo_color} ({dict_unidades[atributo_color]})"},
                      width=1400, height=500)
    
    elif atributo_color == "None":
        fig2 = px.scatter(df_año, x=atributo_x, y=atributo_y, size=atributo_tamaño, 
                      hover_data={"país": True, atributo_x: True, atributo_y: True, 
                                  atributo_tamaño: False, "etiqueta_tamaño": True},
                      labels={atributo_x: f"{atributo_x} ({dict_unidades[atributo_x]})",
                              atributo_y: f"{atributo_y} ({dict_unidades[atributo_y]})",
                              "etiqueta_tamaño": f"{atributo_tamaño} ({dict_unidades[atributo_tamaño]})"},
                      width=1400, height=500)
    
    else: 
        fig2 = px.scatter(df_año, x=atributo_x, y=atributo_y, color=atributo_color, size=atributo_tamaño, 
                      hover_data={"país": True, atributo_x: True, atributo_y: True, 
                                  atributo_color: True, atributo_tamaño: False, "etiqueta_tamaño": True},
                      labels={atributo_x: f"{atributo_x} ({dict_unidades[atributo_x]})",
                              atributo_y: f"{atributo_y} ({dict_unidades[atributo_y]})",
                              atributo_color: f"{atributo_color} ({dict_unidades[atributo_color]})",
                              "etiqueta_tamaño": f"{atributo_tamaño} ({dict_unidades[atributo_tamaño]})"},
                      width=1400, height=500)
    
    #Añadir recta de regresión,  con etiquetas dinámicas adaptadas a cada caso
    if linea_reg:
        if escala == "log":
            equation_text = f"y = 10^{intercept:.6f} * x^{slope:.6f}<br>R² = {r_value**2:.3f}, p = {p_value:.3g}"
        else:
            equation_text = f"y = {slope:.6f}x + {intercept:.6f}<br>R² = {r_value**2:.3f}, p = {p_value:.3g}"
    
    try:
        fig2.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Recta de regresión', 
                              line=dict(color='red'), hoverinfo='text', text=equation_text))
    except: linea_reg = False
    
    #Mejoras estéticas de la figura
    fig2.update_layout(
        plot_bgcolor="white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    #Configuración de ejes, incluida la escala escogida
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', type=escala)
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', type=escala)

    return fig2




# Callback para mostrar u ocultar la sección "Serie Temporal" al hacer clic en el botón

@app.callback(
    Output("ts-content", "style"),   # Cambia el estilo (visible/oculto) del contenido
    Output("store-ts", "data"),      # Actualiza el valor almacenado en el Store (True/False)
    Input("toggle-ts", "n_clicks"),  # Entrada: número de clics en el botón 
    Input("store-ts", "data")        # Entrada: valor actual de visibilidad (True o False)
)
def toggle_ts(n_clicks, is_visible):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}, True  
    return {'display': 'none'}, False # Si es par -->ocultar contenido   
    
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
    # Cálculo del rango del eje X, ya que hay combinaciones de atributos y países con rangos diferentes
    min_año = df_countries["año"].min()
    max_año = df_countries["año"].max()
    
    # Creación de la figura con Plotly-express line 
    fig = px.line(
        df_countries, x="año", y=y_axis, color='país',
        labels={"año": "Año", y_axis: label_y},
        title=title, width=1400, height=500
    )
    # Mejoras estéticas de la figura  
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[min_año, max_año])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    return fig




# Callback para mostrar u ocultar la sección "Consultas SQL" al hacer clic en el botón

@app.callback(
    Output("SQL-content", "style"),  # Cambia el estilo (visible/oculto) del contenido
    Output("store-SQL", "data"),     # Actualiza el valor almacenado en el Store (True/False)
    Input("toggle-SQL", "n_clicks"), # Entrada: número de clics en el botón 
    Input("store-SQL", "data")       # Entrada: valor actual de visibilidad (True o False)
) 
def toggle_SQL(n_clicks, is_visible):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}, True  
    return {'display': 'none'}, False # Si es par -->ocultar contenido    
    
# Callback que ejecuta la consulta SQL escrita por el usuario al hacer clic en el botón de "ejecutar consulta" ('execute-query')
@app.callback(
    Output('query-results', 'children'),   # Salida donde se mostrará la tabla de resultados o los mensajes de error
    Output('query-result-store', 'data'),  # Guarda los resultados en formato diccionario de Python para poderlos descargar si así lo desea el usuario(ver siguiente callback)
    Input('execute-query', 'n_clicks'),    # Clic en el botón para ejecutar la consulta
    State('sql-input', 'value')            # Consulta SQL escrita por el usuario (estado actual del input)
)

def run_query(n_clicks, query):
    if n_clicks == 0 or not query: # Si no se ha hecho clic o no hay consulta escrita, no se hace nada
        return "", {}

    try:
        if not query.strip().lower().startswith("select"): # Solo se permiten consultas SELECT por seguridad y simplicidad
            return html.Div(["Solo se permiten consultas de lectura (instrucciones SELECT)."]), {}
        # Se ejecuta la consulta usando pandasql sobre el DataFrame principal (df), al que llamamos 'tabla' dentro de la consulta (así se ha explicado en las instrucciones en html)
        result = ps.sqldf(query, {"tabla": df})
        # Se convierten los resultados a diccionario para almacenarlos, para si el usuario quiere descargarlos (ver siguiente callback)
        data_dict = result.to_dict('records') 
        # Se muestran los resultados en una tabla interactiva de Dash y se devuelve también el diccionario con los datos almacenados
        return html.Div([
            html.H4("Resultados de la consulta"),
            dash_table.DataTable(
                columns=[{"name": col, "id": col} for col in result.columns],
                data=result.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                page_size=10,  
                sort_action='native',    
            )
        ]), data_dict
    # Si hay un error al ejecutar la consulta, se muestra al usuario en color rojo.
    except Exception as e:
        return html.Div(["Error al ejecutar la consulta: ", str(e)], style={'color': 'red'}), {}

# Callback que permite descargar en CSV los resultados de la consulta SQL ejecutada previamente(almacenados como diccionario en el output query-result-store del callback anterior).
@app.callback(
    Output("download-link", "data"),       # Devuelve los datos preparados para descarga
    Input("download-csv", "n_clicks"),     # Clic en el botón de descarga
    State("query-result-store", "data"),   # Estado: resultados almacenados de la consulta previa
    prevent_initial_call=True              # Evita que se dispare al cargar la app por primera vez
)
def download_csv(n_clicks, query_data):
    if query_data:
        # Se convierten los datos almacenados en un diccionario a un DataFrame para exportarlos como CSV usando el método .to_csv de Pandas
        df_result = pd.DataFrame(query_data)
        return dcc.send_data_frame(df_result.to_csv, "query_result.csv", index=False)   
    return dash.no_update # Si no hay datos, no se actualiza nada


# Callback para mostrar u ocultar la sección "Metadatos" al hacer clic en el botón
@app.callback(
    Output("meta-content", "style"),   # Cambia el estilo (visible/oculto) del contenido
    Output("store-meta", "data"),      # Actualiza el valor almacenado en el Store (True/False)
    Input("toggle-meta", "n_clicks"),  # Entrada: número de clics en el botón  
    Input("store-meta", "data")        # Entrada: valor actual de visibilidad (True o False)
)
def toggle_meta(n_clicks, is_visible):
    n_clicks = n_clicks or 0  # Si aún no se ha hecho clic, se considera 0
    if n_clicks % 2 == 1:  # Si el número de clics es impar --> mostrar contenido
        return {'display': 'block'}, True  
    return {'display': 'none'}, False # Si es par -->ocultar contenido    
 
 
@app.callback(
    Output('meta-results', 'children'),  # Contenedor donde se inserta la tabla de metadatos
    Input('store-meta', 'data')          # Se activa cuando cambia la visibilidad de la sección
)
def meta(is_visible):
        # Se muestra el contenido del DataFrame de metadatos, omitiendo la primera columna (índice)
        result = df_metadatos.reset_index(drop=True).iloc[:, 1:]
        # Se muestran los metadatos en una tabla interactiva de Dash
        return html.Div([
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


# Ejecuta la app con el servidor de desarrollo de Dash (debug desactivado), accesible desde cualquier IP en el puerto 7082 
if __name__ == '__main__': app.run_server(debug=False, host='0.0.0.0', port=7082)
                                        
                                        
































