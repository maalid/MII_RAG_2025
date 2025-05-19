# 📚 RAG Indexing y Consulta con LlamaIndex + Azure/OpenAI

Este proyecto permite crear y consultar índices vectoriales persistentes sobre documentos usando [LlamaIndex](https://github.com/jerryjliu/llama_index), integrando modelos de embedding y respuesta tanto de OpenAI como de Azure OpenAI.

Se utiliza un enfoque RAG (Retrieval-Augmented Generation) para responder preguntas basadas en documentos con soporte para filtros, distintos modos de respuesta y visualización de resultados enriquecida.

## 📂 Estructura del Proyecto

``` bash
project/
│
├── .env.template                  # API key de OpenAI
├── .gitignore
├── README.md
├── requierements.txt
├── renv.lock
└── src
    │
    ├── execute.py                 # Código para ejecutar la creación del index y el retrieval según la pregunta
    ├── config/
    │   │
    │   ├── paths.yaml             # Rutas a recursos estáticos, estilos, logo, etc.
    │   └── llm.yaml               # Configuración del modelo LLM
    │
    ├── utils/
    │   │
    │   ├── config_functions.py    # Carga y resolución de rutas desde YAML
    │   └── llm_functions.py       # Abstracción de modelos de embedding y respuesta
    │
    ├── rag/                       # Archivos estáticos (logo, otros)
    │   │
    │   ├── create_index_functions # Script con las funciones para generar el index a partir de documentos
    │   └── query_index_functions  # Script con las funciones para hacer el retrieval a partir de una pregunta
    │
    └── persisted_index/           # Carpeta donde se guardará el index construido
```

## 🧪 Requisitos

-   Python 3.9+
-   openai
-   python-dotenv
-   LlamaIndex

Instalación rápida:

``` bash
pip install -r requirements.txt
```

Si usas [`renv`](https://rstudio.github.io/renv/articles/python.html) puedes ejecutar el siguiente comando para instalar las librerías python desde el `requirements.txt`:

``` bash
renv::restore()
```

## ⚙️ Configuración

1.  Claves de API

    - Cambia el nombre del archivo `.env.template` a `.env`
    - Si quieres usar los modelos de AZURE OpenAi, cambia a `True` el primer campo
    - Luego pega tu clave de AZURE OpenAi o de OpenAI

``` env
# Usa Azure en vez de OpenAI estandar
USE_AZURE_OPENAI = False

# Azure OpenAI
AZURE_OPENAI_API_KEY = '...'

# OpenAI (estandar)
OPENAI_API_KEY = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

2.  Configuración de rutas (`paths.yaml`)

    Contenido:

``` yaml
# Documents to index path
docs_to_index_path: 'C:/path/to/docs'

# Persisted Index path
persisted_index_path: '../persisted_index'
```

En el archivo anterior, solo debes ajustar `docs_to_index_path` de acuerdo a tu caso (esta ruta es la que define dónde estarán los documentos que quieres indexar).

3.  Configuración del modelo (`llm.yaml`)

    Contenido:

``` yaml
# RAG response LLM and embedding LLM
rag:
  answer_question_llm:
    model_name: 'gpt-4o'
    azure_api_version: '...'
    azure_deployment_name: '...'
    azure_endpoint: '...'
  embedding_llm:
    model_name: 'text-embedding-3-large'
    azure_api_version: '...'
    azure_deployment_name: '...'
    azure_endpoint: '...'
```

Si quieres usar otros modelos de OpenAi, puedes cambiar el nombre del modelo modificando el valor de `model_name`.
Si quieres usar modelos OpenAi de AZURE, debes llenar los campos que están con `...`.

## 🧠 ¿Qué hace este proyecto?
1. Indexación Persistente
Lee documentos desde un directorio, aplica embeddings y construye un índice vectorial que se guarda en disco:

``` bash
create_persisted_index(docs_dir  = "docs/",
                       index_dir = "index/",
                       index_id  = "mi_indice")
```

2. Consulta con Recuperación + Generación (RAG)
Permite consultar el índice con distintas estrategias de respuesta:

``` bash
query_persisted_index(index_dir         = "index/",
                      index_id          = "mi_indice",
                      query             = "¿Qué dice el documento sobre X?",
                      response_mode     = "refine",
                      similarity_cutoff = 0.75)
```

Los modos de respuesta incluyen:

- refine: respuestas iterativas y refinadas.

- compact: respuestas directas y breves.

- simple_summarize: versión simple del modo compacto.

- tree_summarize: fusión jerárquica de fragmentos.

- accumulate: acumulación de fragmentos.

- no_text: solo recuperación sin LLM.

## 📌 Características destacadas
Soporte para OpenAI y Azure OpenAI (conmutación automática vía .env)

- Filtros por metadatos y score (similarity_cutoff)

- Prompts personalizados por modo de respuesta

- Reporte de fragmentos utilizados por archivo

- Output enriquecido y trazabilidad del origen de la información

## 📝 Licencia

Este proyecto es de uso interno / académico / personal. Modifícalo libremente según tus necesidades.