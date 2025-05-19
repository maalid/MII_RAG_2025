from src.rag.create_index_functions import create_persisted_index
from src.utils.config_functions import load_config, resolve_path


paths_config = load_config(config_filename = "paths.yaml")

documents_folder_path = resolve_path(paths_config['docs_to_index_path'])
persisted_index_path  = resolve_path(paths_config["persisted_index_path"])

create_persisted_index(docs_dir  = documents_folder_path,
                       index_dir = persisted_index_path,
                       index_id  = "rag_index")



from src.rag.query_index_functions import query_persisted_index

mi_pregunta = '¿Que es la Inteligencia Artificial?'

query_persisted_index(index_dir         = persisted_index_path,
                      index_id          = "rag_index",
                      response_mode     = 'compact',
                      similarity_cutoff = 0.40,
                      similarity_top_k  = 5,
                      query             = mi_pregunta)
