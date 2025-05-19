from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.settings import Settings
from llama_index.core.prompts import PromptTemplate

from typing import Optional, Dict, Any
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from collections import defaultdict
from src.utils.llm_functions import get_rag_embedding_model, get_rag_answer_question_model
from src.utils.config_functions import load_config


llm_model_config = load_config(config_filename = "llm.yaml")


def get_custom_synthesizer(response_mode: str,
                           llm = None):
    
    """
    Retorna un response_synthesizer personalizado según el modo solicitado.
    """

    if response_mode == "no_text":
        Settings.llm = None
        return get_response_synthesizer(response_mode = response_mode,
                                        llm           = None)

    elif response_mode == "refine":
        initial_prompt = PromptTemplate(
            "Contexto:\n{context_str}\n\n"
            "Pregunta: {query_str}\n\n"
            "Genera una respuesta inicial clara y precisa:"
        )

        refine_prompt = PromptTemplate(
            "Respuesta parcial anterior:\n{existing_answer}\n\n"
            "Nuevo contexto:\n{context_str}\n\n"
            "Refina y mejora la respuesta considerando el nuevo contexto:"
        )

        return get_response_synthesizer(response_mode    = response_mode,
                                        llm              = llm,
                                        text_qa_template = initial_prompt,
                                        refine_template  = refine_prompt)

    elif response_mode == "compact":
        compact_prompt = PromptTemplate(
            "Contesta la siguiente pregunta usando la información dada en el contexto.\n\n"
            "Si la pregunta no tiene relación, responde solicitando contexto más específico.\n\n"
            "Si el contexto para responder la pregunta hace referencia a una imagen, contesta diciendo que la pregunta tiene relación con un diagrma de flujo por lo que es necesario revisar la imagen referenciada para obtener la respuesta precisa.\n\n"
            "Contexto:\n{context_str}\n\n"
            "Pregunta: {query_str}\n\n"
            "Respuesta:"
        )

        return get_response_synthesizer(response_mode    = response_mode,
                                        llm              = llm,
                                        text_qa_template = compact_prompt)

    elif response_mode == "simple_summarize":
        simple_prompt = PromptTemplate(
            "Contesta la siguiente pregunta usando la información dada en el contexto.\n\n"
            "Si la pregunta no tiene relación, responde solicitando contexto más específico.\n\n"
            "Si el contexto para responder la pregunta hace referencia a una imagen, contesta diciendo que la pregunta tiene relación con un diagrma de flujo por lo que es necesario revisar la imagen referenciada para obtener la respuesta precisa.\n\n"
            "Contexto:\n{context_str}\n\n"
            "Pregunta: {query_str}\n\n"
            "Respuesta breve:"
        )

        return get_response_synthesizer(response_mode    = response_mode,
                                        llm              = llm,
                                        text_qa_template = simple_prompt)

    elif response_mode == "tree_summarize":
        tree_prompt = PromptTemplate(
            "Contesta la siguiente pregunta usando la información dada en el contexto.\n\n"
            "Si la pregunta no tiene relación, responde solicitando contexto más específico.\n\n"
            "Si el contexto para responder la pregunta hace referencia a una imagen, contesta diciendo que la pregunta tiene relación con un diagrma de flujo por lo que es necesario revisar la imagen referenciada para obtener la respuesta precisa.\n\n"
            "Lee el siguiente contenido:\n{context_str}\n\n"
            "Luego responde a esta pregunta:\n{query_str}\n\n"
            "Resumen:"
        )

        return get_response_synthesizer(response_mode    = response_mode,
                                        llm              = llm,
                                        text_qa_template = tree_prompt)
    
    elif response_mode == "accumulate":
        accumulate_prompt = PromptTemplate(
            "Pregunta: {query_str}\n"
            "Fragmento:\n{context_str}\n"
            "Responde a la pregunta en base solo a este fragmento. Si no hay información relevante, indica 'No hay información suficiente'."
        )
        return get_response_synthesizer(response_mode    = response_mode,
                                        llm              = llm,
                                        text_qa_template = accumulate_prompt)
    
    else:
        raise ValueError(f"response_mode no reconocido: {response_mode}")

def query_persisted_index(index_dir: str,
                          index_id: str,
                          query: str,
                          response_mode: str = 'refine',
                          similarity_cutoff: float = None,
                          similarity_top_k: int = 5,
                          filters: Optional[MetadataFilters] = None) -> Dict[str, Any]:
    
    # 1. Cargar el contexto de almacenamiento desde el directorio persistido
    storage_context = StorageContext.from_defaults(persist_dir = index_dir)
    
    # 2. Definir el mismo modelo de embeddings usado en la indexación
    # embedding_model = get_embed_model(model_name = config_llm["rag"]["embedding_llm"]["model_name"])
    embedding_model = get_rag_embedding_model(model_name            = llm_model_config["rag"]["embedding_llm"]["model_name"],
                                              azure_api_version     = llm_model_config["rag"]["embedding_llm"]["azure_api_version"],
                                              azure_deployment_name = llm_model_config["rag"]["embedding_llm"]["azure_deployment_name"],
                                              azure_endpoint        = llm_model_config["rag"]["embedding_llm"]["azure_endpoint"])

    # 3. Cargar el índice desde ese contexto
    index = load_index_from_storage(storage_context = storage_context,
                                    index_id        = index_id,
                                    embed_model     = embedding_model)

    # 4. Crear retriever
    retriever = VectorIndexRetriever(index             = index,
                                     similarity_top_k  = similarity_top_k,
                                     filters           = filters,
                                     embed_model       = embedding_model)
    
    # 5. Recuperar los nodos y filtrar segun similarity_cutoff
    nodes = retriever.retrieve(query)
    if similarity_cutoff is not None:
        nodes = [n for n in nodes if n.score >= similarity_cutoff]
    print("\nNodos luego del filtrado manual:")
    for i, node in enumerate(nodes, 1):
        print(f"Nodo {i}: score = {node.score:.4f}")
    
    # 6. Crear sintetizador según el modo
    # llm = None if response_mode == "no_text" else get_answer_question_llm(model_name = config_llm["rag"]["response_llm"]["model_name"])
    llm = None if response_mode == "no_text" else get_rag_answer_question_model(model_name            = llm_model_config["rag"]["answer_question_llm"]["model_name"],
                                                                                azure_api_version     = llm_model_config["rag"]["answer_question_llm"]["azure_api_version"],
                                                                                azure_deployment_name = llm_model_config["rag"]["answer_question_llm"]["azure_deployment_name"],
                                                                                azure_endpoint        = llm_model_config["rag"]["answer_question_llm"]["azure_endpoint"])
    synthesizer = get_custom_synthesizer(response_mode = response_mode,
                                         llm           = llm)
    
    # 7. Ejecutar la consulta
    response = synthesizer.synthesize(query, nodes)

    # 8. Mostrar respuesta
    print("\nRespuesta generada:\n")
    print(response.response)
    
    # 9. Contar fragmentos por archivo y recolectar rutas
    file_fragment_counts = defaultdict(int)
    file_paths = {}
    for node in response.source_nodes:
        metadata  = node.node.metadata
        file_name = metadata.get('file_name')
        file_path = metadata.get('file_path', "path_desconocido")
        if file_name:
            file_fragment_counts[file_name] += 1
            file_paths[file_name] = file_path
            
    
    # 10. Mostrar resumen de documentos únicos utilizados
    print("\nDocumentos únicos utilizados (con cantidad de fragmentos):\n")
    for file, count in file_fragment_counts.items():
        print(f"- {file}: {count} fragmento(s)")
    
    # 11. Mostrar fragmentos ordenados por score
    print("\nFragmentos recuperados (ordenados por score):\n")
    for i, node in enumerate(response.source_nodes, start = 1):
        metadata  = node.node.metadata
        file_name = metadata.get("file_name", "file_desconocido")
        file_path = metadata.get("file_path", "path_desconocido")
        score     = node.score
        start     = node.node.start_char_idx
        end       = node.node.end_char_idx

        print(f"Fragmento {i}")
        print(f"Archivo                    : {file_name}")
        print(f"Ruta                       : {file_path}")
        print(f"Score                      : {score:.4f}")
        print(f"Rango                      : caracteres {start} a {end}")
        print()

    # 12. Preparar salida como diccionario
    sources_data = []
    for file, count in file_fragment_counts.items():
        sources_data.append({
            "file_name": file,
            "file_path": file_paths.get(file, "path_desconocido"),
            "fragment_count": count
        })

    return {
        "answer": response.response,
        "sources_metadata": sources_data
    }
