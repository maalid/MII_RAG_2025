from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from src.utils.llm_functions import get_rag_embedding_model
from src.utils.config_functions import load_config


llm_model_config = load_config(config_filename = "llm.yaml")


def print_document(doc: Document):
    
    print("=== DOCUMENTO ===")
    
    print("Texto:")
    print(doc.text[:500])
    
    print("\nMetadata:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    print("=================\n")

def create_persisted_index(docs_dir: str,
                           index_dir: str,
                           index_id: str) -> VectorStoreIndex:

    # Leer documentos desde el directorio y agregar metadatos
    # all_docs = []
    reader    = SimpleDirectoryReader(input_dir = docs_dir, recursive = True)
    documents = reader.load_data()

    # for doc in documents:
    #     doc.metadata = {
    #         "file_name": doc.metadata.get("file_name", "unknown"),
    #         "file_path": os.path.basename(docs_dir)
    #     }
    #     all_docs.append(doc)
    
    print(' ')
    print('documents')
    for doc in documents:
        print_document(doc)

    # Definir el modelo de embeddings
    # embedding_model = get_embed_model(model_name = config_llm["rag"]["embedding_llm"]["model_name"])
    embedding_model = get_rag_embedding_model(model_name            = llm_model_config["rag"]["embedding_llm"]["model_name"],
                                              azure_api_version     = llm_model_config["rag"]["embedding_llm"]["azure_api_version"],
                                              azure_deployment_name = llm_model_config["rag"]["embedding_llm"]["azure_deployment_name"],
                                              azure_endpoint        = llm_model_config["rag"]["embedding_llm"]["azure_endpoint"])
    
    # Crear índice vectorial a partir de los documentos
    index = VectorStoreIndex.from_documents(documents,
                                            embed_model   = embedding_model,
                                            show_progress = True)

    # Persistir el índice
    index.set_index_id(index_id)
    index.storage_context.persist(persist_dir = index_dir)
    print(f"Índice persistido en {index_dir} con ID: {index_id}")

    return index
