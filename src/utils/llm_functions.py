import os
from dotenv import load_dotenv
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI


load_dotenv()


def using_azure() -> bool:
    
    return os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

def get_rag_embedding_model(model_name: str,
                            azure_api_version: str,
                            azure_deployment_name: str,
                            azure_endpoint: str):

    if using_azure():

        return AzureOpenAIEmbedding(model_name      = model_name,
                                    api_version     = azure_api_version,
                                    deployment_name = azure_deployment_name,
                                    azure_endpoint  = azure_endpoint,
                                    api_key         = os.getenv("AZURE_OPENAI_API_KEY"))

    return OpenAIEmbedding(model_name = model_name,
                           api_key    = os.getenv("OPENAI_API_KEY"))

def get_rag_answer_question_model(model_name: str,
                                  azure_api_version: str,
                                  azure_deployment_name: str,
                                  azure_endpoint: str):

    if using_azure():

        return AzureOpenAI(model           = model_name,
                           api_version     = azure_api_version,
                           deployment_name = azure_deployment_name,
                           azure_endpoint  = azure_endpoint,
                           api_key         = os.getenv("AZURE_OPENAI_API_KEY"))

    return OpenAI(model   = model_name,
                  api_key = os.getenv("OPENAI_API_KEY"))
