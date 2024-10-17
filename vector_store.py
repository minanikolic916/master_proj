
from docs_loader import load_data
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.embed_model = embed_model

def init_vector_store(host, port):
    client = qdrant_client.QdrantClient(
        host=host,
        port=port, 
    )
    vector_store = QdrantVectorStore(client= client, collection_name="nova_solarna_energija")
    return vector_store

def add_nodes_to_vec_store(nodes_path:str):
    vector_store = init_vector_store("localhost", 6333)
    nodes = load_data(folder_path=nodes_path)
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    vector_store.add(nodes)
    return vector_store

def search_index(query: str, top_k:int):
    query_embedding = embed_model.get_query_embedding(query)
    query_mode = "default"
    vec_store_q = VectorStoreQuery(
        query_embedding= query_embedding, similarity_top_k=top_k, mode = query_mode
    )
    result = vector_store.query(vec_store_q)
    return result

vector_store = add_nodes_to_vec_store("./data_without_questions")