from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

def load_data(folder_path):
    docs = SimpleDirectoryReader(folder_path).load_data()
    text_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=20
    )
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(docs):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = docs[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
    return nodes

def data_info(folder_path):
    nodes = load_data(folder_path)
    print(nodes)


