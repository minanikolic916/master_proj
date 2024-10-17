import warnings
warnings.filterwarnings("ignore")
#project imports 
from retrieval import *
from evaluation import *
from inference import *
from utils import format_nodes, get_context_node
from vector_store import add_nodes_to_vec_store
#other imports 
import streamlit as st 
import os
import pandas as pd

st.markdown("<h2 style='text-align: center;'>Retrieval Augmented Generation</h2><br>", unsafe_allow_html=True)
question = st.text_input("Postavite pitanje")

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns= ['question', 'contexts', 'answer'])

@st.cache_resource
def get_model_and_tokenizer(model_name):
    model, tokenizer = load_model_and_tokenizer(model_name)
    return model, tokenizer 

@st.cache_resource
def get_pipeline(model_name):
    model, tokenizer = get_model_and_tokenizer(model_name)
    pipe = load_pipeline(model, tokenizer)
    return model, tokenizer, pipe

def return_nodes(question):
    reranked_nodes = rerank_nodes_colbert(query=question, score = 0.5, top_k=1)
    return reranked_nodes


def display_nodes(nodes):
    nodes = format_nodes(nodes)
    for node in nodes:
        st.write(f"**Dokument ID**: {node.id}")
        st.write(f"**Ime fajla**: {node.file_name}")
        st.write(f"**Pribavljeni kontekst**: {node.text}")
        st.write(f"**Score nakon pribavljanja**: {node.score}")
        st.write(f"**Score nakon rerangiranja**: {node.ret_score}")
        st.divider()


model_id = "mistralai/Mistral-7B-Instruct-v0.3"
model, tokenizer, pipe = get_pipeline(model_name= model_id)
model_on = st.toggle("Pribavi odgovor modela")

if(st.button("Pribavi kontekst", type="primary")):
    ret_nodes_with_scores = return_nodes(question=question)
    st.divider()
    display_nodes(ret_nodes_with_scores)
    if model_on:
        context = ret_nodes_with_scores[0].text
        answer = get_model_response(pipe, context, question)
        #answer = get_model_response_v2(model, tokenizer, context, question)
        st.write(answer)
        st.session_state.data = add_to_eval_dataset(st.session_state.data, question, context, answer)  

if(st.sidebar.button("Prikaz RAG metrika", type = "primary", use_container_width=True)):
    st.session_state.data = dataframe_to_dict(st.session_state.data)
    st.sidebar.write("**Format podataka za evaluaciju:**")
    st.sidebar.write(st.session_state.data)
    ragas_metrics_for_all, ragas_pandas = ragas_eval(st.session_state.data)
    st.sidebar.write("**Ragas evaluacija:**")
    st.sidebar.write(ragas_metrics_for_all)
    st.dataframe(ragas_pandas, use_container_width= True)
    #da bi moglo da se radi sa novim podacima 
    for key in st.session_state.keys():
        del st.session_state[key]

    
  
