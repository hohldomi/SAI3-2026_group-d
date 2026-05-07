import streamlit as st
import os
from transformers import pipeline

#from dotenv import load_dotenv
#from src.retrieval.index import load_index
#from src.retrieval.retrieve import retrieve
#from src.generation.llm import generate

#load_dotenv()

#COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'switzerland_geo')
#TOP_K = int(os.getenv('TOP_K', 5))

st.title("GeoRAG - Switzerland Geography Assistant")
st.caption("Ask anything about Swiss geography")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

query = st.text_input("Your query", value="I love Streamlit")
if query:
    result = model(query)[0]
    st.write(result)
