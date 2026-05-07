import streamlit as st
import pandas as pd
#import os
#from dotenv import load_dotenv
#from src.retrieval.index import load_index
#from src.retrieval.retrieve import retrieve
#from src.generation.llm import generate

#load_dotenv()

#COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'switzerland_geo')
#TOP_K = int(os.getenv('TOP_K', 5))

home_page = st.Page("pages/main_page.py", title="GeoRAG")

pg = st.navigation([home_page])

pg.run()
