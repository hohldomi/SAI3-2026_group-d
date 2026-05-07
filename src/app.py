import streamlit as st

home_page = st.Page("pages/main_page.py", title="GeoRAG")

pg = st.navigation([home_page])

pg.run()
