import streamlit as st
import os
from dotenv import load_dotenv
from retrieval.index import load_index
from retrieval.retrieve import retrieve
from generation.llm import generate

load_dotenv()

COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'switzerland_geo')
TOP_K = int(os.getenv('TOP_K', 5))

st.title("GeoRAG - Switzerland Geography Assistant")
st.caption("Ask anything about Swiss geography")

@st.cache_resource
def get_collection():
    try:
        return load_index(COLLECTION_NAME)
    except Exception as e:
        return e

collection = get_collection()
if isinstance(collection, Exception):
    st.error(
        f"ChromaDB collection **{COLLECTION_NAME}** not found. "
        "Run the indexing pipeline first:\n\n"
        "```\ndocker compose run --rm app python -m retrieval.index\n```"
    )
    st.stop()

query = st.text_input("Your question", placeholder="e.g. What is the highest mountain in Switzerland?")

if query:
    with st.spinner("Searching relevant passages..."):
        docs = retrieve(query, collection, TOP_K)

    if not docs:
        st.warning("No relevant passages found for your query.")
    else:
        with st.spinner("Generating answer..."):
            answer = generate(query, docs)

        st.subheader("Answer")
        st.write(answer)

        with st.expander(f"Sources ({len(docs)} passages)"):
            for doc in docs:
                st.markdown(f"**{doc['name']}** — score: `{doc['score']}`")
                st.write(doc['passage'])
                st.divider()

