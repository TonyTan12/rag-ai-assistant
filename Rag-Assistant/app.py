import streamlit as st
from rag import answer_question

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("RAG Knowledge Assistant")
st.markdown("Ask questions about your uploaded documents.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=4)

# Main input
question = st.text_input("Enter your question:")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        result = answer_question(question, k=k)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Retrieved Sources (Debug)")
    for i, (_, doc, meta) in enumerate(result["retrieved"], start=1):
        st.markdown(f"**S{i}** — {meta.get('source')} (page {meta.get('page')})")
        st.write(doc[:500] + ("..." if len(doc) > 500 else ""))
        st.divider()
