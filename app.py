import streamlit as st
import tempfile
import os
from rag_engine import VaultMindRAG
import time
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="VaultMind Local RAG", page_icon="🧠", layout="wide")

# --- Custom Premium CSS ---
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #00FF41;} /* Matrix Green Accent */
    .stChatMessage {background-color: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 10px;}
    .stButton>button {
        border-radius: 8px; font-weight: bold; border: 1px solid #00FF41; 
        background-color: transparent; color: #00FF41; transition: 0.3s;
    }
    .stButton>button:hover {background-color: #00FF41; color: #000000;}
    .source-box {
        background-color: #0d1117; padding: 10px; border-radius: 5px; 
        border-left: 3px solid #00FF41; font-family: monospace; font-size: 0.9em; margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Core Engine ---
@st.cache_resource(show_spinner="Booting Neural Engine & Loading Vector Database...")
def initialize_rag():
    return VaultMindRAG()

rag = initialize_rag()

# --- Main Layout ---
st.title("🧠 VaultMind: Hyper-Local RAG Intelligence")
st.markdown("A completely offline, secure, and locally-hosted AI assistant with conversational memory, powered by your RTX 2060 GPU.")

# --- Sidebar (Ingestion & Settings) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9150/9150095.png", width=120)
    st.title("⚙️ Vault Config")
    
    selected_model = st.selectbox("🤖 Local Neural Model", ["llama3", "mistral", "gemma", "phi3"])
    if selected_model != rag.model_name:
        rag.update_model(selected_model)
        st.success(f"Switched engine to {selected_model}")
        
    if st.button("🗑️ Clear Chat Memory"):
        rag.clear_memory()
        st.session_state.messages = [{"role": "assistant", "content": "Memory wiped. Ready for a new topic.", "sources": []}]
        st.rerun()

    st.markdown("---")
    st.title("📚 Knowledge Vault")
    st.markdown("Upload your study materials or private documents. They are parsed locally using `pdfplumber` and sliced into semantic chunks.")
    
    uploaded_files = st.file_uploader("Ingest High-Res PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 Process Documents", type="primary") and uploaded_files:
        with st.spinner("Ingesting knowledge into Vector Database..."):
            total_chunks = 0
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name
                
                try:
                    chunks = rag.ingest_pdf(temp_path)
                    total_chunks += chunks
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success(f"✅ Successfully ingested {len(uploaded_files)} files into {total_chunks} neural chunks!")

    st.markdown("---")
    st.caption("Engine: LangChain + ChromaDB")
    st.caption("Embeddings: FastEmbed (BGE-Small)")
    
    # Export Chat Feature
    if st.session_state.get("messages") and len(st.session_state.messages) > 1:
        export_str = "\n\n".join([f"**{m['role'].capitalize()}**: {m['content']}" for m in st.session_state.messages])
        st.download_button("📥 Export Chat Log (MD)", export_str, "VaultMind_Chat.md", "text/markdown")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to VaultMind. I have access to your local Vector Database and remember our conversation. Ask me anything.", "sources": []}]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Cited References"):
                for source in msg["sources"]:
                    st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Query your local intelligence..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner(f"Synthesizing Answer via {selected_model}..."):
            answer, sources = rag.ask_question(prompt)
            
            st.markdown(answer)
            if sources:
                with st.expander("📚 Cited References"):
                    for source in sources:
                        st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)
                        
    # Append Assistant Message to History
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
