import os
# Premium Security Fix: Disable ChromaDB telemetry to ensure 100% offline privacy
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Deep-Tech Fix: Prevent HuggingFace Tokenizer deadlocks when forking processes in Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

class VaultMindRAG:
    def __init__(self, db_dir="./chroma_db", model_name="llama3"):
        self.db_dir = db_dir
        self.model_name = model_name
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model=model_name) 
        self.vector_store = None
        
        self.memory = ConversationBufferWindowMemory(
            k=3, 
            memory_key="chat_history", 
            return_messages=True, 
            output_key="answer"
        )
        self._load_existing_db()

    def update_model(self, new_model_name):
        self.model_name = new_model_name
        self.llm = Ollama(model=new_model_name)

    def clear_memory(self):
        self.memory.clear()

    def _load_existing_db(self):
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            try:
                self.vector_store = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            except Exception as e:
                pass

    def ingest_pdf(self, file_path):
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        # Premium Bug Fix: Filter out empty pages to prevent embedding crashes
        documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if not documents:
            raise ValueError("No extractable text found. This might be a scanned image requiring OCR.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        chunks = text_splitter.split_documents(documents)
        
        # Double check chunks are valid to prevent VectorDB dimension mismatch errors
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        if not chunks:
            raise ValueError("Failed to create valid neural chunks.")
        
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.db_dir)
        else:
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
        
        return len(chunks)

    def ask_question(self, query):
        """Retrieves relevant chunks and generates a conversationally-aware answer."""
        if not query or not query.strip():
            return "⚠️ Please enter a valid question.", []
            
        if not self.vector_store:
            return "⚠️ System Error: No documents indexed in the Vault. Please upload a PDF first.", []
        
        prompt_template = """You are VaultMind, an elite AI research assistant.
        Use the following pieces of context to answer the user's question accurately.
        If the answer is not contained within the context, clearly state that you do not know based on the provided documents.
        Always maintain a professional, academic tone.
        
        Context: {context}
        
        Chat History: {chat_history}
        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        try:
            response = qa_chain({"question": query})
        except Exception as e:
            return f"⚠️ LLM Engine Error: Ensure Ollama is running (`ollama run {self.model_name}`). Details: {e}", []
        
        # Extract cited sources and page numbers safely
        sources = []
        for doc in response.get('source_documents', []):
            file_name = os.path.basename(doc.metadata.get('source', 'Unknown_Document.pdf'))
            page_num = int(doc.metadata.get('page', -1)) + 1 if 'page' in doc.metadata else 'Unknown'
            source_info = f"📄 {file_name} (Page {page_num})"
            if source_info not in sources:
                sources.append(source_info)
                
        return response['answer'], sources
