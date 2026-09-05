import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class VaultMindRAG:
    def __init__(self, db_dir="./chroma_db"):
        self.db_dir = db_dir
        # FastEmbed is highly optimized for local CPU/GPU embedding without heavy PyTorch overhead
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # We assume Ollama is running locally with Llama-3 or Mistral. 
        # For an RTX 2060 8GB, Llama-3 8B (4-bit quantized) runs perfectly.
        self.llm = Ollama(model="llama3") 
        self.vector_store = None
        self._load_existing_db()

    def _load_existing_db(self):
        """Loads an existing Chroma database if it was previously built."""
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            try:
                self.vector_store = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            except Exception as e:
                print(f"Warning: Could not load existing DB. Starting fresh. ({e})")

    def ingest_pdf(self, file_path):
        """Extracts text from a PDF, chunks it, and ingests it into the Vector Database."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split text into optimal chunks for LLM context windows
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.db_dir)
        else:
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
        
        return len(chunks)

    def ask_question(self, query):
        """Retrieves relevant chunks and generates an answer using the Local LLM."""
        if not self.vector_store:
            return "⚠️ System Error: No documents indexed in the Vault. Please upload a PDF first.", []
        
        prompt_template = """
        You are VaultMind, an elite, hyper-intelligent AI research assistant.
        Use the following pieces of context to answer the user's question accurately.
        If the answer is not contained within the context, clearly state that you do not know based on the provided documents.
        Always maintain a professional, academic tone.
        
        Context: {context}
        
        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        try:
            response = qa_chain({"query": query})
        except Exception as e:
            return f"⚠️ LLM Engine Error: Ensure Ollama is running (`ollama run llama3`). Details: {e}", []
        
        # Extract cited sources and page numbers
        sources = []
        for doc in response.get('source_documents', []):
            file_name = os.path.basename(doc.metadata.get('source', 'Unknown_Document.pdf'))
            page_num = int(doc.metadata.get('page', -1)) + 1 if 'page' in doc.metadata else 'Unknown'
            source_info = f"📄 {file_name} (Page {page_num})"
            if source_info not in sources:
                sources.append(source_info)
                
        return response['result'], sources
