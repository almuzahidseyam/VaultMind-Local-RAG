import os
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
        # FastEmbed is highly optimized for local CPU/GPU embedding without heavy PyTorch overhead
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        self.llm = Ollama(model=model_name) 
        self.vector_store = None
        
        # Premium Bug Fix: Prevent LLM Context Window Overflow on 8GB VRAM
        # Only remembers the last 3 conversational turns (6 messages) to prevent OOM crashes.
        self.memory = ConversationBufferWindowMemory(
            k=3, 
            memory_key="chat_history", 
            return_messages=True, 
            output_key="answer"
        )
        self._load_existing_db()

    def update_model(self, new_model_name):
        """Dynamically switch the local LLM model."""
        self.model_name = new_model_name
        self.llm = Ollama(model=new_model_name)

    def clear_memory(self):
        """Reset the conversation context."""
        self.memory.clear()

    def _load_existing_db(self):
        """Loads an existing Chroma database if it was previously built."""
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            try:
                self.vector_store = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            except Exception as e:
                print(f"Warning: Could not load existing DB. Starting fresh. ({e})")

    def ingest_pdf(self, file_path):
        """Extracts text from a PDF, chunks it, and ingests it into the Vector Database."""
        # Advanced Parsing with PDFPlumber (Handles tables, columns better than PyPDF2)
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        # Split text into optimal chunks for LLM context windows
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        chunks = text_splitter.split_documents(documents)
        
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.db_dir)
        else:
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
        
        return len(chunks)

    def ask_question(self, query):
        """Retrieves relevant chunks and generates a conversationally-aware answer."""
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
