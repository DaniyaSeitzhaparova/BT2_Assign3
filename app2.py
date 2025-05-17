import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os
import time
import json
from datetime import datetime

st.set_page_config(page_title="Constitutional Assistant of Kazakhstan", page_icon="📚")
st.title("Kazakhstan Constitution Assistant")

def save_chat_history():
    """Saves chat history to JSON file"""
    if not st.session_state.messages:
        return
    
    if not os.path.exists("./chat_history"):
        os.makedirs("./chat_history")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./chat_history/chat_{timestamp}.json"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {str(e)}")

def load_latest_chat_history():
    """Loads latest chat history if exists"""
    if not os.path.exists("./chat_history"):
        return None
    
    try:
        files = [f for f in os.listdir("./chat_history") if f.startswith("chat_") and f.endswith(".json")]
        if files:
            latest_file = max(files)
            with open(f"./chat_history/{latest_file}", "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load history: {str(e)}")
    return None

def check_ollama_availability():
    try:
        try:
            st.session_state.embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBED_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            st.sidebar.success(f"Embedding model '{OLLAMA_EMBED_MODEL}' available")
        except Exception as e:
            st.sidebar.warning(f"Primary embedding model unavailable: {str(e)[:100]}...")
            try:
                st.session_state.embeddings = OllamaEmbeddings(
                    model="llama2",
                    base_url=OLLAMA_BASE_URL
                )
                st.sidebar.success("Using fallback model 'llama2' for embeddings")
            except Exception as fallback_e:
                st.sidebar.error(f"Fallback model also unavailable: {str(fallback_e)[:100]}...")
                raise

        try:
            st.session_state.llm = ChatOllama(
                model=OLLAMA_CHAT_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0
            )
            st.sidebar.success(f"Chat model '{OLLAMA_CHAT_MODEL}' available")
            st.session_state.ollama_setup_valid = True
        except Exception as e:
            st.sidebar.error(f"Chat model unavailable: {str(e)[:100]}...")
            st.session_state.ollama_setup_valid = False
            raise

    except Exception as e:
        st.session_state.ollama_setup_valid = False
        st.sidebar.error(f"Ollama connection error. Ensure server is running and models are downloaded.")
        st.sidebar.code(f"Try running: ollama pull {OLLAMA_EMBED_MODEL} && ollama pull {OLLAMA_CHAT_MODEL}")

def process_document(file):
    tmp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_file_path)
        else:
            st.error("Only PDF and DOCX files are supported")
            return None

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        return splits
    except Exception as e:
        st.error(f"Error processing file {file.name}: {str(e)[:200]}")
        return None
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def process_uploaded_files():
    if not st.session_state.uploaded_files:
        return

    if not st.session_state.ollama_setup_valid:
        st.warning("Please configure Ollama connection first")
        return

    with st.spinner("Processing documents..."):
        all_splits = []
        processed_files = []
        
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(st.session_state.uploaded_files):
            try:
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_files), 
                                   text=f"Processing {uploaded_file.name}...")
                splits = process_document(uploaded_file)
                if splits:
                    all_splits.extend(splits)
                    processed_files.append(uploaded_file.name)
                time.sleep(0.1)  
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)[:200]}")

        if all_splits:
            try:
                if st.session_state.vectorstore:
                    del st.session_state.vectorstore
                
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=st.session_state.embeddings,
                    persist_directory="./chroma_db"
                )
                
                st.success(f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}")
                st.session_state.processed_files = processed_files
            except Exception as e:
                st.error(f"Error creating vector store: {str(e)[:200]}")
                st.session_state.vectorstore = None
        else:
            st.error("Failed to process any documents")

with st.sidebar:
    st.header("Ollama Settings")
    OLLAMA_BASE_URL = st.text_input(
        "Ollama Server Address", 
        "http://localhost:11434",
        help="Typically http://localhost:11434"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        OLLAMA_EMBED_MODEL = st.selectbox(
            "Embedding Model",
            ["nomic-embed-text", "llama2", "mxbai-embed-large"],
            index=0
        )
    with col2:
        OLLAMA_CHAT_MODEL = st.selectbox(
            "Chat Model",
            ["llama2", "mistral", "gemma:7b"],
            index=0
        )

    if st.button("Test Ollama Connection"):
        with st.spinner("Checking connection..."):
            check_ollama_availability()

    if st.button("Save Chat History"):
        save_chat_history()
        st.success("Chat history saved!")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = load_latest_chat_history() or []
if "ollama_setup_valid" not in st.session_state:
    st.session_state.ollama_setup_valid = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

st.header("Document Upload")
uploaded_files = st.file_uploader(
    "Select Constitution files (PDF/DOCX)",
    type=['pdf', 'docx'],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents", key="process_btn"):
        st.session_state.uploaded_files = uploaded_files
        process_uploaded_files()

st.header("Chat with Assistant")

if st.session_state.ollama_setup_valid and st.session_state.vectorstore:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Information Sources"):
                    for i, (source, score) in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** (similarity: {score:.2f}):")
                        st.text(source[:500] + ("..." if len(source) > 500 else ""))

    if prompt := st.chat_input("Ask a question about Kazakhstan Constitution"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    return_source_documents=True
                )

                with st.spinner("Searching for answer..."):
                    response = qa_chain.invoke({"query": prompt})
                    result_text = response.get("result", "Failed to get answer.")
                    source_docs = response.get("source_documents", [])

                    st.write(result_text)

                    sources_with_scores = []
                    if source_docs:
                        with st.expander("Answer Sources"):
                            for i, doc in enumerate(source_docs):
                                score = doc.metadata.get("score", 0) if hasattr(doc, "metadata") else 0
                                sources_with_scores.append((doc.page_content, score))
                                st.markdown(f"**Source {i+1}** (similarity: {score:.2f}):")
                                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result_text,
                        "sources": sources_with_scores
                    })
                
                save_chat_history()

            except Exception as e:
                error_msg = f"Error: {str(e)[:200]}" + ("..." if len(str(e)) > 200 else "")
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

elif not st.session_state.ollama_setup_valid:
    st.warning("Please configure Ollama connection in the sidebar first")
elif not st.session_state.vectorstore:
    st.info("Please upload and process documents to begin")