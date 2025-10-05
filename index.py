import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Modelo de embedding multilíngue
file_path = "machu-picchu-guide.txt" # Caminho para o arquivo de texto do guia turístico
INDEX_PATH = "faiss_index"  # Pasta onde o índice FAISS será salvo/carregado

# Verifica se o arquivo de texto existe
if os.path.exists(INDEX_PATH):
        print(f"Índice existente encontrado em '{INDEX_PATH}'. Removendo para recriar...")
        shutil.rmtree(INDEX_PATH) # Remove o índice existente para recriar

# --- PREPARAÇÃO DA INDEXAÇÃO ---
try:
    # Carrega o modelo de embedding multilíngue
    print("Carregando modelo de embedding multilíngue...")
    model_kwargs = {'device': 'cpu'} 
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs
    )
    print("Modelo de embedding carregado.")

except Exception as e:
    print(f"Erro durante a inicialização do modelo de embedding. Erro: {e}")
    exit()

# Carrega o documento de texto
loader = TextLoader(file_path, encoding="utf-8") 
docs = loader.load()

# Divide o documento em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
print(f"O documento foi dividido em {len(split_docs)} chunks.\n")

# Cria o banco de dados vetorial usando FAISS
print("Criando banco de dados vetorial...")
vector_store = FAISS.from_documents(split_docs, embeddings)

# Salva o índice FAISS localmente
vector_store.save_local(INDEX_PATH)
print(f"Índice vetorial criado e salvo na pasta: '{INDEX_PATH}'")