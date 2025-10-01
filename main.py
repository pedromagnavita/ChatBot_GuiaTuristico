import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURAÇÃO INICIAL ---
LLM_MODEL = "llama3:8b-instruct-q4_0" # Modelo LLM local via Ollama
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Modelo de embedding multilíngue
INDEX_PATH = "faiss_index" # Pasta onde o índice FAISS está salvo

# --- PREPARAÇÃO DO RAG ---
try:
    # Inicializa o modelo LLM local via Ollama
    print("Carregando modelo LLM (Ollama)...")
    llm = ChatOllama(model=LLM_MODEL)
    print("Modelo LLM (Ollama) carregado!")

    # Carrega o índice FAISS localmente
    print(f"Carregando índice vetorial de '{INDEX_PATH}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Banco de dados vetorial carregado!")

except Exception as e:
    print(f"Erro durante a inicialização do modelo. Erro: {e}")
    exit()

# Configura o retriever (busca por similaridade)
retriever = vector_store.as_retriever()
print("Banco de dados vetorial pronto.")

# --- CONFIGURAÇÃO DA CONVERSA (PROMPT E MEMÓRIA) ---
# Define o template do prompt para o chatbot
prompt_template = ChatPromptTemplate.from_template("""
Você é um guia turístico educacional e objetivo.
Sua função é responder às perguntas do usuário **exclusivamente** com base no documento de referência fornecido.

Não invente informações.
                                                   
Não informe que você está seguindo o documento de referência. Apenas responda às perguntas.                                                   

Explique de forma clara, direta e instrutiva.

Se a resposta não estiver no contexto, diga apenas: "Não tenho essa informação no guia."

Histórico da Conversa:
{chat_history}

Contexto Relevante (retirado do guia):
{context}

Pergunta do Usuário:
{input}
""")

# Cria a cadeia de documentos para combinar com o LLM
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Cria a cadeia de recuperação (RAG)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- LOOP PRINCIPAL DO CHATBOT ---
chat_history = []

print("\n--- TurisBot ---")
print("Me pergunte sobre Machu Picchu. Digite 'sair' para terminar.")

while True:
    try:
        # Obtém o input do usuário
        user_input = input("\nVocê: ")
        if user_input.lower() == 'sair':
            break
        
        # Invoca a cadeia de recuperação com o input do usuário e o histórico da conversa
        response = retrieval_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        # Exibe a resposta do chatbot
        answer = response["answer"]
        print(f"Guia: {answer}")

        # Atualiza o histórico da conversa
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

        # Limita o histórico a 6 mensagens (3 de cada) para não exceder o contexto
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

    except KeyboardInterrupt:
        print("\nSaindo...")
        break
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

print("\nChat encerrado.")