# Chatbot Guia Turístico Offline com RAG e LLM Local

Este projeto implementa um chatbot de linha de comando que funciona como um guia turístico inteligente e totalmente offline. Ele utiliza um Large Language Model (LLM) rodando localmente e a técnica de Geração Aumentada por Recuperação (RAG) para responder perguntas com base em um documento de texto específico (por exemplo, um guia sobre Machu Picchu).

O principal objetivo é fornecer acesso a informações detalhadas em locais remotos onde o acesso à internet é inexistente ou pouco confiável.

## ✨ Funcionalidades

  - **Totalmente Offline:** Não requer conexão com a internet para funcionar.
  - **LLM Local:** Utiliza o [Ollama](https://ollama.com/) para rodar modelos de linguagem como o Llama 3 diretamente na sua máquina.
  - **Base de Conhecimento Customizável:** Responde perguntas com base no conteúdo de um arquivo de texto que você fornece.
  - **Memória Conversacional:** Mantém o contexto das últimas interações para permitir perguntas de acompanhamento.

## ⚙️ Arquitetura

O projeto é dividido em dois scripts principais:

1.  **`index.py`**: Este é o script de indexação. Ele lê o arquivo de texto (`machu-picchu-guide.txt`), o divide em pedaços, gera os embeddings (vetores numéricos) usando um modelo da Hugging Face e salva o índice vetorial no disco usando FAISS. **Este script é executado apenas uma vez** ou sempre que o documento guia for atualizado.

2.  **`main.py`**: Esta é a aplicação principal do chatbot. Ele carrega o LLM via Ollama e o índice vetorial FAISS já pré-processado. Ele gerencia a interação com o usuário, busca informações relevantes no índice e usa o LLM para gerar respostas contextuais.

## 🚀 Como Usar

Siga os passos abaixo para configurar e executar o projeto.

### 1\. Pré-requisitos

  - [Python](https://www.python.org/downloads/) 3.8 ou superior.
  - [Ollama](https://ollama.com/) instalado e rodando.
  - [Git](https://git-scm.com/) para clonar o repositório.

### 2\. Instalação

Primeiro, clone o repositório e navegue para a pasta do projeto.

```bash
git clone <url-do-seu-repositorio>
cd <nome-da-pasta-do-projeto>
```

É altamente recomendado criar um ambiente virtual para isolar as dependências do projeto.

```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente virtual
# No Windows:
.venv\Scripts\activate
# No macOS/Linux:
source .venv/bin/activate
```

Instale as dependências do `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3\. Configuração dos Modelos (Ollama)

Baixe o modelo através do Ollama. Abra seu terminal e execute:

```bash
# Modelo de Linguagem Principal (LLM)
ollama pull llama3:8b-instruct-q4_0
# Você pode esolher seu modelo, desde que o renomeie no código "LLM_MODEL =..." 
```

### 4\. Preparando o Conteúdo

1.  Crie um arquivo de texto na raiz do projeto chamado `machu-picchu-guide.txt` (ou o nome que preferir, mas lembre-se de atualizar a variável `FILE_PATH` em `index.py`).
2.  Cole todo o conteúdo do seu guia turístico neste arquivo.

### 5\. Execução

O processo é feito em duas etapas:

**Passo 1: Indexar o Conteúdo**
Execute o script `index.py` para processar seu arquivo de texto e criar o banco de dados vetorial.

```bash
python index.py
```

Isso criará uma pasta chamada `meu_faiss_index` no seu diretório. Deve ser executado apenas uma vez, desde que o guia não seja modificado após a execução.

**Passo 2: Iniciar o Chatbot**
Agora você pode conversar com o bot executando o `main.py`.

```bash
python main.py
```

## 🛠️ Customização

  - **Mudar a Fonte de Conhecimento:** Simplesmente altere o conteúdo do arquivo `.txt` e execute `python index.py` novamente para reindexar.
  - **Mudar os Modelos:** Você pode experimentar outros modelos alterando as variáveis `LLM_MODEL` e `EMBEDDING_MODEL` nos arquivos `.py`. Lembre-se de baixar os modelos correspondentes via Ollama ou garantir que a biblioteca correta esteja instalada.

## 💻 Tecnologias Utilizadas

  - **Python**
  - **LangChain:** Framework para desenvolvimento de aplicações com LLMs.
  - **Ollama:** Ferramenta para rodar LLMs localmente.
  - **FAISS:** Biblioteca para busca de similaridade eficiente e banco de dados vetorial.
  - **Hugging Face Sentence Transformers:** Para gerar os embeddings de texto.
  - **PyTorch:** Biblioteca de machine learning.

-----
