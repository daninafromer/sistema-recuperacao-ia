# Sistema de Recuperação de Informações (RAG) com FAISS e LangChain

Este projeto implementa um sistema de busca semântica e geração de respostas aumentada por recuperação (RAG - Retrieval-Augmented Generation) utilizando **LangChain**, **FAISS** e **Ollama** rodando localmente. 

O sistema lê um texto sobre Inteligência Artificial, divide o conteúdo em chunks (pedaços menores), gera embeddings para cada chunk, armazena no banco vetorial FAISS e responde a perguntas do usuário com base nos trechos mais relevantes recuperados.

## 🛠️ Tecnologias Utilizadas
- Python 3.10+
- LangChain (langchain-community e langchain-classic)
- FAISS (Facebook AI Similarity Search)
- Ollama (rodando localmente com o modelo `nomic-embed-text` para embeddings e `llama3.2` para geração de texto)

## 🚀 Instalação e Execução

Siga os passos abaixo para reproduzir o projeto na sua máquina.

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/rag-faiss.git
   cd rag-faiss

2. Instale as dependências Python:
   ```bash
   pip3 install langchain langchain-community langchain-ollama langchain-classic faiss-cpu

3. Certifique-se de que o Ollama está instalado e rodando. Baixe os modelos necessários:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2

4. Execute o script:
   ```bash
   python3 rag_system.py

💬 Exemplos de Perguntas e Respostas
Abaixo estão exemplos de interações com o sistema, baseados no arquivo meu_texto.txt.

Pergunta: O que é Deep Learning
Resposta: O Deep Learning é uma subárea do Machine Learning que utiliza redes neurais artificiais com muitas camadas (daí o termo "profundo") para resolver problemas complexos, como reconhecimento de imagens e processamento de linguagem natural.

Pergunta: Quais são os tipos de aprendizado de máquina
Resposta: Existem três tipos principais: aprendizado supervisionado, não supervisionado e por reforço.

Pergunta: O que são embeddings
Resposta: Os embeddings são representações vetoriais de texto. Eles capturam o significado semântico das palavras, permitindo que computadores entendam que "cachorro" e "cão" são similares.

📂 Estrutura do Projeto
rag_system.py: Script principal que contém a lógica de divisão em chunks, criação do banco vetorial e o pipeline do RetrievalQA.

meu_texto.txt: Base de conhecimento de exemplo utilizada para as buscas.
