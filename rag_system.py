from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# Carregar o texto
loader = TextLoader("meu_texto.txt", encoding="utf-8")
documentos = loader.load()

# Dividir o texto em chunks
# chunk_size=500 e overlap=50 são valores comuns
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documentos)

# Gerar embeddings e criar o banco vetorial FAISS
# Usando o modelo nomic-embed-text que já foi baixado no exercício anterior
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Criar o Retriever  e o modelo de linguagem (LLM)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Busca os 2 chunks mais relevantes
llm = OllamaLLM(model="llama3.2") # Modelo para gerar a resposta final

# Configurar o RetrievalQA (o pipeline completo)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" significa que ele junta os chunks encontrados no prompt
    retriever=retriever,
    return_source_documents=True # para mostrar de onde tirou a resposta
)

# Loop para fazer perguntas
print("\n🤖 Sistema de Recuperação de Informações (RAG) pronto!")
print("Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Sua pergunta: ")
    if pergunta.lower() == "sair":
        break
    
    resultado = qa_chain.invoke({"query": pergunta})
    
    print(f"\n💡 Resposta: {resultado['result']}")
    print("\n📄 Trechos utilizados:")
    for i, doc in enumerate(resultado['source_documents']):
        print(f"  [{i+1}] {doc.page_content[:150]}...") # Mostra um pedaço do chunk
    print("-" * 50)
