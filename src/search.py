import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
def configurate_embeddings(documents):
    embeddings = OpenAIEmbeddings(
      model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return embeddings

def configurate_database_vectorstore(embeddings):
    store = PGVector(
      embeddings=embeddings,
      collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
      connection=os.getenv("PGVECTOR_URL"),
      use_jsonb=True
    )

    return store

def search_for_similarity(store, query):
    """Busca documentos similares usando similarity search com scores."""
    results = store.similarity_search_with_score(query, k=10)

    return results

def create_context(results):
    context = "\n\n---\n\n".join([doc.page_content for doc, score in results])

    return context

def configurate_model_llm():
    """Configura o modelo LLM da OpenAI."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0
    )

    return llm

def create_prompt_template():
    """Cria o template de prompt para o chat."""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt

def search_prompt(question):
    """Função principal que orquestra a busca e resposta usando chains."""
    if not question:
        print("❌ Por favor, forneça uma pergunta!")
        return None

    print(f"\n🔍 Buscando resposta para: {question}")

    # 1. Configurar embeddings
    print("1. Configurando embeddings...")
    embeddings = configurate_embeddings(None)

    # 2. Conectar ao vector store
    print("2. Conectando ao vector store...")
    store = configurate_database_vectorstore(embeddings)

    # 3. Buscar documentos similares
    print("3. Buscando documentos similares...")
    results = search_for_similarity(store, question)
    print(f"✓ {len(results)} documentos encontrados")

    # 4. Criar contexto
    print("4. Criando contexto...")
    context = create_context(results)

    # 5. Configurar LLM
    print("5. Configurando modelo LLM...")
    llm = configurate_model_llm()

    # 6. Criar prompt template
    print("6. Criando prompt...")
    prompt_template = create_prompt_template()

    # 7. Criar e executar a chain
    print("7. Executando chain...")
    chain = (
        {
            "contexto": lambda x: context,
            "pergunta": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    resposta = chain.invoke(question)

    print("\n" + "="*50)
    print("📝 RESPOSTA:")
    print("="*50)
    print(resposta)
    print("="*50 + "\n")

    return resposta


if __name__ == "__main__":
    # Exemplo de uso
    pergunta = input("Digite sua pergunta: ")
    search_prompt(pergunta)
