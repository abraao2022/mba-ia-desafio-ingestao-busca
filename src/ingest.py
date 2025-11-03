import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

def load_pdf(pdf_path=PDF_PATH):
    """Carrega o PDF e retorna os documentos."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_pdf(documents):
    """Divide os documentos em chunks menores."""
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False
    ).split_documents(documents)

    if not splits:
        raise Exception("Nenhum chunk foi gerado.")
    return splits

def clear_metadata(splits):
    """Limpa metadados vazios dos documentos."""
    enriched = [
        Document(
            page_content=doc.page_content,
            metadata={
                key: value
                for key, value in doc.metadata.items()
                if value not in ("", None)
            }
        )
        for doc in splits
    ]

    return enriched

def generate_embeddings(enriched):
    """Gera embeddings e IDs para os documentos."""
    ids = [f"doc-{i}" for i in range(len(enriched))]
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    return embeddings, ids

def save_to_vectorstore(enriched, embeddings, ids):
    """Salva os documentos e embeddings no PGVector."""
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True
    )

    store.add_documents(documents=enriched, ids=ids)
    print(f"✓ {len(enriched)} documentos salvos no vector store!")

def ingest_pdf():
    """Função principal que orquestra todo o processo de ingestão."""
    print(f"Iniciando ingestão do PDF: {PDF_PATH}")

    # 1. Carregar PDF
    print("1. Carregando PDF...")
    documents = load_pdf()
    print(f"✓ {len(documents)} páginas carregadas")

    # 2. Dividir em chunks
    print("2. Dividindo em chunks...")
    splits = split_pdf(documents)
    print(f"✓ {len(splits)} chunks criados")

    # 3. Limpar metadados
    print("3. Limpando metadados...")
    enriched = clear_metadata(splits)
    print(f"✓ Metadados limpos")

    # 4. Gerar embeddings
    print("4. Gerando embeddings...")
    embeddings, ids = generate_embeddings(enriched)
    print(f"✓ Embeddings configurados")

    # 5. Salvar no vector store
    print("5. Salvando no vector store...")
    save_to_vectorstore(enriched, embeddings, ids)

    print("\n✓ Ingestão concluída com sucesso!")


if __name__ == "__main__":
    ingest_pdf()