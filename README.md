# Sistema RAG - Retrieval Augmented Generation

Sistema de perguntas e respostas baseado em documentos PDF usando RAG (Retrieval Augmented Generation) com LangChain, PostgreSQL + pgvector e OpenAI.

## Arquitetura

O sistema é composto por três módulos principais:

```
┌─────────────────┐
│   ingest.py     │  ← Ingestão de documentos PDF
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │  ← Armazena embeddings vetoriais
│   + pgvector    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   search.py     │  ← Busca semântica + LLM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    chat.py      │  ← Interface interativa
└─────────────────┘
```

## Funcionalidades

- **Ingestão de PDFs**: Processa documentos PDF, divide em chunks e gera embeddings
- **Busca Semântica**: Busca documentos relevantes usando similaridade vetorial
- **Respostas Contextualizadas**: Usa LLM (GPT) para responder perguntas baseadas apenas no contexto recuperado
- **Interface de Chat**: Modo interativo para múltiplas perguntas
- **LangChain Expression Language (LCEL)**: Usa chains modernas com operador pipe `|`

## Requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API da OpenAI

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/abraao2022/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

Copie o arquivo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e configure suas credenciais:

```env
# OpenAI Configuration
OPENAI_API_KEY=sua-chave-aqui
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Google AI (Opcional)
GOOGLE_API_KEY=
GOOGLE_EMBEDDING_MODEL=models/embedding-001

# Database Configuration
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5433/rag
PG_VECTOR_COLLECTION_NAME=PGVECTOR_COLLECTION

# Document Configuration
PDF_PATH=document.pdf
```

### 4. Inicie o PostgreSQL com pgvector

```bash
docker-compose up -d
```

Aguarde alguns segundos para o banco inicializar. Verifique o status:

```bash
docker-compose ps
```

## Uso

### 1. Ingerir Documentos

Primeiro, processe o PDF e armazene no banco vetorial:

```bash
python src/ingest.py
```

Saída esperada:
```
Iniciando ingestão do PDF: document.pdf
1. Carregando PDF...
✓ 34 páginas carregadas
2. Dividindo em chunks...
✓ 67 chunks criados
3. Limpando metadados...
✓ Metadados limpos
4. Gerando embeddings...
✓ Embeddings configurados
5. Salvando no vector store...
✓ 67 documentos salvos no vector store!

✓ Ingestão concluída com sucesso!
```

### 2. Fazer Perguntas (Modo Interativo)

```bash
python src/chat.py
```

Exemplo de uso:
```
🤖 CHAT RAG - Sistema de Perguntas e Respostas
============================================================
📚 Faça perguntas sobre o documento ingerido
💡 Digite 'sair', 'exit' ou 'quit' para encerrar
============================================================

💬 Você: Qual o faturamento da empresa?

🔍 Buscando resposta para: Qual o faturamento da empresa?
1. Configurando embeddings...
2. Conectando ao vector store...
3. Buscando documentos similares...
✓ 5 documentos encontrados
4. Criando contexto...
5. Configurando modelo LLM...
6. Criando prompt...
7. Executando chain...

==================================================
📝 RESPOSTA:
==================================================
[Resposta baseada no contexto do documento]
==================================================
```

### 3. Fazer Perguntas (Modo Único)

Você também pode fazer uma pergunta única via linha de comando:

```bash
python src/chat.py "Qual o faturamento da empresa?"
```

### 4. Busca Direta (Sem Interface)

Para usar apenas o módulo de busca:

```bash
python src/search.py
```

## Estrutura do Projeto

```
mba-ia-desafio-ingestao-busca/
├── src/
│   ├── ingest.py          # Módulo de ingestão de PDFs
│   ├── search.py          # Módulo de busca e resposta (com chains)
│   └── chat.py            # Interface de chat interativo
├── document.pdf           # Documento de exemplo
├── docker-compose.yml     # Configuração do PostgreSQL + pgvector
├── requirements.txt       # Dependências Python
├── .env.example          # Template de variáveis de ambiente
├── .env                  # Variáveis de ambiente (não versionado)
└── README.md             # Este arquivo
```

## Detalhes Técnicos

### Ingestão (ingest.py)

1. **Carregamento**: Usa `PyPDFLoader` para extrair texto do PDF
2. **Chunking**: Divide o texto em chunks de 1000 caracteres (overlap de 150)
3. **Limpeza**: Remove metadados vazios
4. **Embeddings**: Gera vetores usando `text-embedding-3-small` da OpenAI
5. **Armazenamento**: Salva no PostgreSQL com extensão pgvector

### Busca (search.py)

Implementa busca semântica usando **LangChain Expression Language (LCEL)**:

```python
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
```

#### Fluxo:
1. Recebe pergunta do usuário
2. Gera embedding da pergunta
3. Busca top-5 documentos similares no vector store
4. Cria contexto concatenando os documentos
5. Envia contexto + pergunta para o LLM via chain
6. Retorna resposta baseada apenas no contexto

### Chat (chat.py)

- Loop interativo para múltiplas perguntas
- Tratamento de erros e exceções
- Comandos: `sair`, `exit`, `quit`, `q`
- Suporte a pergunta única via argumento CLI

## Configurações Avançadas

### Alterar Tamanho dos Chunks

Edite [src/ingest.py](src/ingest.py#L22-23):

```python
splits = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Tamanho do chunk
    chunk_overlap=150,    # Overlap entre chunks
    add_start_index=False
).split_documents(documents)
```

### Alterar Número de Documentos Recuperados

Edite [src/search.py](src/search.py#L56):

```python
results = store.similarity_search_with_score(query, k=5)  # Altere k
```

### Usar Google Gemini ao invés de OpenAI

1. Configure `GOOGLE_API_KEY` no `.env`
2. Edite [src/search.py](src/search.py#L64-67):

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def configurate_model_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0
    )
    return llm
```

## Troubleshooting

### Erro: "connection failed"

Verifique se o PostgreSQL está rodando:

```bash
docker-compose ps
docker-compose logs postgres
```

### Erro: "OpenAI API key not found"

Verifique se o arquivo `.env` está configurado corretamente:

```bash
cat .env | grep OPENAI_API_KEY
```

### Erro: "No module named 'langchain'"

Reinstale as dependências:

```bash
pip install -r requirements.txt
```

### Banco de dados corrompido

Reinicie o PostgreSQL:

```bash
docker-compose down -v
docker-compose up -d
```

**Atenção**: Isso apagará todos os dados! Você precisará executar `ingest.py` novamente.

## Tecnologias Utilizadas

- **LangChain**: Framework para aplicações com LLMs
- **OpenAI**: API para embeddings e chat completion
- **PostgreSQL**: Banco de dados relacional
- **pgvector**: Extensão para busca vetorial no PostgreSQL
- **PyPDF**: Biblioteca para processamento de PDFs
- **Python dotenv**: Gerenciamento de variáveis de ambiente

## Licença

Este projeto foi desenvolvido como parte do MBA em Engenharia de Software com IA - Full Cycle.

## Autor

Desenvolvido para o desafio de ingestão e busca do MBA Full Cycle.