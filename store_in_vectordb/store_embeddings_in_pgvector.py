import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_postgres import PGEngine

PDF_JSON_FILE = "pdf_files.json"
PDF_FOLDER = ""
PDF_FILES = []


pdf_json_file = open(PDF_JSON_FILE)
pdf_file_info = json.load(pdf_json_file)
PDF_FOLDER = pdf_file_info["pdf_folder"]
PDF_FILES = pdf_file_info["pdf_files"]
documents = []

for file in PDF_FILES:
    file_path = PDF_FOLDER + '/' + file
    print("load file: {}".format(file_path))
    documents.append(PyPDFLoader(file_path).load())

documents_chunks = []
for document in documents:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    ).split_documents(document)
    for chunk in chunks:
        documents_chunks.append(chunk)
print("chunk size: {}", len( documents_chunks))

embedding = OllamaEmbeddings(model="bge-large",  base_url="http://localhost:8080/")

PGVECTOR_JSON_FILE = "pgvector.json"
pgvector_json_file = open(PGVECTOR_JSON_FILE)
pgvector_file_info = json.load(pgvector_json_file)

POSTGRES_USER = pgvector_file_info["POSTGRES_USER"]  # @param {type: "string"}
POSTGRES_PASSWORD = pgvector_file_info["POSTGRES_PASSWORD"]  # @param {type: "string"}
POSTGRES_HOST = pgvector_file_info["POSTGRES_HOST"]  # @param {type: "string"}
POSTGRES_PORT = pgvector_file_info["POSTGRES_PORT"]  # @param {type: "string"}
POSTGRES_DB = pgvector_file_info["POSTGRES_DB"]  # @param {type: "string"}
CONNECTION_STRING = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
    f":{POSTGRES_PORT}/{POSTGRES_DB}"
)

COLLECTION_NAME="rag_collection"
db = PGVector.from_documents(
    embedding=embedding,
    documents=documents_chunks,
    connection=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    use_jsonb=True,
    pre_delete_collection=False,
)
