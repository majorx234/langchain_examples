import json
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings


PDF_JSON_FILE = "pdf_files.json"
PDF_FOLDER = ""
PDF_FILES = []


pdf_json_file = open(PDF_JSON_FILE);
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

documents_chunks_str= []
for document_chunk in documents_chunks:
    documents_chunks_str.append(document_chunk.page_content)

documents_chunks_embeddings = embedding.embed_documents(documents_chunks_str)
print(documents_chunks_embeddings[0])

#for chunk_enbedding in documents_chunks_embeddings[:3]:
#    print(chunk_embedding)
