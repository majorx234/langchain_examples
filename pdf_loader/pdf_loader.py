import json
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

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
    documents.append(PyPDFLoader(file_path))

