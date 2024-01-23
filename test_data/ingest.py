from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

DATA_PATH="data/"
DB_FAISS_PATH = "vectorstores/db_faiss/"

def create_vector_db():
    documents=[]
    processed_htmls=0
    processed_pdfs=0
    for f in os.listdir("web_data/docs_openvino_ai/2023_2"):
        try:
            if f.endswith(".pdf"):
                pdf_path = './web_data/docs_openvino_ai/2023_2/' + f
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
                processed_pdfs+=1
            elif f.endswith(".html"):
                html_path = './web_data/docs_openvino_ai/2023_2/' + f
                loader = BSHTMLLoader(html_path)
                documents.extend(loader.load())
                processed_htmls+=1
        except Exception as ex:
            print("issue with ",f)
            print(ex)
            pass
    print("Processed",processed_htmls,"html files and ",processed_pdfs,"pdf files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts=text_splitter.split_documents(documents)

    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device':'cpu'})
    print("------------from docs")
    db=FAISS.from_documents(texts,embeddings)
    print("-------------saving")
    db.save_local(DB_FAISS_PATH)
    print("-----------saved")

if __name__=="__main__":
    create_vector_db()