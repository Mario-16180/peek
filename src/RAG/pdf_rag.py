import faiss
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing import List
from langchain_core.documents import Document

from os.path import join
from consts import PDFS_DIRECTORY


def upload_pdf(file):
    """This function uploads a PDF file to the server.

    Args:
        file (_type_): _description_
    """
    with open(join(PDFS_DIRECTORY, file.name), "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path: str) -> list[Document]:
    """This function loads a PDF file and returns a list of documents.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list[Document]: Returns a list of documents.
    """
    documents = PDFPlumberLoader(file_path).load()
    return documents


def split_text(documents) -> List[Document]:
    return RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=150, add_start_index=True, strip_whitespace=True
    ).split_documents(documents)


def index_documents(documents, vector_store):
    # Check if vector_store is a FAISS instance
    if isinstance(vector_store, FAISS):
        texts = [doc.page_content for doc in documents]
        vector_store.add_texts(texts)
    else:
        vector_store.add_documents(documents)


def retrieve_documents(query, vector_store):
    return vector_store.similarity_search(query)


def answer_question(question, documents, template, model):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chained_prompt = prompt | model

    return chained_prompt.invoke({"question": question, "context": context})


def save_faiss_index(vector_store: FAISS, path: str):
    faiss.write_index(vector_store.index, path)


def load_faiss_index(path: str, embeddings: OllamaEmbeddings) -> FAISS:
    index = faiss.read_index(path)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
