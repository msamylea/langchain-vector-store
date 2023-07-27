from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Load PDF file from data path. Convert docx to pdf if needed for word docs
loader = DirectoryLoader('add/you/doc/path/here',
                         glob="*.pdf",
                         loader_cls=PyPDFLoader)
documents = loader.load()

# chunk the pdf text - adjust chunk size and overlap as needed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# get embeddings model from huggingface
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

# Build vector store
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('vectorstore/db_faiss')