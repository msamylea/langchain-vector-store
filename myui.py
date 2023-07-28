import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers


st.set_page_config(page_title="MYPAGE", page_icon="", layout="wide", )     
st.markdown(f"""
            <style>
            .stApp {{background-image: url("URLPATH"); 
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)
         
st.title("MYPAGE")

# set prompt template
qa_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

# CTransformers wrapper for GGML model
llm = CTransformers(model='/path/to/model/', # Location of downloaded GGML model
                    model_type='llama', # Model type 
                    config={'max_new_tokens': 256, #adjust as needed
                            'temperature': 0.01}) #adjust as needed
 
 
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question']) #adjust here for context/question if qa_template format is changed
    return prompt


# build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa


# create QA object
def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", #may need HF token, not sure
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local('D:/Llama2 Inference/vectorstore/db_faiss', embeddings) #if vectorstore name was changed, update this
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
    return dbqa

if __name__ == "__main__":
    # setup QA object
    dbqa = setup_dbqa()
   
    question = st.text_input("What would you like to search for in the documents?",)    
    if question:
        response = dbqa({'query': question})        
        st.write(response)        
        
        source_docs = response['source_documents']
        for i, doc in enumerate(source_docs):
            st.write(f'\nREFERENCE DOCUMENTS: {i+1}\n')
            st.write(f'\nSource Document {i+1}\n')
            st.write(f'Source Text: {doc.page_content}')
            st.write(f'Document Name: {doc.metadata["source"]}')
            st.write(f'Page Number: {doc.metadata["page"]}\n')
       
    
        
