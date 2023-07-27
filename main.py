import argparse
import timeit
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers


qa_template = """Use the following information to answer the presented question.
Do not make up an answer. If you don't know, say I don't know.
Context: {context}
Question: {question}
Only return the answer below.
Answer:
"""
# CTransformers wrapper for GGML model
llm = CTransformers(model='yourpath/to/model/', # Location of downloaded GGML model
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
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings) #if vectorstore name was changed, update this
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    start = timeit.default_timer() # Start timer

    # setup QA object
    dbqa = setup_dbqa()
    
    # parse input from argparse into QA object
    response = dbqa({'query': args.input})
    end = timeit.default_timer() # End timer

    # print QA response
    print(f'\nAnswer: {response["result"]}')
    print('='*50) # Formatting separator, adjust if needed for display preferences (this outputs to CMD so likely not needed in GUI)

    # display source docs referenced for the response display
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i+1}\n')
        print(f'Source Text: {doc.page_content}')
        print(f'Document Name: {doc.metadata["source"]}')
        print(f'Page Number: {doc.metadata["page"]}\n')
        print('='* 50) # Formatting separator, adjust if needed for display preferences (this outputs to CMD so likely not needed in GUI)
        
    # display time taken for CPU inference, can remove, good for testing in CMD
    print(f"Time to retrieve response: {end - start}")