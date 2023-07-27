# langchain-vector-store
Create FAISS vector db and chunk PDF docs for query by Lllama 2

If don't already have (you likely have the rest used):
pip install langchain
pip install pypdf
pip install faiss-cpu
pip install ctransformers
pip install sentence-transformers

Prepare your PDF docs in single location folder. Update create_db.py with the path to your docs.  Run create_db.py and ensure vectorstore was created. Download your model locally (for this, it should be a GGML model).
Update main.py with all your paths, etc.  Run main.py with a query.  Example: python main.py "What are 3 types of verifications that can be used for Medicaid SSI unearned income?"
