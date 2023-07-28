# langchain-vector-store
Create FAISS vector db and chunk PDF docs for query by Lllama 2 (UI and CMD line versions)

If don't already have (you likely have the rest used):
pip install langchain
pip install pypdf
pip install faiss-cpu
pip install ctransformers
pip install sentence-transformers
pip install streamlit

Prepare your PDF docs in single location folder. Update create_db.py with the path to your docs.  Run create_db.py and ensure vectorstore was created. Download your model locally (for this, it should be a GGML model).

To run with Streamlit UI: use myui.py after updating paths. 'streamlit run myui.py'

To run CMD line: Update main.py with all your paths, etc.  Run main.py with a query.  Example: python main.py "What are 3 types of verifications that can be used for Medicaid SSI unearned income?"
