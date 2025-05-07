from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
from vector import docsearch
import os
import dotenv
from langchain_together import Together

dotenv.load_dotenv()

TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"]=os.getenv("TOGETHER_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# RAG 
def initialize_rag():
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    llm=Together(model="mistralai/Mistral-7B-Instruct-v0.1",api_key=TOGETHER_API_KEY,temperature=0.5,max_tokens=500)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Initialize RAG chain
qa_chain = initialize_rag()

@app.route("/chat", methods=["POST"])
def handle_query():
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({"error": "Missing message in request"}), 400
    
    try:
        result = qa_chain.invoke({"query": data["message"]})
        answer = result["result"]
        
        return jsonify({
            "response": answer
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to process query", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True) 