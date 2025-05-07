from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
from vector import docsearch
import os
import dotenv
from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import RunnablePassthrough

dotenv.load_dotenv()

TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"]=os.getenv("TOGETHER_API_KEY")

app = Flask(__name__)
CORS(app)  

def is_medical_query(query: str) -> bool:
    """Classify if a query is medical-related."""
    medical_keywords = [
        'health', 'medical', 'disease', 'illness', 'symptom', 'treatment', 'medicine', 'doctor',
        'patient', 'hospital', 'clinic', 'diagnosis', 'therapy', 'surgery', 'prescription',
        'vaccine', 'infection', 'virus', 'bacteria', 'pain', 'fever', 'cough', 'headache',
        'blood', 'heart', 'lung', 'brain', 'cancer', 'diabetes', 'hypertension', 'asthma',
        'allergy', 'immune', 'vaccination', 'pharmacy', 'drug', 'medication', 'dose',
        'side effect', 'recovery', 'rehabilitation', 'prevention', 'screening', 'test',
        'scan', 'x-ray', 'mri', 'ct', 'ultrasound', 'biopsy', 'surgery', 'operation',
        'emergency', 'urgent', 'acute', 'chronic', 'condition', 'syndrome', 'disorder'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in medical_keywords)

# RAG 
def initialize_rag():
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    llm = Together(model="mistralai/Mistral-7B-Instruct-v0.1", api_key=TOGETHER_API_KEY, temperature=0.5, max_tokens=500)
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a medical assistant specialized in providing medical information and advice. 
IMPORTANT: You must ONLY answer questions related to medical topics, health conditions, treatments, and healthcare.
If the question is not medical-related or if you cannot find relevant medical information in the context, respond with:
"I apologize, but I can only answer questions related to medical topics and healthcare. Please ask a medical-related question."

When answering medical questions:
1. Only use information from the provided context
2. Be clear and concise
3. Include relevant medical terms when appropriate
4. If the context doesn't contain enough information, say "I don't have enough medical information to answer this question accurately."

Context: {context}
Question: {question}
Answer:"""
    )   
    
    # Create the document chain using the new pattern
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # Create the QA chain
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | document_chain
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
        # Check if the query is medical-related
        if not is_medical_query(data["message"]):
            return jsonify({
                "response": "I apologize, but I can only answer questions related to medical topics and healthcare. Please ask a medical-related question."
            }), 200
            
        result = qa_chain.invoke(data["message"])
        
        return jsonify({
            "response": result
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to process query", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True) 