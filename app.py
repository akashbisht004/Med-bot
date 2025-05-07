from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
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

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    conversation_state = db.Column(db.String(20), default="registration")  

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

# Create database 
with app.app_context():
    db.create_all()

# Initialize RAG chain
qa_chain = initialize_rag()

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    
    if not all(k in data for k in ["name", "age", "location", "email"]):
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        age = int(data["age"])
        new_user = User(
            name=data["name"],
            age=age,
            location=data["location"],
            email=data["email"],
            conversation_state="ready"
        )
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            "message": "Registration successful",
            "user": {
                "name": new_user.name,
                "email": new_user.email
            }
        }), 201
        
    except ValueError:
        return jsonify({"error": "Invalid age format"}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Registration failed", "details": str(e)}), 400

@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    
    if not data or "query" not in data or "email" not in data:
        return jsonify({"error": "Missing query or email"}), 400
    
    user = User.query.filter_by(email=data["email"]).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    try:
        result = qa_chain.invoke({"query": data["query"]})
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        
        response = {
            "answer": answer,
            "sources": sources[:2]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to process query", "details": str(e)}), 500

@app.route("/api/reset/<email>", methods=["DELETE"])
def reset_user(email):
    """Admin endpoint to reset a user's state"""
    user = User.query.filter_by(email=email).first()
    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": f"User {email} reset successfully"}), 200
    return jsonify({"error": "User not found"}), 404

if __name__ == "__main__":
    app.run(debug=True) 