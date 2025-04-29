from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from twilio.twiml.messaging_response import MessagingResponse
from langchain.chains import RetrievalQA
from vector import docsearch
import os
import dotenv
from langchain_together import Together

dotenv.load_dotenv()

TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"]=os.getenv("TOGETHER_API_KEY")

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    phone_number = db.Column(db.String(20), unique=True, nullable=False)
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

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.form
    phone_number = data.get("From").split(":")[-1]
    message_body = data.get("Body").strip()

    print(f"Extracted Phone: {phone_number}")
    print(f"RAW Message Body: '{message_body}'")
    resp = MessagingResponse()
    user = User.query.filter_by(phone_number=phone_number).first()

    if not user:
        if "," in message_body:  
            try:
                name, age, location = [x.strip() for x in message_body.split(",")]
                age = int(age)  
                new_user = User(
                    name=name, 
                    age=age, 
                    location=location, 
                    phone_number=phone_number,
                    conversation_state="ready"  
                )
                db.session.add(new_user)
                db.session.commit()

                print(f"User Added: {name}, {age}, {location}, {phone_number}")
                resp.message(f"Thanks {name}! You're registered. Now send your medical question and I'll help you find an answer.")

            except ValueError:
                resp.message(" Invalid format! Send as: Name, Age, Location")
        else:
            resp.message("Welcome! Please send your details in this format:\nName, Age, Location")
        
        return str(resp)

    if user.conversation_state == "ready":
        try:
            result = qa_chain.invoke({"query": message_body})
            answer = result["result"]
            sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
            response_text = f"{answer}\n\nSources:\n" + "\n".join(sources[:2])  
            if len(response_text) > 1500:
                response_text = response_text[:1497] + "..."
            resp.message(response_text)
            
        except Exception as e:
            print(f"RAG ERROR: {str(e)}")
            resp.message("I'm having trouble processing your question. Please try again or ask something else.")
    else:
        user.conversation_state = "ready"
        db.session.commit()
        resp.message(f"Hi {user.name}, what medical question can I help you with today?")
    return str(resp)

@app.route("/reset/<phone>", methods=["GET"])
def reset_user(phone):
    """Admin endpoint to reset a user's state"""
    user = User.query.filter_by(phone_number=phone).first()
    if user:
        db.session.delete(user)
        db.session.commit()
        return f"User {phone} reset successfully"
    return "User not found"

if __name__ == "__main__":
    app.run(debug=True) 