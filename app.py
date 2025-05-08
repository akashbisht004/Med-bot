from flask import Flask, request, jsonify
from flask_cors import CORS
from vector import docsearch
import os
import dotenv
from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import RunnablePassthrough
from collections import defaultdict
import re
from doctor_recommender import DoctorRecommender
import pandas as pd

dotenv.load_dotenv()

TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"]=os.getenv("TOGETHER_API_KEY")

app = Flask(__name__)
CORS(app)  

# Store conversation history for each user
conversation_history = defaultdict(list)

# Initialize doctor recommender
doctor_recommender = DoctorRecommender("Hospitaldata.xlsx")

def load_patterns_from_file():
    """Load patterns and terms from patterns.txt file."""
    patterns = {
        'JAILBREAK_PATTERNS': [],
        'NON_MEDICAL_PATTERNS': [],
        'MEDICAL_KEYWORDS': [],
        'PROMPT_TEMPLATE': ''
    }
    
    current_section = None
    with open('patterns.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                continue
                
            if current_section in ['JAILBREAK_PATTERNS', 'NON_MEDICAL_PATTERNS']:
                patterns[current_section].append(line)
            elif current_section == 'MEDICAL_KEYWORDS':
                if not line.startswith('#'):  # Skip comments
                    patterns[current_section].append(line)
            elif current_section == 'PROMPT_TEMPLATE':
                patterns[current_section] += line + '\n'
    
    return patterns

# Load patterns from file
PATTERNS = load_patterns_from_file()

def check_jailbreak_attempt(query: str) -> bool:
    """Check if the query contains jailbreak attempts."""
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in PATTERNS['JAILBREAK_PATTERNS'])

def check_non_medical_query(query: str) -> bool:
    """Check if the query is clearly non-medical using improved pattern matching."""
    query_lower = query.lower()
    
    # First, check if query matches any non-medical patterns
    if any(re.search(pattern, query_lower) for pattern in PATTERNS['NON_MEDICAL_PATTERNS']):
        # If it matches a non-medical pattern, check if it also has strong medical context
        medical_context_words = {'doctor', 'hospital', 'clinic', 'medical', 'health', 'treatment', 'symptoms', 'disease', 'condition'}
        has_strong_medical_context = any(word in query_lower for word in medical_context_words)
        
        # If it has strong medical context, don't consider it non-medical
        if has_strong_medical_context:
            return False
            
        return True
        
    return False

def is_medical_query(query: str) -> bool:
    """Classify if a query is medical-related using a more sophisticated approach."""
    query_lower = query.lower()
    words = set(query_lower.split())

    # Always treat these as medical triggers
    symptom_triggers = [
        "symptom", "symptoms", "sign", "signs", "effect", "effects", "cause", "causes",
        "treatment", "treatments", "cure", "cures", "diagnosis", "diagnose", "risk", "risks",
        "what happens if", "what should i do if", "how do i know if", "how to treat", "how to cure",
        "what are", "what is", "how to", "how do", "tell me about", "explain", "describe",
        "common", "typical", "usual", "normal", "regular", "frequent", "often"
    ]
    for trigger in symptom_triggers:
        if trigger in query_lower:
            return True

    # Common medical conditions and diseases
    common_conditions = {
        'flu', 'cold', 'fever', 'covid', 'cancer', 'diabetes', 'asthma', 'arthritis',
        'hypertension', 'migraine', 'headache', 'allergy', 'infection', 'virus',
        'bacteria', 'disease', 'illness', 'condition', 'syndrome', 'scarlet', 'q fever',
        'rheumatic', 'glomerulonephritis', 'measles', 'rash', 'sore throat', 'influenza',
        'common cold', 'stomach flu', 'gastroenteritis', 'bronchitis', 'pneumonia',
        'sinusitis', 'tonsillitis', 'ear infection', 'urinary tract infection'
    }
    
    # Check for any condition in the query
    if any(condition in query_lower for condition in common_conditions):
        return True

    # Medical context words
    medical_context_words = {
        'doctor', 'hospital', 'clinic', 'medical', 'health', 'treatment', 
        'symptoms', 'disease', 'condition', 'patient', 'diagnosis', 'therapy',
        'medicine', 'medication', 'prescription', 'vaccine', 'vaccination',
        'pain', 'ache', 'swelling', 'rash', 'infection', 'virus', 'illness',
        'sick', 'sickness', 'unwell', 'healthy', 'healthcare', 'medical care',
        'prevent', 'prevention', 'cure', 'heal', 'healing', 'recover', 'recovery'
    }
    if any(word in words for word in medical_context_words):
        return True

    # Check for question patterns about medical topics
    question_patterns = [
        r"what (are|is) .*(symptoms|signs|effects|causes|treatments|cures)",
        r"how (do|does|can) .*(treat|cure|prevent|diagnose|heal)",
        r"tell me about .*(symptoms|signs|effects|causes|treatments|cures)",
        r"explain .*(symptoms|signs|effects|causes|treatments|cures)",
        r"describe .*(symptoms|signs|effects|causes|treatments|cures)"
    ]
    if any(re.search(pattern, query_lower) for pattern in question_patterns):
        return True

    # Fallback: check for at least one medical keyword from your patterns
    medical_keyword_count = sum(1 for keyword in PATTERNS['MEDICAL_KEYWORDS'] if keyword.lower() in query_lower)
    return medical_keyword_count >= 1

def format_conversation_history(history):
    """Format conversation history for the prompt."""
    if not history:
        return "No previous conversation."
    formatted_history = "Previous conversation:\n"
    for i, (user_msg, assistant_msg) in enumerate(history):
        formatted_history += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
    return formatted_history

# RAG 
def initialize_rag():
    retriever = docsearch.as_retriever(search_kwargs={"k": 5})  # Increased k for better context
    llm = Together(model="mistralai/Mistral-7B-Instruct-v0.1", api_key=TOGETHER_API_KEY, temperature=0.5, max_tokens=500)
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question", "conversation_history"],
        template=PATTERNS['PROMPT_TEMPLATE']
    )   
    
    # Create the document chain using the new pattern
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # Create the QA chain
    def get_conversation_history(input_dict):
        session_id = input_dict.get("session_id", "default")
        return format_conversation_history(conversation_history.get(session_id, []))
    
    def process_query(input_dict):
        query = input_dict["query"]
        # For follow-up questions, include previous context in the search
        session_id = input_dict.get("session_id", "default")
        history = conversation_history.get(session_id, [])
        
        # Extract key medical terms from the query
        medical_terms = extract_medical_terms(query)
        
        # Build enhanced query
        enhanced_query = query
        if history:
            # Include the last question and answer in the search query
            last_q, last_a = history[-1]
            enhanced_query = f"{last_q} {last_a} {query}"
            
        # Add medical terms to enhance search
        if medical_terms:
            enhanced_query = f"{enhanced_query} {' '.join(medical_terms)}"
            
        # Get relevant documents with increased similarity threshold
        docs = retriever.get_relevant_documents(enhanced_query)
        
        # If no relevant documents found, try a more general search
        if not docs:
            # Try searching with medical terms only
            if medical_terms:
                docs = retriever.get_relevant_documents(' '.join(medical_terms))
            # If still no results, try with first word
            if not docs:
                docs = retriever.get_relevant_documents(query.split()[0])
            
        return {
            "context": docs,
            "question": query,
            "conversation_history": get_conversation_history(input_dict)
        }
    
    qa_chain = process_query | document_chain
    
    return qa_chain

def extract_medical_terms(query: str) -> list:
    """Extract medical terms from the query to enhance search."""
    query_lower = query.lower()
    words = set(query_lower.split())
    
    # Common medical conditions and symptoms
    medical_terms = {
        # Fever types and related conditions
        'fever', 'scarlet fever', 'q fever', 'rheumatic fever', 'typhoid fever',
        'dengue fever', 'yellow fever', 'hay fever', 'feverish', 'pyrexia',
        'hyperthermia', 'febrile', 'afebrile',
        
        # Cancer types
        'cancer', 'tumor', 'malignant', 'benign', 'metastasis', 'carcinoma', 'sarcoma',
        'leukemia', 'lymphoma', 'melanoma', 'breast cancer', 'lung cancer', 'prostate cancer',
        'colon cancer', 'liver cancer', 'pancreatic cancer', 'ovarian cancer',
        
        # Symptoms
        'lump', 'mass', 'swelling', 'pain', 'ache', 'tenderness', 'discharge',
        'bleeding', 'bruising', 'rash', 'fever', 'chills', 'fatigue', 'weakness',
        'nausea', 'vomiting', 'diarrhea', 'constipation', 'cough', 'shortness of breath',
        'sore throat', 'runny nose', 'congestion', 'headache', 'dizziness',
        
        # Body parts
        'breast', 'lung', 'liver', 'kidney', 'heart', 'brain', 'bone', 'blood',
        'lymph', 'node', 'gland', 'tissue', 'organ', 'muscle', 'joint', 'bone',
        'throat', 'nose', 'eye', 'ear', 'skin',
        
        # Medical procedures
        'biopsy', 'mammogram', 'ultrasound', 'x-ray', 'mri', 'ct scan', 'endoscopy',
        'surgery', 'operation', 'treatment', 'therapy', 'medication', 'prescription',
        
        # Medical conditions
        'infection', 'inflammation', 'disease', 'disorder', 'syndrome', 'condition',
        'chronic', 'acute', 'severe', 'mild', 'moderate', 'progressive', 'recurrent',
        'bacterial', 'viral', 'fungal', 'parasitic', 'contagious', 'infectious'
    }
    
    # Find matching medical terms
    found_terms = []
    for term in medical_terms:
        if term in query_lower:
            found_terms.append(term)
            
    return found_terms

# Initialize RAG chain
qa_chain = initialize_rag()

def extract_diagnosis_from_conversation(history: list) -> str:
    """Extract the most recent diagnosis from conversation history."""
    if not history:
        return None
        
    # Look for diagnosis-related keywords in the last few messages
    diagnosis_keywords = ['diagnosis', 'diagnosed', 'condition', 'disease', 'illness']
    for message, response in reversed(history[-3:]):  # Check last 3 exchanges
        for keyword in diagnosis_keywords:
            if keyword in response.lower():
                # Extract the sentence containing the diagnosis
                sentences = response.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()
    return None

@app.route("/chat", methods=["POST"])
def handle_query():
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({"error": "Missing message in request"}), 400
    
    try:
        session_id = data.get("session_id", "default")
        user_message = data["message"]
        
        is_medical = is_medical_query(user_message)
        
        # If it's a medical query, process it regardless of jailbreak attempts
        if is_medical:
            history = conversation_history.get(session_id, [])
            
            is_followup = False
            if history:
                last_user_msg, last_assistant_msg = history[-1]
                followup_indicators = ['it', 'this', 'that', 'these', 'those', 'they', 'them']
                words = user_message.lower().split()
                if any(indicator in words for indicator in followup_indicators):
                    is_followup = True
                    user_message = f"{last_user_msg} {last_assistant_msg} {user_message}"
            
            response = qa_chain.invoke({
                "query": user_message,
                "session_id": session_id
            })
            
            non_medical_response = "I apologize, but I can only answer questions related to medical topics and healthcare. Please ask a medical-related question."
            if response != non_medical_response:
                conversation_history[session_id].append((user_message, response))
                
                diagnosis = extract_diagnosis_from_conversation(conversation_history[session_id])
                if diagnosis:
                    # Get recommendations
                    recommendations = doctor_recommender.get_recommendations(
                        diagnosis=diagnosis
                    )
                    
                    if recommendations['recommendations']:
                        response += "\n\nBased on your diagnosis, here are some recommended healthcare providers:\n"
                        
                        for rec in recommendations['recommendations']:
                            response += f"\nHospital: {rec['hospital_name']}"
                            response += f"\nDoctor: {rec['doctor']}"
                            if pd.notna(rec['category']):
                                response += f"\nSpecialization: {rec['category']}"
                            response += f"\nContact: {rec['contact']}\n"
            
            return jsonify({"response": response})
        
        if check_jailbreak_attempt(user_message):
            return jsonify({
                "response": "I apologize, but I can only answer questions related to medical topics and healthcare. Please ask a medical-related question."
            }), 200
            
        return jsonify({
            "response": "I apologize, but I can only answer questions related to medical topics and healthcare. Please ask a medical-related question."
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True) 