from flask import Flask, render_template, request, jsonify
import flask_cors 
import openai
import sys
import os
import re
import langchain
import json
import langchain.embeddings
import langchain_community
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama


#first run the modelfile to run the final model locally 
#local product model
# ollama create prod-model -f chat_model
ollama_int = Ollama(base_url='http://localhost:11434', model='prod-model')



default_prompt = """You are a helpful legal Assistant who answers users' questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence is the context of the PDF extract with metadata. 
        
    Reply "Not relevant" if text is irrelevant to the field of legal processes. 

    give the answer in plain text
    
    """
products = '''product_ID, product_Name, product_Description
1, gst_service, provide the service to make GST registration forms for enterprises
2, PAN_service, provide assistance with PAN registration of new enterprises
3, agreement_drafting, provide assitance with amking different kind of agreements like NDA and intellectual property and other papers'''


os.environ["OPENAI_API_KEY"] = ""
loader = PyPDFLoader("chatdata.pdf")
pages = loader.load_and_split()
chunks = pages
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, verbose=True)
qa = ConversationalRetrievalChain.from_llm(llm = chat_model, retriever=db.as_retriever())

discussion = []


def get_prod_ID():
    temp_prod = json.dumps(discussion)
    query = '''product details in CSV form: \n''' + products +  '''\n\nuser's conversation with the assistant:''' + temp_prod + '''\n\nreturn only the product ID'''
    print(query)
    response = ollama_int(query)
    print(response)
    return response


def res(prompt):
    result = qa({"question": prompt, "chat_history":discussion, "system": default_prompt})
    # print(result)
    discussion.append((prompt, result['answer']))
    # print(discussion)
    return result['answer']
    
app = Flask(__name__)
flask_cors.CORS(app)

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid]
    prod = "0"
    if (len(discussion) >= 3):
        prod = get_prod_ID()
        print(prod)

    response = res(text)
    message = {"answer": response, "prod_ID": prod}
    return jsonify(message)

@app.post("/transcribe")
def transcribe():
    vid = request.get_json().get("message")


if __name__ == "__main__":
    app.run()