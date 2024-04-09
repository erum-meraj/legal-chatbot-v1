from flask import Flask, render_template, request, jsonify
import flask_cors 
# import chat
import openai
import sys
import os
import re
from pathlib import Path
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage
)
model_name = "gpt-3.5-turbo"
def generate_iso_date():
    current_date = datetime.now()
    return re.sub(r"\.\d+$", "", current_date.isoformat().replace(':', '')) 
class ChatFile:
    def __init__(self, current_file: Path, model_name: str) -> None:
        self.current_file = current_file
        self.model_name = "gpt-3.5-turbo"
        print(f"Writing to file {current_file}")
        with open(self.current_file, 'w') as f:
            f.write(f"Langchain Session at {generate_iso_date()} with {self.model_name}\n\n")
    
    def store_to_file(self, question: str, answer: str):
        print(f"{answer}")
        with open(self.current_file, 'a') as f:
            f.write(f"{generate_iso_date()}:\nQ: { question}\nA: {answer}\n\n")

# Create a chat file
chat_file = ChatFile(Path(f"{model_name}_{generate_iso_date()}.txt"), model_name)


chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
openai.api_key = "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"
pre_train = '''pretend to be a chatbot that fullfils the following criteria:
automate most of the tasks for the Lawyer/CA, 
will be able to solve user’s basic legal and platform support queries verbally as well as in text format
also serve as a legal education channel for users where it will provide answers to law related questions(mostly factual based)
MOreover the lawyer will answer only based on the the questions and should not over-answer.
'''
discussion = [ {"role":"system", "content":pre_train}]

def res(prompt):
    # discussion.append({"role":"user", "content":prompt})
    # # print(discussion)
    # resp = openai.chat.completions.create(
    #     model= "gpt-3.5-turbo",
    #     messages=discussion
    # )
    # # for chunk in resp:
    # #     if chunk.choices[0].delta.content is not None:
    # #         print(chunk.choices[0].delta.content, end="")
    # result = resp.choices[0].message.content.strip()
    # # discussion.pop()
    # discussion.append({"role": "assistant", "content": result})
    # return result

    resp = chat([HumanMessage(content=prompt)])
    answer = resp.content
    chat_file.store_to_file(prompt, answer)
app = Flask(__name__)
flask_cors.CORS(app)

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = res(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run()