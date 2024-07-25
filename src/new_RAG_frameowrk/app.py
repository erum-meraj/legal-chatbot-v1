# Import necessary libraries
import openai
from brain import get_index_for_pdf
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO

# Set up the OpenAI API key
openai.api_key = ""

# Function to create a vectordb for the provided PDF files
def create_vectordb(files, filenames):
    vectordb = get_index_for_pdf(
        files, filenames, openai.api_key
    )
    return vectordb

# Read the PDF content from a local file
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    # Extract raw bytes from the Document objects
    raw_bytes = []
    for doc in documents:
        with open(file_path, "rb") as file:
            raw_bytes.append(file.read())
    return raw_bytes

# Define the template for the chatbot prompt
prompt_template = """
    You are a helpful legal Assistant who answers users' questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence is the context of the PDF extract with metadata. 
    
    Carefully focus on the metadata, especially 'filename' and 'page' whenever answering.
        
    Reply "Not applicable" if text is irrelevant.
     
    The PDF content is:
    {pdf_extract}
"""

# Get the file path and names
pdf_file_path = "chatdata.pdf"
pdf_file_names = ["chatdata.pdf"]

# Debugging steps to check the current working directory and file existence
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

if not os.path.isfile(pdf_file_path):
    raise FileNotFoundError(f"The file {pdf_file_path} does not exist in the directory {current_directory}.")

# Read the PDF content
pdf_content = read_pdf(pdf_file_path)

# Create the vectordb
vectordb = create_vectordb(pdf_content, pdf_file_names)

# Main loop for user interaction
while True:
    # Get the user's question
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Update the prompt with the pdf extract
    prompt = [
        {
            "role": "system",
            "content": prompt_template.format(pdf_extract=pdf_extract),
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Call ChatGPT and display the response
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )

    result = response.choices[0].message.content.strip()
    print(result)
