import csv
from langchain.docstore.document import Document 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

docs = []
with open('prod_data.csv', newline="", encoding='utf-8-sig') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        metadata = {col: row[col] for col in row}  # Treat all columns as metadata
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())  # Combine all columns into content
        new_doc = Document(page_content=content, metadata=metadata)
        docs.append(new_doc)

print("Loaded documents:")
for doc in docs:
    print(doc)

splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0, length_function=len)
documents = splitter.split_documents(docs)


for i, doc in enumerate(documents):
    doc.metadata['id'] = f'doc_{i}'

texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]
ids = [doc.metadata['id'] for doc in documents]

embeddings = OllamaEmbeddings(model='prod-model', base_url='http://localhost:11434')
vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, ids=ids)

def ollama_llm(question, context):
    query = f"Product details in CSV form:\n{context}\n\nUser's conversation with the assistant:\n{question}\n\nReturn only the product ID."
    # print(query)
    client = Ollama(base_url='http://localhost:11434', model='prod-model')
    response = client(query)
    return response

# Set up RAG
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

#test
question = '''[('what is gst?', 'GST (Goods and Services Tax) is a comprehensive tax levied on the manufacture, sale, and consumption of goods and services at a national level, aiming to create a single, streamlined tax structure.'), ('what is gst 2B?', 'GST 2B is an auto-drafted input tax credit (ITC) statement that is generated monthly based on the data filed by suppliers in their GSTR-1, GSTR-5, and GSTR-6 forms. Its purpose is to help recipients reconcile and accurately claim their input tax credits.'), ('do startups need gst 2B?', 'Startups may require GST 2B if they are registered under GST and are looking to reconcile and make accurate claims of Input Tax Credit (ITC) based on the data filed by their suppliers in their GSTR-1, GSTR-5, and GSTR-6 forms. It is particularly useful for recipients to ensure correct ITC claims.')]'''
result = rag_chain(question)
print(result)