from langchain_community.llms import Ollama


ollama_int = Ollama(base_url='http://localhost:11434', model='prod-model')

# Use the model to generate a response
discussion = []
products = ""

def get_prod_ID():

    query = '''prduct details: {}

    user's conversation with the assitant: {}

    return only the relavent product ID'''

    query.format(products, discussion)
    response = ollama_int(query)
    print(response)
    return response
