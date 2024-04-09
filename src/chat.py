# import chat
import openai

openai.api_key = "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"
pre_train = '''pretend to be a chatbot that fullfils the following criteria:
automate most of the tasks for the Lawyer/CA, 
will be able to solve user’s basic legal and platform support queries verbally as well as in text format
also serve as a legal education channel for users where it will provide answers to law related questions(mostly factual based)
MOreover the lawyer will answer only based on the the questions and should not over-answer.
'''
discussion = [ {"role":"system", "content":pre_train}]

def res(prompt):
   
    # resp = openai.ChatCompletion.create( model="gpt")
    discussion.append({"role":"user", "content":prompt})
    print(discussion)
    resp = openai.chat.completions.create(
        model= "gpt-3.5-turbo",
        messages=discussion
    )
    # for chunk in resp:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")
    result = resp.choices[0].message.content.strip()
    # discussion.pop()
    discussion.append({"role": "assistant", "content": result})
    return result

while(True):
    result = res(input())
    print(result)
