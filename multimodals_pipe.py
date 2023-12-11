import os
import openai
from dotenv import load_dotenv
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.prompts import PromptTemplate
import re
import pandas as pd
import numpy as np
from PIL import Image
from FlagEmbedding import FlagModel
import json
from fastapi import FastAPI
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from llama_index import VectorStoreIndex
from fastapi.responses import FileResponse
from pathlib import Path
import base64
import time


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", max_new_tokens=1500,temperature = 0.02
)

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
)

qa_tmpl_str = (
    # """
    # Context information is below.\
    # ---------------------\
    # {context_str}\
    # Given the context information and not prior knowledge, \
    # If the answer to the query can be found by extracting step-by-step information from the provided PDF, please present the response in the following format:

    # Step 1: [Step 1 Text]
    # Step 2: [Step 2 Text]
    # ...
    # Step N: [Step N Text]

    # Ensure that the response adheres strictly to the step-wise structure present in the context of the PDF. If the information is not explicitly laid out in steps, present the answer in a paragraph or a format that aligns with the context information extracted from the PDF.
    # Query: {query_str}\
    # Answer: 
    # """
    """
    Context information is below.\
    ---------------------\
    Context: {context_str}\
    Given the context information and not prior knowledge, \
    If the context mentions step-by-step information in the given context, please present the response in the following format:
    Step 1: [Step 1 Text]
    Step 2: [Step 2 Text]
    ...
    Step N: [Step N Text]
    don't add any unecessary steps or knowledge that does not exist in context.
    Ensure that the response adheres strictly to the step-wise structure if steps are present in the context.
    If the information is not explicitly laid out in steps, present the answer in a paragraph or a format
    that aligns with the context information.
    Query: {query_str}\
    Answer:
    """
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(
     multi_modal_llm=openai_mm_llm, text_qa_template=qa_tmpl
)

def input_user_guery(query_str):
    response = query_engine.query(query_str)
    return response

def response_to_list(response):
    steps_pattern = re.compile(r"Step \d+: .+")
    steps_list = steps_pattern.findall(str(response))
    
    # Wrap each step in the desired format
    formatted_steps = [f"{step}" for step in steps_list]
    
    return formatted_steps

model = FlagModel('BAAI/bge-large-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True)

def load_image(url_or_path):
    # if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
    #     return Image.open(requests.get(url_or_path, stream=True).raw)
    # else:
    try:
        img = Image.open(url_or_path)
        return img
    except FileNotFoundError:
        return None
    

def cosine_similarity(vec1, vec2):
    # Compute the dot product of vec1 and vec2
    dot_product = np.dot(vec1, vec2)

    # Compute the L2 norm of vec1 and vec2
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity

data = pd.read_excel("bank_app_data_embedding_3.xlsx")

def convert_embedding(x):
    try:
        return np.array(json.loads(x))
    except (json.JSONDecodeError, TypeError):
        return np.nan  # or any other value to represent missing data

# Apply the function only to non-empty values in the 'embedding' column
data['embedding'] = data['embedding'].apply(lambda x: convert_embedding(x) if x else np.nan)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def top_products(user_input):
    user_embedding=model.encode(user_input)
    data['scores']=None
    for i,row in data.iterrows():
        data.at[i,'scores']=cosine_similarity(user_embedding,row['embedding'])
    data['scores'] = pd.to_numeric(data['scores'], errors='coerce')
    top_product=data.nlargest(1,'scores') 
    if pd.notna(top_product['image_path'].iloc[0]):
        image_path = top_product['image_path'].iloc[0]
        base64_image = encode_image(image_path)

        return {'image_path' : image_path,'base64_image': base64_image}
        


@app.get("/get_response/{user_guery}")
def get_response(user_guery:str):
    start_time = time.time()

    response = input_user_guery(user_guery)
    steps = response_to_list(str(response))
    list_of_step_image = []
    if steps:
        for step in steps:
            inp = step
            print('nnnn' + step)
            image_data = top_products(inp)
            
            if image_data:
                list_of_step_image.append({"response": step, "image": image_data})
            else:
                list_of_step_image.append({"response": step})

    else: 
      

       elapsed_time = time.time() - start_time
       print(elapsed_time)
       return {"response": str(response), "image": None}
    #    return {"response": str(response), "image": None} 
       

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(elapsed_time)
    return list_of_step_image