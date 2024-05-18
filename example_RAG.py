#__all__ = []

from datetime import datetime

import streamlit as st

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import PyPDF2
import os
import json

def extract_text_from_pdf(script,file_name,chunk_list):
    pdf = PyPDF2.PdfReader(script)
    text = ''
    for n,page in enumerate(range(len(pdf.pages))):
        page_obj = pdf.pages[page]
        chunk_list.append(f'{page_obj.extract_text()}, from "{file_name}",page {n}\n')
    return chunk_list


def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation

    return formatted_input  

@st.cache_resource
def load():
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")
    model_id = "nvidia/Llama3-ChatQA-1.5-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,trust_remote_code=True, device_map="auto")
    
    ## load retriever tokenizer and model
    retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
    query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
    context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')

    cwd = os.getcwd()
    print("Current working directory:", cwd)
    if os.path.exists('RAG_dict.json'):
        # Load the dictionary from the file
        with open('RAG_dict.json', 'r') as file:
            chunk_list = json.load(file)
        messages = [{"role": "user", "content": ""}]       
        formatted_input = get_formatted_input(messages, "".join(chunk_list))
        tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)
        print(f"RAG data tokenized count = {len(tokenized_prompt[0])}")
    else:
        folder_path = "Docs"
        chunk_list = []
        pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
        n = 1
        for file_name in pdf_files:
            print(n)
            n+=1
            file_path = os.path.join(folder_path, file_name)
            print(f"Text extracting from {file_name}:")
            chunk_list = extract_text_from_pdf(file_path,file_name,chunk_list)
            print(chunk_list[-1][:128])
            print("-----------------------")
        with open('RAG_dict.json', 'w') as file:
            json.dump(chunk_list, file, indent=4)

    return model_id,tokenizer,model,retriever_tokenizer,query_encoder,context_encoder,chunk_list

def main():
    model_id,tokenizer,model,retriever_tokenizer,query_encoder,context_encoder,chunk_list = load()    

    document = ""

    col1, col2 = st.columns([3, 1])
    with col2:
        
        st.image("fmcv.png", caption="www.fmcv.my")
    with col1:
        st.title("FMCV GenAI")
        st.title("LLAMA3 NVIDIA CHATQA 8B RAG Documents Demo")
    
    # Create an input text box
    input_text = st.text_input("Enter your text", "Who is Fortune Machine Computer? What they doing? What it relation to open source?")

    analyze_button_clicked = st.button("Analyze")
    
    # Create a button to trigger model inference
    if analyze_button_clicked:

        messages = [
                    {"role": "user", "content": input_text}
                   ]
       

        ### running retrieval
        ## convert query into a format as follows:
        ## user: {user}\nagent: {agent}\nuser: {user}
        formatted_query_for_retriever = '\n'.join([turn['role'] + ": " + turn['content'] for turn in messages]).strip()
        print("1")
        query_input = retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt')
        print("2")
        ctx_input = retriever_tokenizer(chunk_list, padding=True, truncation=True, max_length=128, return_tensors='pt')
        print("3")
        query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
        print("4")
        st.write("Searching ...")
        ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
        print("5")
        ## Compute similarity scores using dot product and rank the similarity
        similarities = query_emb.matmul(ctx_emb.transpose(0, 1)) # (1, num_ctx)
        print("6")
        ranked_results = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)
        print("7")
        ## get top-n chunks (n=5)
        retrieved_chunks = [chunk_list[idx] for idx in ranked_results.tolist()[0][:10]]
        print("8")
        context = "\n\n".join(retrieved_chunks)
        print(context)
        print("9")
        st.write("Answering ...")

        formatted_input = get_formatted_input(messages, context)
        tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)
        print(f"tokenized_prompt = {len(tokenized_prompt[0])}")
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=150, eos_token_id=terminators)

        response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))
        st.write("Prediction : ",tokenizer.decode(response, skip_special_tokens=True))

if __name__ == "__main__":
    load()
    main()
