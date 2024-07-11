import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.title("LLaMA 2 Chat App")

model, tokenizer = load_model()

user_input = st.text_input("You:", "")

if user_input:
    response = generate_response(user_input, model, tokenizer)
    st.text_area("LLaMA 2:", value=response, height=200, max_chars=None, key=None)
