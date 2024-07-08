# ollama_app.py

import streamlit as st
from langchain_community.llms import Ollama
import time
import psutil

def main():
    st.title('Ollama Language Model Demo')
    
    model_name = st.text_input('Enter model name (default: gemma2)', 'gemma2')
    input_text = st.text_area('Enter your question or code snippet', height=150)
    
    if st.button('Submit'):
        # Initialize the Ollama model
        start_time = time.time()
        llm = Ollama(model=model_name)
        
        # Measure memory usage before invoking the model
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Invoke the model with the input text
        response = llm.invoke(input_text)
        
        # Measure memory usage after invoking the model
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Measure elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Display the response, timing info, and memory usage
        st.subheader('Response:')
        st.write(response)
        st.subheader('Performance Metrics:')
        st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")
        st.write(f"Memory Usage: {end_memory - start_memory:.2f} MB")
    
if __name__ == '__main__':
    main()
