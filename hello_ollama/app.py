import streamlit as st
from langchain_community.llms import Ollama
import time
import psutil
import json
import pyperclip
import pandas as pd
from io import StringIO

def generate_title(llm, question, response):
    prompt = f"Please provide a brief, catchy title (5 words or less) for this conversation:\nQuestion: {question}\nResponse: {response}"
    return llm.invoke(prompt, temperature=0.7)

def summarize_history(llm, history):
    history_text = "\n".join([f"Title: {entry['title']}\nQuestion: {entry['question']}\nResponse: {entry['response']}" for entry in history])
    prompt = f"Please provide a brief, catchy title (5 words or less) summarizing this entire conversation history:\n{history_text}"
    return llm.invoke(prompt, temperature=0.7)

def update_model():
    st.session_state.llm = Ollama(model=st.session_state.selected_model)

def update_summary_model():
    st.session_state.summary_llm = Ollama(model=st.session_state.selected_summary_model)

def copy_to_clipboard(text):
    pyperclip.copy(text)
    st.success("Copied to clipboard!")

def export_history(format):
    if format == 'json':
        return json.dumps(st.session_state.history, indent=2)
    elif format == 'csv':
        df = pd.DataFrame(st.session_state.history)
        return df.to_csv(index=False)

def main():
    st.title('Ollama Language Model Demo')

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'gemma2'
    if 'selected_summary_model' not in st.session_state:
        st.session_state.selected_summary_model = 'qwen2:0.5b'

    model_options = ['gemma2', 'phi', 'qwen2', 'qwen2:0.5b']

    model_name = st.selectbox(
        'Choose an Ollama Model for Main Interaction:',
        model_options,
        index=model_options.index(st.session_state.selected_model),
        on_change=update_model,
        key='selected_model'
    )

    summary_model_name = st.selectbox(
        'Choose an Ollama Model for Title Summarization:',
        model_options,
        index=model_options.index(st.session_state.selected_summary_model),
        on_change=update_summary_model,
        key='selected_summary_model'
    )
    if 'llm' not in st.session_state:
        st.session_state.llm = Ollama(model=st.session_state.selected_model)
    if 'summary_llm' not in st.session_state:
        st.session_state.summary_llm = Ollama(model=st.session_state.selected_summary_model)

    input_text = st.text_area('Enter your question or code snippet', height=150)

    temperature = st.slider('Temperature (0.0 - 1.0)', 0.0, 1.0, 0.7)
    token_limit = st.number_input('Maximum Tokens', min_value=1, value=1024)
    
    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'history_loaded' not in st.session_state:
        try:
            with open('conversation_history.json', 'r') as f:
                st.session_state.history = json.load(f)
        except FileNotFoundError:
            st.session_state.history = []
        st.session_state.history_loaded = True

    col1, col2, col3 = st.columns(3)
    with col1:
        submit_button = st.button('Submit')
    with col2:
        clear_button = st.button('Clear Input')
    with col3:
        if st.button('Model Info'):
            st.info(f"Main Model: {st.session_state.selected_model}\nSummary Model: {st.session_state.selected_summary_model}\nDetails: [Add model details here]")

    if clear_button:
        input_text = ""
        st.rerun()

    if submit_button:
        if not input_text.strip():
            st.warning("Please enter some text before submitting.")
        else:
            with st.spinner('Generating response...'):
                try:
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    response = st.session_state.llm.invoke(input_text, temperature=temperature)
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    st.subheader('Response:')
                    st.code(response)
                    
                    if st.button('Copy Response'):
                        copy_to_clipboard(response)

                    st.subheader('Performance Metrics:')
                    st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")
                    st.write(f"Memory Usage: {end_memory - start_memory:.2f} MB")

                    title = generate_title(st.session_state.summary_llm, input_text, response)
                    st.session_state.history.append({'title': title, 'question': input_text, 'response': response})

                    with open('conversation_history.json', 'w') as f:
                        json.dump(st.session_state.history, f)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    st.sidebar.header("Conversation History")
    
    if st.session_state.history:
        overall_title = summarize_history(st.session_state.summary_llm, st.session_state.history)
        st.sidebar.subheader(f"Overall: {overall_title}")

    for i, entry in enumerate(st.session_state.history):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button(f"{entry['title']}", key=f"convo_{i}"):
                st.header(f"Conversation: {entry['title']}")
                st.write(f"**Question:** {entry['question']}")
                st.write(f"**Response:** {entry['response']}")
        with col2:
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.history.pop(i)
                with open('conversation_history.json', 'w') as f:
                    json.dump(st.session_state.history, f)
                st.rerun()

    export_format = st.sidebar.selectbox("Export Format", ["JSON", "CSV"])
    if st.sidebar.button("Export History"):
        export_data = export_history(export_format.lower())
        st.sidebar.download_button(
            label="Download Export",
            data=export_data,
            file_name=f"conversation_history.{export_format.lower()}",
            mime="application/json" if export_format == "JSON" else "text/csv"
        )

if __name__ == '__main__':
    main()