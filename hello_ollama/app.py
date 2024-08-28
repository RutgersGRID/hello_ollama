import streamlit as st
from langchain_community.llms import Ollama
import time
import psutil
import json
import pyperclip
import pandas as pd
from io import StringIO
import ollama

def generate_title(llm, question, response):
    if not llm:
        return "Title Unavailable"
    prompt = f"Please provide a brief, catchy title (5 words or less) for this conversation:\nQuestion: {question}\nResponse: {response}"
    return llm.invoke(prompt, temperature=0.7)

def summarize_history(llm, history):
    if not llm:
        return "History Summary Unavailable"
    history_text = "\n".join([f"Title: {entry['title']}\nQuestion: {entry['question']}\nResponse: {entry['response']}" for entry in history])
    prompt = f"Please provide a brief, catchy title (5 words or less) summarizing this entire conversation history:\n{history_text}"
    return llm.invoke(prompt, temperature=0.7)

def update_model():
    if st.session_state.selected_model:
        st.session_state.llm = Ollama(model=st.session_state.selected_model)
    else:
        st.session_state.llm = None

def update_summary_model():
    if st.session_state.selected_summary_model:
        st.session_state.summary_llm = Ollama(model=st.session_state.selected_summary_model)
    else:
        st.session_state.summary_llm = None

def copy_to_clipboard(text):
    pyperclip.copy(text)
    st.success("Copied to clipboard!")

def export_history(format):
    if format == 'json':
        return json.dumps(st.session_state.history, indent=2)
    elif format == 'csv':
        df = pd.DataFrame(st.session_state.history)
        return df.to_csv(index=False)

@st.cache_data
def get_available_models():
    try:
        models = ollama.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

@st.cache_data
def get_model_details(model_name):
    try:
        return ollama.show(model_name)
    except Exception as e:
        st.error(f"Error fetching model details: {str(e)}")
        return None

def load_history():
    try:
        with open('conversation_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def display_history(history):
    st.sidebar.header("Conversation History")

    max_conversations = 10
    shown_history = history[:max_conversations]

    if shown_history and 'summary_llm' in st.session_state and st.session_state.summary_llm:
        overall_title = summarize_history(st.session_state.summary_llm, shown_history)
        st.sidebar.subheader(f"Overall: {overall_title}")

    for i, entry in enumerate(shown_history):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.sidebar.button(f"{entry['title']}", key=f"convo_{i}"):
                st.session_state.selected_conversation = i
        with col2:
            if st.sidebar.button("Delete", key=f"delete_{i}"):
                st.session_state.history.remove(entry)
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

def main():
    st.set_page_config(page_title="Ollama Language Model Demo", page_icon="🦙", layout="wide")
    st.title('Ollama Language Model Demo')

    # Initialize session state variables
    if 'history' not in st.session_state:
        st.session_state.history = load_history()
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'selected_summary_model' not in st.session_state:
        st.session_state.selected_summary_model = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'summary_llm' not in st.session_state:
        st.session_state.summary_llm = None

    # Check if Ollama is available
    try:
        ollama.list()
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}. Please make sure Ollama is installed and running.")
        st.stop()

    # Get available models
    model_options = get_available_models()

    if not model_options:
        st.error("No models available. Please check your Ollama installation.")
        st.stop()

    # Model selection for main and summary interactions
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.selected_model = st.selectbox(
            'Choose an Ollama Model for Main Interaction:',
            model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            on_change=update_model
        )
    with col2:
        st.session_state.selected_summary_model = st.selectbox(
            'Choose an Ollama Model for Title Summarization:',
            model_options,
            index=model_options.index(st.session_state.selected_summary_model) if st.session_state.selected_summary_model in model_options else 0,
            on_change=update_summary_model
        )

    # Ensure models are initialized
    update_model()
    update_summary_model()

    # Load and display history
    display_history(st.session_state.history)

    # Input and submission handling
    input_text = st.text_area('Enter your question or code snippet', height=150)
    temperature = st.slider('Temperature (0.0 - 1.0)', 0.0, 1.0, 0.7)
    token_limit = st.number_input('Maximum Tokens', min_value=1, value=1024)

    col1, col2, col3 = st.columns(3)
    with col1:
        submit_button = st.button('Submit')
    with col2:
        clear_button = st.button('Clear Input')
    with col3:
        if st.button('Model Info'):
            main_model_details = get_model_details(st.session_state.selected_model)
            summary_model_details = get_model_details(st.session_state.selected_summary_model)

            if main_model_details:
                st.subheader(f"Main Model: {st.session_state.selected_model}")
                st.json(main_model_details)

            if summary_model_details:
                st.subheader(f"Summary Model: {st.session_state.selected_summary_model}")
                st.json(summary_model_details)

    if clear_button:
        st.session_state.input_text = ""
        st.rerun()

    if submit_button:
        if not input_text.strip():
            st.warning("Please enter some text before submitting.")
        elif not st.session_state.llm:
            st.error("Please select a valid model for main interaction.")
        else:
            handle_submission(input_text, temperature)

    # Display selected conversation details
    if 'selected_conversation' in st.session_state:
        display_selected_conversation(st.session_state.selected_conversation)

def handle_submission(input_text, temperature):
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

def display_selected_conversation(i):
    entry = st.session_state.history[i]
    st.header(f"Conversation: {entry['title']}")
    st.write(f"**Question:** {entry['question']}")
    st.write(f"**Response:** {entry['response']}")

if __name__ == '__main__':
    main()