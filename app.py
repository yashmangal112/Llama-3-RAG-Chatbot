# Author: Safiullah Rahu

# Necessary Libraries
import os

import base64
import gc
import random
import tempfile
import time
import uuid

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, SimpleDirectoryReader

import streamlit as st
from streamlit import _bottom
from st_social_media_links import SocialMediaIcons

from IPython.display import Markdown, display

st.set_page_config(layout="wide")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None



def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
    
# ====== Customise prompt template ======
qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
st.sidebar.title('🦙💬 Llama 3 Chatbot')
st.sidebar.write("This RAG Chatbot is created using the open-source Llama 3 model from Meta.")
selected_model = st.sidebar.selectbox('Choose a Llama model', ['Llama3-7B', 'Llama2-7B'], key='selected_model')
if selected_model == 'Llama3-7B':
    llm_model = "llama3"
elif selected_model == 'Llama2-7B':
    llm_model = "llama2"
st.sidebar.text_area(label="System Prompt", value=qa_prompt_tmpl_str, height=275)

st.sidebar.info("Thank you for your interest in my application. Please note that this is solely a Proof of Concept system, which may include bugs or unfinished features. If you enjoy this app, you can stay updated by following me for news and updates.")
social_media_links = [
    "https://www.linkedin.com/in/safiullahrahu/",
    "https://www.twitter.com/safiullah_rahu",
    "https://github.com/Safiullah-Rahu",
]

social_media_icons = SocialMediaIcons(social_media_links)

social_media_icons.render(sidebar=True)


col1, col2 = st.columns([6, 3])
col11, col22 = st.columns([5, 2])
with col22:
    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Embedding & Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                            loader = SimpleDirectoryReader(
                                input_dir = temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load_data()

                    # setup llm & embedding model
                    llm=Ollama(model=llm_model, request_timeout=120.0)
                    embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    # Creating an index over loaded data
                    Settings.embed_model = embed_model
                    
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    # Create the query engine, where we use a cohere reranker on the fetched nodes
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    
                    
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     



with col1:
    st.header(f"Talk with PDF using Llama-3")

with col2:
    st.button("Clear ↺", on_click=reset_chat)


with col11:

    # Initialize chat history
    if "messages" not in st.session_state:
        reset_chat()


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Accept user input
    if prompt := _bottom.chat_input("What's up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate stream of response with milliseconds delay
            streaming_response = query_engine.query(prompt)
            
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            # full_response = query_engine.query(prompt)

            message_placeholder.markdown(full_response)
            # st.session_state.context = ctx

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})