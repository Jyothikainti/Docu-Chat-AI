import os
import streamlit as st
import openai
from dotenv import load_dotenv
from brain import get_index_for_files

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Failed to retrieve API key. Please check your .env file.")
    st.stop()

openai.api_key = openai_api_key

# Streamlit app title
st.title("📄 DOCU CHAT AI")
st.subheader("Chat with your PDFs and DOCX files effortlessly!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = []

if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None

# Cache function to create vector database
@st.cache_data(show_spinner=False)
def create_vectordb(files, filenames):
    with st.spinner("Creating vector database..."):
        return get_index_for_files([file.getvalue() for file in files], filenames, openai_api_key)

# File upload section
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    try:
        vectordb = create_vectordb(uploaded_files, file_names)
        if vectordb:
            st.session_state["vectordb"] = vectordb
        else:
            st.error("Failed to create a vector database. Ensure your documents contain readable text.")
    except ValueError as e:
        st.error(str(e))

# Sidebar for chat history
st.sidebar.title("📜 Chat History")
for i, chat in enumerate(st.session_state["chat_history"]):
    if st.sidebar.button(f"Chat {i+1}", key=f"chat_{i}"):
        st.session_state["current_chat"] = chat.copy()  # Load selected chat history

# Button to start a new chat
if st.sidebar.button("🆕 New Chat", key="new_chat"):
    if st.session_state["current_chat"]:  # Save current chat before starting a new one
        st.session_state["chat_history"].append(st.session_state["current_chat"].copy())
    st.session_state["current_chat"] = []  # Reset for a new conversation

# Display previous messages
for message in st.session_state["current_chat"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input for user question
question = st.chat_input("Ask a question about your documents")

if question:
    vectordb = st.session_state.get("vectordb")
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("Please upload a document first.")
        st.stop()

    # Search for relevant information
    search_results = vectordb.similarity_search(question, k=3)
    doc_extract = "\n".join([result.page_content for result in search_results])

    system_prompt = f"""
    You are a helpful assistant who answers user questions based on document contexts.
    Your responses must be at least four lines long, providing clear and useful insights.
    Cite the filename and page number when applicable.

    Document Content:
    {doc_extract}
    """


    # Append user question to the current chat
    st.session_state["current_chat"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    # Display assistant response dynamically
    with st.chat_message("assistant"):
        botmsg = st.empty()
        response = []

        for chunk in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}] + st.session_state["current_chat"],
            stream=True,
        ):
            text = chunk.choices[0].get("delta", {}).get("content")
            if text:
                response.append(text)
                botmsg.write("".join(response).strip())

    result = "".join(response).strip()
    st.session_state["current_chat"].append({"role": "assistant", "content": result})

    # Store updated chat in session state correctly
    if st.session_state["current_chat"]:
        if st.session_state["chat_history"]:
            st.session_state["chat_history"][-1] = st.session_state["current_chat"].copy()
        else:
            st.session_state["chat_history"].append(st.session_state["current_chat"].copy())
