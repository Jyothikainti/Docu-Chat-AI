
**Docu Chat AI**

**Overview**

Docu Chat AI is a powerful Streamlit-based application that allows users to upload PDF and DOCX files and interactively chat with the document's content. The app leverages OpenAI's GPT models and FAISS for efficient document retrieval and answering user queries based on the uploaded documents.

**Features**

- 📂 Upload multiple PDF and DOCX files.
- 🧠 Automatic document processing and vector database creation.
- 🔎 AI-powered question-answering from document content.
- 💬 Chat history management for multiple conversations.
- 🚀 Streamlit UI for an interactive and user-friendly experience.

**Installation**

**Prerequisites**

Ensure you have the following installed:

- Python 3.8+
- pip

**Setup**

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Jyothikainti/chat-docu-AI
    cd Docu-chat-AI
    ```

2. **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**

    Create a `.env` file in the project root and add your OpenAI API key:

    ```text
    OPENAI_API_KEY=your_openai_api_key
    ```

**Usage**

1. **Run the Application**

    ```bash
    streamlit run app.py
    ```

2. **Upload Documents**

    Upload one or multiple PDF/DOCX files. The app will process and store them in a vector database.

3. **Ask Questions**

    Type your question in the chat input field. The AI will retrieve relevant content from your uploaded files and generate an answer.

4. **Manage Chat History**

    View previous chat interactions from the sidebar. Start a new chat session when needed.

**File Structure**

```
Docu-chat-AI/
│── app.py               # Main Streamlit application
│── brain.py             # Document processing and vector database creation
│── requirements.txt     # Python dependencies
│── .env                 # Environment variables (not included in repo)
│── README.md            # Project documentation
└── other files & folders
```

**Technologies Used**

- Python
- Streamlit
- OpenAI API
- FAISS (Facebook AI Similarity Search)
- LangChain (for document parsing and chunking)
- PyPDF and python-docx for file processing

**Contributing**

Contributions are welcome! Feel free to submit issues and pull requests on the GitHub repository.

https://github.com/user-attachments/assets/03abcbd6-0291-4447-8da1-b68d7b8f8846
