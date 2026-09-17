import os
import requests
import streamlit as st

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GOOGLE_API_KEY = None

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not configured.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# --------------------------------------------------
# WEBSITE TEXT EXTRACTION
# --------------------------------------------------

def scrape_website(url):
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary elements
        for element in soup(
            ["script", "style", "noscript"]
        ):
            element.decompose()

        return soup.get_text(
            separator="\n",
            strip=True
        )

    except requests.RequestException as e:
        st.error(f"Failed to retrieve website: {e}")
        return ""


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
    )

    return text_splitter.split_text(text)


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


# --------------------------------------------------
# CREATE FAISS VECTOR STORE
# --------------------------------------------------

def get_vector_store(text_chunks):

    embeddings = get_embeddings()

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings,
    )

    vector_store.save_local(
        "faiss_index"
    )


# --------------------------------------------------
# LOAD FAISS VECTOR STORE
# --------------------------------------------------

def load_vector_store():

    if not os.path.exists(
        "faiss_index/index.faiss"
    ):
        return None

    embeddings = get_embeddings()

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

def generate_answer(context, question):

    model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

    prompt = f"""
You are a helpful document-based AI assistant.

Answer the user's question using ONLY the
provided context.

If the answer is not available in the context,
say:

"Answer is not available in the provided context."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.invoke(prompt)

    if isinstance(response.content, list):
        return "\n".join(
        item["text"]
        for item in response.content
        if isinstance(item, dict) and item.get("type") == "text"
    )

    return response.content


# --------------------------------------------------
# HANDLE USER QUESTION
# --------------------------------------------------

def handle_user_input(user_question):

    vector_store = load_vector_store()

    if vector_store is None:

        st.warning(
            "Please upload and process a PDF "
            "or website first."
        )

        return

    docs = vector_store.similarity_search(
        user_question,
        k=4,
    )

    context = "\n\n".join(
        document.page_content
        for document in docs
    )

    answer = generate_answer(
        context,
        user_question,
    )

    st.session_state.chat_history.append(
        {
            "question": user_question,
            "answer": answer,
        }
    )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    st.title("🤖 AI Agent")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ----------------------------------------------
    # SIDEBAR
    # ----------------------------------------------

    with st.sidebar:

        st.header("Knowledge Base")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Process PDFs"):

            if not pdf_docs:

                st.warning(
                    "Please upload at least one PDF."
                )

            else:

                with st.spinner(
                    "Processing documents..."
                ):

                    raw_text = get_pdf_text(
                        pdf_docs
                    )

                    if not raw_text.strip():

                        st.error(
                            "No readable text found."
                        )

                    else:

                        text_chunks = (
                            get_text_chunks(
                                raw_text
                            )
                        )

                        get_vector_store(
                            text_chunks
                        )

                        st.success(
                            f"Processed {len(text_chunks)} chunks."
                        )

        st.divider()

        website_url = st.text_input(
            "Website URL"
        )

        if st.button("Process Website"):

            if not website_url:

                st.warning(
                    "Please enter a website URL."
                )

            else:

                with st.spinner(
                    "Processing website..."
                ):

                    website_text = scrape_website(
                        website_url
                    )

                    if website_text:

                        text_chunks = (
                            get_text_chunks(
                                website_text
                            )
                        )

                        get_vector_store(
                            text_chunks
                        )

                        st.success(
                            f"Processed {len(text_chunks)} chunks."
                        )

    # ----------------------------------------------
    # CHAT INTERFACE
    # ----------------------------------------------

    user_question = st.text_input(
        "Ask a question about your documents"
    )

    if st.button("Submit Question"):

        if user_question.strip():

            handle_user_input(
                user_question
            )

    # ----------------------------------------------
    # CHAT HISTORY
    # ----------------------------------------------

    for chat in st.session_state.chat_history:

        st.markdown(
            f"**You:** {chat['question']}"
        )

        st.markdown(
            f"**AI Agent:** {chat['answer']}"
        )

        st.divider()


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()