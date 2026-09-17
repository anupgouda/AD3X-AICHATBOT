import os
import logging
import asyncio

from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_community.vectorstores import FAISS


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ============================================================
# CHECK API KEYS
# ============================================================

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Please add GOOGLE_API_KEY to your .env file."
    )

if not TELEGRAM_BOT_TOKEN:
    raise ValueError(
        "TELEGRAM_BOT_TOKEN is not set. "
        "Please add TELEGRAM_BOT_TOKEN to your .env file."
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# LOAD EXISTING FAISS VECTOR DATABASE
# ============================================================

VECTOR_DB_PATH = "faiss_index"

try:

    new_db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    logger.info("FAISS vector database loaded successfully.")

except Exception as e:

    logger.error("Failed to load FAISS vector database.")
    logger.error(str(e))

    new_db = None


# ============================================================
# GEMINI MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
)

# ============================================================
# START COMMAND
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:

    message = """
Hello! 👋

Welcome to the AD3X AI Setup Guide Bot.

I can answer questions using the available
setup documentation.

You can ask questions like:

• How do I create a VPC in AWS?
• How do I create a subnet?
• How do I configure a security group?
• How do I create an EC2 instance?
• How do I configure DigitalOcean?
• How do I configure GCP?

Just type your question.
"""

    await update.message.reply_text(message)


# ============================================================
# HELP COMMAND
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:

    message = """
Available commands:

/start
Start the bot.

/help
Show this help message.

You can also directly ask questions.

Examples:

How do I create a VPC in AWS?

How do I create a subnet?

How do I create an EC2 instance?

How do I configure AWS?

How do I configure GCP?

How do I configure DigitalOcean?
"""

    await update.message.reply_text(message)


# ============================================================
# CLEAN GEMINI RESPONSE
# ============================================================

def clean_response(response):

    """
    Convert Gemini response into normal text.
    """

    if hasattr(response, "content"):

        content = response.content

        # Gemini may return content as a list
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":

                        text_parts.append(
                            item.get("text", "")
                        )

                elif isinstance(item, str):

                    text_parts.append(item)

            return "\n".join(text_parts)

        return str(content)

    return str(response)


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def ask_question(user_question: str) -> str:

    """
    RAG pipeline:

    User Question
        ↓
    FAISS Similarity Search
        ↓
    Relevant Document Chunks
        ↓
    Gemini
        ↓
    Final Answer
    """

    # --------------------------------------------------------
    # CHECK VECTOR DATABASE
    # --------------------------------------------------------

    if new_db is None:

        return (
            "The knowledge base is currently unavailable. "
            "Please try again later."
        )


    try:

        # ----------------------------------------------------
        # SIMILARITY SEARCH
        # ----------------------------------------------------

        docs = new_db.similarity_search(
            user_question,
            k=4
        )


        # ----------------------------------------------------
        # CHECK RESULTS
        # ----------------------------------------------------

        if not docs:

            return (
                "Answer is not available in the provided context."
            )


        # ----------------------------------------------------
        # CREATE CONTEXT
        # ----------------------------------------------------

        context_text = "\n\n".join(
            doc.page_content
            for doc in docs
        )


        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are an AI assistant for the AD3X setup guide.

Your job is to answer the user's question using ONLY
the information provided in the context.

IMPORTANT RULES:

1. Do not use your own general knowledge.
2. Do not make up information.
3. Do not guess.
4. If the answer is not present in the context,
   respond exactly:

Answer is not available in the provided context.

5. Give a clear and concise answer.
6. If the context contains steps, present them as
   numbered steps.

CONTEXT:

{context_text}


USER QUESTION:

{user_question}


ANSWER:
"""


        # ----------------------------------------------------
        # SEND TO GEMINI
        # ----------------------------------------------------

        response = model.invoke(prompt)


        # ----------------------------------------------------
        # CLEAN RESPONSE
        # ----------------------------------------------------

        answer = clean_response(response)


        if not answer:

            return (
                "Answer is not available in the provided context."
            )


        return answer.strip()


    except Exception as e:

        logger.exception(
            "Error while processing question."
        )

        return (
            "Sorry, I encountered an error while "
            "processing your question."
        )


# ============================================================
# HANDLE NORMAL TELEGRAM MESSAGES
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:

    if not update.message:

        return


    user_message = update.message.text


    logger.info(
        "User question: %s",
        user_message
    )


    # --------------------------------------------------------
    # SHOW PROCESSING MESSAGE
    # --------------------------------------------------------

    processing_message = await update.message.reply_text(
        "🔎 Searching the documentation..."
    )


    # --------------------------------------------------------
    # RUN RAG IN BACKGROUND THREAD
    # --------------------------------------------------------

    answer = await asyncio.to_thread(
        ask_question,
        user_message
    )


    # --------------------------------------------------------
    # DELETE PROCESSING MESSAGE
    # --------------------------------------------------------

    try:

        await processing_message.delete()

    except Exception:

        pass


    # --------------------------------------------------------
    # TELEGRAM MESSAGE LIMIT
    # --------------------------------------------------------

    max_length = 4000


    if len(answer) <= max_length:

        await update.message.reply_text(answer)

    else:

        # Split long responses
        for i in range(
            0,
            len(answer),
            max_length
        ):

            await update.message.reply_text(
                answer[i:i + max_length]
            )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
) -> None:

    logger.error(
        "Exception while handling update:",
        exc_info=context.error
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    logger.info(
        "Starting AD3X Telegram RAG Bot..."
    )


    # --------------------------------------------------------
    # CREATE TELEGRAM APPLICATION
    # --------------------------------------------------------

    application = (
        Application
        .builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )


    # --------------------------------------------------------
    # COMMAND HANDLERS
    # --------------------------------------------------------

    application.add_handler(
        CommandHandler(
            "start",
            start
        )
    )


    application.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )


    # --------------------------------------------------------
    # NORMAL TEXT MESSAGE HANDLER
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )


    # --------------------------------------------------------
    # ERROR HANDLER
    # --------------------------------------------------------

    application.add_error_handler(
        error_handler
    )


    # --------------------------------------------------------
    # START BOT
    # --------------------------------------------------------

    logger.info(
        "Telegram bot is running..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()