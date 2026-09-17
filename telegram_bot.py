import logging
import asyncio
import nest_asyncio
import json
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import difflib


nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
qa_pairs=None
responses=None
# Load data from vector.json
try:
    with open('./vector.json', 'r') as f:
        data = json.load(f)
    qa_pairs = data.get('qa_pairs', {})
    responses = data.get('responses', {})
    print(qa_pairs)
    print(responses)
except FileNotFoundError:
    logging.error("The file vector.json was not found.")
    qa_pairs = {}
    responses = {}
except json.JSONDecodeError:
    logging.error("Error decoding JSON from vector.json.")
    qa_pairs = {}
    responses = {}



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! Welcome to the setup guide bot.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("""
    The following commands are available:
    
    /start -> Welcome to the bot
    /help -> This message
    /create_vpc -> How to create a VPC network
    /setup_firewall -> How to set up firewall rules
    /create_droplet -> How to create a droplet
    /add_droplet_to_firewall -> How to add a droplet to firewall rules
    /setup_nms -> How to set up NMS
    /log_gcp_console -> How to log onto the GCP Console
    /setup_vpc_gcp -> How to set up a VPC on GCP
    /setup_firewall_policy_gcp -> How to set up a firewall policy on GCP
    /create_vm_gcp -> How to create a VM on GCP
    /vultr_node_setup -> Vultr node setup procedure
    /aws_access_rules -> How to set up access rules in AWS
    /aws_node_registration -> Node registration procedure in AWS
    """)


def find_best_match(user_message, qa_pairs):
    # Normalize the user message to lowercase and strip whitespace
    user_message = user_message.lower().strip()
    
    # Create a list of questions from the qa_pairs dictionary
    questions = list(qa_pairs.keys())
    
    # First, check for an exact match
    if user_message in questions:
        return qa_pairs[user_message]
    
    # If no exact match, use fuzzy matching to find the closest match
    best_match = difflib.get_close_matches(user_message, questions, n=1, cutoff=0.1)
    
    if best_match:
        # If a close match is found, return the corresponding response
        return qa_pairs.get(best_match[0])
    else:
        # If no match is found, return a default response
        return "I'm sorry, I don't have information on that topic. Use /help to see the available commands."
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    response = find_best_match(user_message, qa_pairs)
    await update.message.reply_text(response)

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    command = update.message.text[1:].lower()  # Remove the leading '/' and convert to lowercase
    response = responses[command]
    await update.message.reply_text(response)

async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")  # Fetch token from environment variable
    if token is None:
        logging.error("Bot token is not set. Please set the TELEGRAM_BOT_TOKEN environment variable.")
        return
    
    application = Application.builder().token(token).build()
    
    # Command handlers
    commands = [
        "create_vpc", "setup_firewall", "create_droplet",
        "add_droplet_to_firewall", "setup_nms", "log_gcp_console", "setup_vpc_gcp",
        "setup_firewall_policy_gcp", "create_vm_gcp", "vultr_node_setup",
        "aws_access_rules", "aws_node_registration"
    ]
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    for cmd in commands:
        application.add_handler(CommandHandler(cmd, handle_command))
    
    # Message handler for non-command text
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    await application.run_polling()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())