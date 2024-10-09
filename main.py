from bot import TelegramBot
from llm_handler import LLMHandler
from llm import GeminiLLM ,TitanLLM, ClaudeLLM
from chroma_text_processing import TextSplitter, RecursiveCharacterTextSplitterAdapter, NLTKTextSplitterAdapter, CustomSentenceTransformerEmbedding, ChromaInterface
from dotenv import load_dotenv
import os



def main():
    # Load environment variables from .env file
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_TOKEN")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    bot = TelegramBot(telegram_token)
    collection_name = "taw_bio"
    db_path = "DB/chroma_db"
    # Instantiate and configure GeminiLLM
    #llm = GeminiLLM(model_name='gemini-1.0-pro-latest')
    #llm.configure(api_key=gemini_api_key)



    # Step 1: Create an instance of the TitanLLM class
    #llm = TitanLLM()
    #llm.configure(region_name='us-west-2', model_id="amazon.titan-text-express-v1")
# Example usage:
    # llm = ClaudeLLM()
    # llm.configure(region_name='us-west-2', model_id="arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-instant-v1")

    llm = GeminiLLM(model_name='gemini-1.0-pro-latest')
    llm.configure(api_key=gemini_api_key)

    # Step 2: Configure the instance with AWS credentials
    llm_handler = LLMHandler(collection_name, db_path,llm,template_name= 'detailed_ar')
    bot.set_llm_handler(llm_handler)
    bot.start()
    bot.run()

if __name__ == "__main__":
    main()