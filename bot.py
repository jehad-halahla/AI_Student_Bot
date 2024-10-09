import telebot

class TelegramBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        self.llm_handler = None

    def set_llm_handler(self, llm_handler):
        self.llm_handler = llm_handler

    def start(self):
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            self.bot.reply_to(message, "Welcome! I'm a bot powered by an LLM. Send me a message, and I'll generate a response.")

        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            if self.llm_handler:
                try:
                    response = self.llm_handler.generate_response(message.text)
                    self.bot.reply_to(message, response)
                except Exception as e:
                    self.bot.reply_to(message, f"An error occurred: {str(e)}")
            else:
                self.bot.reply_to(message, "LLM handler not set. Unable to process message.")

    def run(self):
        print("Bot is running...")
        self.bot.infinity_polling()