import os
import textwrap
import time
from llm import GeminiLLM
from typing import List
from dotenv import load_dotenv


class ArabicTextCorrector:
    def __init__(self, model: GeminiLLM):
        self.model = model

    def configure(self):
        # Configuration for the model if needed
        pass

    def read_text_from_file(self, file_path: str) -> str:
        """Reads text from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def save_text_to_file(self, text: str, file_path: str):
        """Saves text to a file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)

    def split_text(self, text: str, chunk_size: int) -> List[str]:
        """Splits text into chunks."""
        return textwrap.wrap(text, chunk_size)

    def generate_correction_prompt(self, text: str, custom_prompt: str = None) -> str:
        """Generates a correction prompt, allowing for a custom prompt."""
        if custom_prompt:
            return f"{custom_prompt}\n{text}"
        return f"يرجى إعادة كتابة النص التالي بشكل صحيح دون أي ملاحظات أو توضيحات:\n{text}"

    def correct_text(self, text: str, custom_prompt: str = None) -> str:
        """Corrects the text using the provided model and custom prompt."""
        prompt = self.generate_correction_prompt(text, custom_prompt)
        if not self.model:
            raise ValueError("Model is not configured. Please call 'configure' first.")
        try:
            response = self.model.generate_content(prompt)
            return response
        except Exception as e:
            print(f"Error occurred while generating content: {e}")
            # Return a note indicating the chunk was not processed
            return f"\n\n[تنبيه: لم يتم معالجة هذا الجزء بسبب خطأ]\n\n{text}"

    def process_file(self, input_file: str, output_file: str, output_folder: str, chunk_size: int = 1000, custom_prompt: str = None):
        """Processes the input file in chunks, corrects, and saves the output."""
        start_time = time.time()

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Read the text from the input file
        text = self.read_text_from_file(input_file)
        chunks = self.split_text(text, chunk_size)

        corrected_text = ""
        for i, chunk in enumerate(chunks):
            corrected_chunk = self.correct_text(chunk, custom_prompt)
            corrected_text += corrected_chunk
            # Save each processed chunk
            output_path = os.path.join(output_folder, f'corrected_chunk_{i}.txt')
            self.save_text_to_file(corrected_chunk, output_path)
            print(f"Processed chunk {i + 1} of {len(chunks)}")

        # Save the final corrected text to the output file
        self.save_text_to_file(corrected_text, os.path.join(output_folder, output_file))
        print(f"All chunks processed and saved in {output_file}.")

        # Logging the runtime
        end_time = time.time()
        runtime = end_time - start_time
        log_message = f"{os.path.basename(__file__)} --> time taken: {runtime:.2f} seconds --> start of the run: {time.ctime(start_time)}\n"
        self.log_runtime(log_message)

    def save_chunks_without_processing(self, input_file: str, chunk_numbers: List[int], output_folder: str, chunk_size: int = 1000):
        """Saves specified chunks without processing them."""
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Read and split the input file
        text = self.read_text_from_file(input_file)
        chunks = self.split_text(text, chunk_size)

        for index in chunk_numbers:
            if index < 1 or index > len(chunks):
                print(f"Chunk number {index} is out of range.")
                continue
            chunk = chunks[index - 1]  # Convert 1-based index to 0-based
            output_path = os.path.join(output_folder, f'corrected_chunk_{index}.txt')
            self.save_text_to_file(chunk, output_path)
            print(f"Saved chunk {index} of {len(chunks)}")

    def log_runtime(self, log_message: str):
        """Logs runtime information into runtime.log."""
        with open("runtime.log", 'a', encoding='utf-8') as log_file:
            log_file.write(log_message)


# Example usage
if __name__ == "__main__":
    load_dotenv()  # Load environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Instantiate and configure GeminiLLM
    model = GeminiLLM(model_name='gemini-1.0-pro-latest')
    model.configure(api_key=gemini_api_key)

    corrector = ArabicTextCorrector(model)
    corrector.configure()

    # Processing the file with custom prompt and logging runtime
    corrector.process_file(
        input_file="taw_hist.txt", 
        output_file="corrected_hist.txt", 
        output_folder="correct_hist",
        chunk_size=1000
    )

    # Saving specific chunks without processing
    chunk_numbers = []
