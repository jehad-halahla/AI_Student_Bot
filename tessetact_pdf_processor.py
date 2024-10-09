import pytesseract
from pdf2image import convert_from_path
import os
from dotenv import load_dotenv

import time

class PDFTextExtractor:
    """
    A class to extract text from a PDF file using OCR (Optical Character Recognition) with Tesseract.
    
    This class converts each page of the PDF into an image and processes the image to extract text using
    pytesseract. The extracted text can be written to an output file.

    Attributes:
        dpi (int): The resolution used when converting PDF to images. Defaults to 300.
        language (str): The language to be used by Tesseract for OCR. Defaults to 'ara' (Arabic).
    """

    def __init__(self, dpi: int = 500, language: str = 'ara'):
        """
        Initialize the PDFTextExtractor class with DPI and language.

        Args:
            dpi (int): Dots per inch for image conversion from PDF. Defaults to 300.
            language (str): Language to be used by Tesseract for OCR. Defaults to 'ara' (Arabic).
        """
        self.dpi = dpi
        self.language = language

    def convert_pdf_to_images(self, pdf_path: str):
        """
        Convert the PDF file into images for each page.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List: A list of images representing each page in the PDF.
        """
        return convert_from_path(pdf_path, self.dpi)

    def extract_text_from_image(self, image):
        """
        Extract text from a single image using Tesseract OCR.

        Args:
            image: The image to process.

        Returns:
            str: The extracted text from the image.
        """
        return pytesseract.image_to_string(image, lang=self.language)

    def log_runtime(self, log_message: str):
        """
        Logs the runtime message to a file.

        Args:
            log_message (str): The message containing runtime information.
        """
        load_dotenv()
        log_file = os.getenv('LOG_FILE')
        with open(log_file, 'a') as log:
            log.write(log_message)

    def process_pdf(self, pdf_path: str, output_path: str):
        """
        Process the PDF by converting it into images and extracting text from each page.
        Logs the runtime information and saves the extracted text to the specified output file.

        Args:
            pdf_path (str): The path to the input PDF file.
            output_path (str): The path to save the extracted text.
        """
        start_time = time.time()  # Start time logging
        pages = self.convert_pdf_to_images(pdf_path)

        with open(output_path, 'w', encoding="utf-8") as f:
            for page_number, page in enumerate(pages):
                print(f"Processing page {page_number + 1}")
                text = self.extract_text_from_image(page)
                f.write(f'--- Page {page_number + 1} ---\n')
                f.write(text)
                f.write('\n\n')

        end_time = time.time()  # End time logging
        runtime = end_time - start_time
        log_message = f"{os.path.basename(__file__)} --> time taken: {runtime:.2f} seconds --> start of the run: {time.ctime(start_time)}\n"
        self.log_runtime(log_message)

        print(f'Text extracted and saved to {output_path}')
        print(f"Process took {runtime:.2f} seconds")

# Usage example
if __name__ == '__main__':
    # Create an instance of PDFTextExtractor with the desired parameters
    pdf_extractor = PDFTextExtractor(dpi=600, language='ara')
    
    # Process the PDF and save the extracted text
    pdf_extractor.process_pdf(pdf_path='short_stories_ar.pdf', output_path='short_stories_ar.txt')
