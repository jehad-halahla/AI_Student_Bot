from abc import ABC, abstractmethod
import google.generativeai as genai
import boto3
import json
import requests
from dotenv import load_dotenv
import os

class LLM(ABC):
    """
    Abstract base class for different LLM implementations.
    """

    @abstractmethod
    def configure(self, api_key: str = None, **kwargs):
        """
        Configure the LLM with necessary parameters like API keys or other credentials.
        
        :param api_key: Optional API key for the service (default: None).
        :param kwargs: Additional configuration parameters specific to the LLM implementation.
        """
        pass

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """
        Generate content based on the provided prompt.

        :param prompt: The text prompt to generate content for.
        :return: The generated content as a string.
        """
        pass

class TitanLLM(LLM):
    def __init__(self, region_name='us-west-2'):
        self.bedrock = None
        self.model_id = None
        self.region_name = region_name

    def configure(self, api_key: str = None, **kwargs):
        """
        Configure the Titan LLM by setting up the boto3 client and other required parameters.
        
        :param api_key: (Not used for Titan, included for compatibility).
        :param kwargs: Additional configuration parameters, e.g., model_id and region_name.
        """
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=kwargs.get('region_name', self.region_name)
        )
        self.model_id = kwargs.get('model_id', "amazon.titan-text-express-v1")

    def generate_content(self, prompt: str) -> str:
        """
        Generate content using the configured Titan LLM based on the provided prompt.

        :param prompt: The text prompt to generate content for.
        :return: The generated content as a string.
        """
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 8192,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
        })

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        response_body = json.loads(response['body'].read())
        return response_body['results'][0]['outputText']
    


class GeminiLLM(LLM):
    def __init__(self, model_name: str = 'gemini-1.0-pro-latest'):
        """
        Initializes the GeminiLLM with the specified model name.
        
        :param model_name: The name of the generative model to use (default: 'gemini-1.0-pro-latest').
        """
        self.model_name = model_name
        self.model = None

    def configure(self, api_key: str, **kwargs):
        """
        Configures the GeminiLLM with the provided API key.
        
        :param api_key: API key for accessing the Google Generative AI service.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate_content(self, prompt: str) -> str:
        """
        Generates content based on the provided prompt using Gemini model.
        
        :param prompt: The text prompt to generate content for.
        :return: The generated content as a string.
        """
        if not self.model:
            raise ValueError("Model is not configured. Please call 'configure' first.")
        response = self.model.generate_content(prompt)
        return response.text


class ClaudeLLM(LLM):
    def __init__(self, region_name='us-east-1'):
        self.bedrock = None
        self.model_id = None
        self.region_name = region_name

    def configure(self, api_key: str = None, **kwargs):
        """
        Configure the Claude LLM by setting up the boto3 client and other required parameters.
        
        :param api_key: (Not used for Claude, included for compatibility).
        :param kwargs: Additional configuration parameters, e.g., model_id and region_name.
        """
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=kwargs.get('region_name', self.region_name)
        )
        self.model_id = kwargs.get('model_id', "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2")

    def generate_content(self, prompt: str) -> str:
        """
        Generate content using the configured Claude LLM based on the provided prompt.

        :param prompt: The text prompt to generate content for.
        :return: The generated content as a string.
        """
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": 3000,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31"
        })

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="*/*",
            body=body
        )

        response_body = json.loads(response['body'].read())
        return response_body['completion']

class CustomLLM(LLM):
    """
    Custom LLM implementation for interacting with an external API to generate content.

    This class sends a POST request to a specified API endpoint using the provided prompt and API token.
    """

    def __init__(self, url: str):
        """
        Initializes the CustomLLM with the API URL.

        :param url: The endpoint URL of the API to which requests will be sent.
        """
        self.url = url
        self.auth_token = None

    def configure(self, api_key: str = None, **kwargs):
        """
        Configure the Custom LLM with necessary parameters such as the API token.

        :param api_key: The API token used for authentication (optional).
        :param kwargs: Additional configuration parameters (for example, `auth_token`).
        """
        # Use the provided API key or get it from kwargs
        self.auth_token = kwargs.get("auth_token", api_key)

        if not self.auth_token:
            raise ValueError("Authentication token must be provided.")

    def generate_content(self, prompt: str) -> str:
        """
        Generate content by sending a prompt to the configured API.

        This method sends the prompt as part of the POST request's payload to the API and processes the response.

        :param prompt: The text prompt that will be sent to the API to generate content.
        :return: The generated content as a string, or an error message if something goes wrong.
        """
        # Ensure the LLM is configured with an authentication token
        if not self.auth_token:
            raise ValueError("Authentication token is not set. Please configure the LLM before generating content.")

        # Payload for the API request
        payload = {
            "prompt": prompt,
            "auth_token": self.auth_token
        }

        # HTTP headers, specifying that the content is JSON
        headers = {
            "Content-Type": "application/json"
        }

        # Make the POST request and handle potential errors
        try:
            response = requests.post(self.url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad HTTP statuses
        except requests.exceptions.RequestException as e:
            return f"Error during the API request: {e}"

        # Parse the response data
        try:
            response_json = response.json()  # Get the JSON response
            response_body = json.loads(response_json['body'])  # Decode the 'body' part of the response

            # Extract relevant fields from the response
            content = response_body.get('content', 'No content returned')
            message = response_body.get('message', 'No message returned')
        except (KeyError, json.JSONDecodeError) as e:
            return f"Error parsing the API response: {str(e)}"

        # Prepare the formatted result text
        result_text = f"Status Code: {response.status_code}\n"
        result_text += f"Message: {message}\n"
        result_text += f"Content: {content}\n"

        return content

    def save_response_to_file(self, result_text: str, filename: str = 'api_response.txt'):
        """
        Save the API response to a file.

        This method writes the formatted response text to a specified file in UTF-8 encoding.

        :param result_text: The formatted response text to be saved.
        :param filename: The name of the file where the response will be saved (default: 'api_response.txt').
        """
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(result_text)
            print(f"Response has been written to {filename}")
        except IOError as e:
            print(f"Error writing the response to file: {e}")


# Example usage of the CustomLLM class
if __name__ == "__main__":
    # Initialize the LLM with the API URL
    load_dotenv()
    custom_llm = CustomLLM("https://j0aoonwgxl.execute-api.eu-north-1.amazonaws.com/dev")

    # Configure the LLM with the API token
    auth_token = os.getenv("AUTH_TOKEN_AWS")
    custom_llm.configure(auth_token=auth_token)

    # Generate content based on a prompt
    result = custom_llm.generate_content("مرحبا حدثني عن الحياة")
    print(result)
    # Save the response to a file
    # custom_llm.save_response_to_file(result)

