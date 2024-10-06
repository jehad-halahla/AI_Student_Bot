from abc import ABC, abstractmethod
import google.generativeai as genai
import boto3
import json

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

