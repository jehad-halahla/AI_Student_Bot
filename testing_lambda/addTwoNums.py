import requests
import json

# Replace with your API Gateway endpoint
api_url = "https://pdj7g4ezpi.execute-api.us-east-1.amazonaws.com/test"  # Replace with your API URL

def call_lambda_api():
    # Define the data to send in the request body
    payload = {
        "num1": 5,
        "num2": 3
    }

    # Send a POST request to the API Gateway
    response = requests.post(api_url, json=payload)

    # Check the response status
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        print("Response:", json.dumps(result, indent=4))
    else:
        print(f"Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    call_lambda_api()
