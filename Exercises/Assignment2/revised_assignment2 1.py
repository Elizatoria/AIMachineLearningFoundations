'''Create a function calling chatbot script that supports parallel function calling. 
Have at least 3 functions listed in your tool calls and try to call them all in one prompt. 

Extra credit if you make the second API call for a natural language response using the data returned from the function call.

Please submit your code file (either .py or .ipynb) with some comments explaining your process.'''
from dotenv import load_dotenv
from openai import OpenAI
import os, json

# Load environment variables from .env file
load_dotenv('C:/Users/Eliza/Documents/AI_API/class7_api.env')

# Initialize OpenAI API
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def fetch_weather(location):
    # Mock function to fetch weather data
    return {"location": location, "temperature": 75, "description": "Sunny"}

def fetch_news(topic):
    # Mock function to fetch news articles
    return {"topic": topic, "articles": ["Article 1", "Article 2", "Article 3"]}

def summarize_text(text):
    # Mock function to summarize text
    return f"Summary of: {text}"

'''
Up to here in the code, you're on the right path. 
Below you need to add a few things and make some changes:

    1. Create the list of dictionaries that hold the natural language description of the functions available 
    and how the arguments are connected to the function. (tools parameter)

    2. Create the list of dictionaries that hold the message list. (messages parameter)

    3. Make the chat completions API call using the API client you created earlier. Make sure to have the 4 parameters
    needed for function calling ability.

    4. Append the API response ("assistant" message) to the message list and then run the functions the model picked 
    and append the function result as the "tool" message.

    5. Make the second API call. This only needs the 2 required parameters (model and the updated message list).
'''


def chatbot_function_call(prompt):
    # Step 1: Define the functions to call in parallel
    function_calls = [
        {"function": fetch_weather, "args": {"location": "San Francisco"}},
        {"function": fetch_news, "args": {"topic": "Technology"}},
    ]
    
    # Step 2: Execute the function calls in parallel
    responses = []
    for call in function_calls:
        func = call["function"]
        args = call["args"]
        response = func(**args)
        responses.append(response)
    
    # Step 3: Use the data from the first function call in the second API call
    weather_data = responses[0]
    weather_summary = summarize_text(f"The weather in {weather_data['location']} is {weather_data['description']} with a temperature of {weather_data['temperature']}°F.")
    
    # Step 4: Print the results
    print("Weather Data:", json.dumps(weather_data, indent=2))
    print("Weather Summary:", weather_summary)
    print("News Data:", json.dumps(responses[1], indent=2))

if __name__ == "__main__":
    prompt = "Fetch weather and news, then summarize the weather."
    chatbot_function_call(prompt)