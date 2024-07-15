'''Create a function calling chatbot script that supports parallel function calling. 
Have at least 3 functions listed in your tool calls and try to call them all in one prompt. 

Extra credit if you make the second API call for a natural language response using the data returned from the function call.

Please submit your code file (either .py or .ipynb) with some comments explaining your process.'''
from dotenv import load_dotenv
from openai import OpenAI
import os
import json

# Load environment variables from .env file
load_dotenv('C:/Users/Eliza/Documents/MachineLearning/AI_API/class7_api.env')

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define functions
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def ask_wolfram(query):
    """Ask Wolfram Alpha a question for factual information"""
    # Dummy implementation
    return json.dumps({"query": query, "answer": "42"})

def get_current_time_and_date():
    """Get the current time and date"""
    # Dummy implementation
    return json.dumps({"current_time": "2024-07-09 12:00:00"})

# Function to handle the conversation
def run_conversation():
    # Step 1: Send the conversation and available functions to the model
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_wolfram",
                "description": "Ask Wolfram Alpha a question for factual information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The question to ask Wolfram Alpha"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time_and_date",
                "description": "Get the current time and date",
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # Step 2: Check if the model wanted to call a function
    if tool_calls:
        # Step 3: Call the functions
        available_functions = {
            "get_current_weather": get_current_weather,
            "ask_wolfram": ask_wolfram,
            "get_current_time_and_date": get_current_time_and_date,
        }
        messages.append(response_message)
        
        # Step 4: Send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        print(messages)
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        return second_response.choices[0].message.content

print(run_conversation())