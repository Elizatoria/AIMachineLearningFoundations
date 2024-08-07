'''
1. Set up Evviroment
pip install psutil requests python-dotenv openai

2. Create Data Retrieval Functions
get_current_weather: Returns the current weather for a specified location using the OpenWeatherMap API.
ask_wolfram: Returns a factual answer to a query using the Wolfram Alpha Short Answers API.
get_current_time_and_date: Returns the current date and time.
get_top_headlines: Returns the top news headlines using the NewsAPI.
get_battery_status: Returns the battery percentage and whether the device is plugged in.
'''
import datetime
import psutil
import requests
from dotenv import load_dotenv
from openai import OpenAI
import os, json

# Load environment variables from .env file
load_dotenv("C:/Users/Eliza/Documents/MachineLearning/AI_API/midterm_api.env")

# Access environment variables
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
newsapi_key = os.getenv("NEWSAPI_KEY")
wolframalpha_app_id = os.getenv("WOLFRAMALPHA_APP_ID")

# Define the functions to interact with external APIs and libraries
def get_current_weather(location, unit="fahrenheit"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&units={'imperial' if unit == 'fahrenheit' else 'metric'}&appid={openweathermap_api_key}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        temperature = data['main']['temp']
        weather_description = data['weather'][0]['description']
        return f"The current weather in {location} is {temperature} degrees {unit} with {weather_description}."
    else:
        return "Sorry, I couldn't retrieve the weather data."

def ask_wolfram(query):
    url = f"http://api.wolframalpha.com/v1/result?i={query}&appid={wolframalpha_app_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return f"{query}: {response.text}."
    else:
        return "Sorry, I couldn't retrieve the answer."

def get_current_time_and_date():
    now = datetime.datetime.now()
    return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}."

def get_top_headlines():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={newsapi_key}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        headlines = [article['title'] for article in data['articles'][:5]]
        return "Here are the top headlines:\n" + "\n".join(headlines)
    else:
        return "Sorry, I couldn't retrieve the news."
    
def get_battery_status():
    battery = psutil.sensors_battery()
    if battery:
        return f"The battery is at {battery.percent}% and it is {'charging' if battery.power_plugged else 'not charging'}."
    else:
        return "Sorry, I couldn't retrieve the battery status."

'''3. Create the Function to make the Conversation Chatbot'''
# Function to handle the conversation
def run_conversation():
    # Step 1: Send the conversation and available functions to the model
    '''Test Messages'''
    # messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    # messages = [{"role": "user", "content": "What is the percent of 68 out of 100?"}]
    # messages = [{"role": "user", "content": "What is the current date and time?"}]
    # messages = [{"role": "user", "content": "What are the top headlines?"}]
    # messages = [{"role": "user", "content": "What is the battery status?"}]
    '''Actual Messages (Message_List)'''
    prompt = input("Enter a prompt: ")
    messages = [
    # This is the system message. You can add more detail here about how you want the model to respond.
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    # This is the user message. This is the prompt that you want the model to respond to.
    {
        "role": "user",
        "content": prompt
    }
]

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
        },
        {
            "type": "function",
            "function": {
            "name": "get_top_headlines",
            "description": "Get the current top news headlines.",
            }
        },
        {
            "type": "function",
            "function": {
            "name": "get_battery_status",
            "description": "Get the current battery status of the device.",
            }
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
            "get_top_headlines": get_top_headlines,
            "get_battery_status": get_battery_status
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