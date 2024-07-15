# '''
# 1. Set up Evviroment by importing libraries and loading in the API Keys
# pip install python-dotenv openai psutil requests

# 2. Create Data Retrieval Functions
# get_current_datetime: Returns the current date and time.
# get_battery_status: Returns the battery percentage and whether the device is plugged in.
# get_top_headlines: Returns the top news headlines using the NewsAPI.
# get_current_weather: Returns the current weather for a specified location using the OpenWeatherMap API.
# get_wolfram_alpha_answer: Returns a factual answer to a query using the Wolfram Alpha Short Answers API.
# '''
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

# Initialize the message list
message_list = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    }
]

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

def get_current_time_and_date():
    now = datetime.datetime.now()
    return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}."

def ask_wolfram(query):
    url = f"http://api.wolframalpha.com/v1/result?i={query}&appid={wolframalpha_app_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return "Sorry, I couldn't retrieve the answer."
    

# print(get_current_weather("Rome"))
# print(get_top_headlines())
# print(get_battery_status())
# print(get_current_time_and_date())
# print(ask_wolfram("What is the percent of 68 out of 100?"))