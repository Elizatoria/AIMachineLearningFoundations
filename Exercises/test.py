# import datetime
# import psutil
# import requests
# import json

# # Function to get the current time
# def get_current_time():
#     return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# # Function to get system statistics
# def get_system_stats():
#     cpu_usage = psutil.cpu_percent(interval=1)
#     memory_info = psutil.virtual_memory()
#     return {
#         "cpu_usage": cpu_usage,
#         "memory_info": {
#             "total": memory_info.total,
#             "available": memory_info.available,
#             "percent": memory_info.percent,
#             "used": memory_info.used,
#             "free": memory_info.free
#         }
#     }

# # Function to get weather information for a given city
# def get_weather_info(city):
#     api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your OpenWeatherMap API key
#     url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         weather = {
#             "city": data["name"],
#             "temperature": data["main"]["temp"],
#             "description": data["weather"][0]["description"]
#         }
#         return weather
#     else:
#         return {"error": "City not found"}

# # Main function to handle sequential function calls
# def main():
#     # Call function to get current time
#     current_time = get_current_time()
#     print(f"Current Time: {current_time}")

#     # Call function to get system statistics
#     system_stats = get_system_stats()
#     print(f"System Stats: {json.dumps(system_stats, indent=2)}")

#     # Example: Use current time or system stats in the next function call (if needed)
#     # Here we use a hardcoded city for simplicity
#     city = "London"
    
#     # Call function to get weather information
#     weather_info = get_weather_info(city)
#     print(f"Weather Info: {json.dumps(weather_info, indent=2)}")

# if __name__ == "__main__":
#     main()

# import os
# import openai
# import requests
# from dotenv import load_dotenv
# import json

# # Load environment variables from .env file
# load_dotenv('C:/Users/Eliza/Documents/MachineLearning/AI_API/class7_api.env')

# # Set up OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')

# # Function to get current weather using OpenWeatherMap API
# def get_weather(city):
#     api_key = os.getenv('OPENWEATHERMAP_API_KEY')
#     url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
#     response = requests.get(url)
#     return response.json()

# # Function to get top news headline using NewsAPI
# def get_news_headline():
#     api_key = os.getenv('NEWSAPI_KEY')
#     url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
#     response = requests.get(url)
#     news_data = response.json()
#     if news_data['articles']:
#         return news_data['articles'][0]['title']
#     else:
#         return "No news available."

# # Function to calculate the factorial of a number
# def calculate_factorial(number):
#     if number == 0 or number == 1:
#         return 1
#     else:
#         return number * calculate_factorial(number - 1)

# # Define a function to call all three APIs sequentially
# def perform_sequential_calls(city, number):
#     weather = get_weather(city)
#     weather_description = weather['weather'][0]['description']
#     news_headline = get_news_headline()
#     factorial = calculate_factorial(number)
    
#     return {
#         "weather": weather_description,
#         "news_headline": news_headline,
#         "factorial": factorial
#     }

# # Example usage
# city = "San Francisco"
# number = 5

# # Perform sequential calls
# result = perform_sequential_calls(city, number)

# # Print the result
# print("Weather:", result["weather"])
# print("News Headline:", result["news_headline"])
# print("Factorial:", result["factorial"])

# # Save the result to a JSON file
# with open('result.json', 'w') as json_file:
#     json.dump(result, json_file, indent=4)

from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import time

# Load environment variables
load_dotenv('C:/Users/Eliza/Documents/MachineLearning/AI_API/class7_api.env')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function definitions
def get_weather(location):
    # Simulate API call delay
    time.sleep(1)
    return {"location": location, "temperature": 22, "condition": "Sunny"}

def get_news(category):
    # Simulate API call delay
    time.sleep(1)
    return {"category": category, "headline": "AI Chatbots Revolutionize Customer Service"}

def get_stock_price(symbol):
    # Simulate API call delay
    time.sleep(1)
    return {"symbol": symbol, "price": 150.25}

# Function to handle parallel function calls
def parallel_function_call(weather, news, stock):
    weather_result = get_weather(weather["location"])
    news_result = get_news(news["category"])
    stock_result = get_stock_price(stock["symbol"])
    
    return {
        "weather": weather_result,
        "news": news_result,
        "stock": stock_result
    }

# Main chatbot function
def chatbot():
    # Define available functions
    functions = [
        {
            "name": "parallel_function_call",
            "description": "Call multiple functions in parallel",
            "parameters": {
                "type": "object",
                "properties": {
                    "weather": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    },
                    "news": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"}
                        }
                    },
                    "stock": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"}
                        }
                    }
                }
            }
        }
    ]
    
    # Simulate user input
    user_input = "Give me a summary of the weather in New York, the latest technology news, and Apple's stock price."
    
    # First API call to get function calls
    messages = [{"role": "user", "content": user_input}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    
    response_message = response.choices[0].message
    
    if response_message.function_call:
        function_name = response_message.function_call.name
        function_args = json.loads(response_message.function_call.arguments)
        
        if function_name == "parallel_function_call":
            function_results = parallel_function_call(**function_args)
            
            # Second API call for natural language response
            messages.append(response_message)
            messages.append({
                "role": "function",
                "name": "parallel_function_call",
                "content": json.dumps(function_results)
            })
            
            nl_response = client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                messages=messages
            )
            
            print("Chatbot:", nl_response.choices[0].message.content)
        else:
            print("Error: Unexpected function call")
    else:
        print("Error: No function call received")

# Run the chatbot
if __name__ == "__main__":
    chatbot()