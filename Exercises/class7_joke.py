'''Write a simple chat completion API call that prompts GPT 3.5 Turbo to respond with a joke.'''

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv('C:/Users/Eliza/Documents/AI_Class_7/class_7_api.env')
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

messages = [
    {"role": "system","content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."}
]

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=messages
)

print(response.choices[0].message.content)