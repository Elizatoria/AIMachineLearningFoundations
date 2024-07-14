from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv('C:/Users/Eliza/Documents/MachineLearning/AI_API/class7_api.env')
openai_api_key = os.environ.get("OPENAI_API_KEY")
# print(openai_api_key)

# Create an instance of the ChatOpenAI class
lim = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
# Use the instance to send a message to the model and get a response
response = lim.invoke("Tell me a Joke")
# Print the response content
print(response.content)