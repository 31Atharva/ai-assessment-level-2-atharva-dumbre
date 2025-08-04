import os
from pyexpat.errors import messages 
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in the .env file.")

# Initialize the Groq client with the API key
client = Groq(api_key=api_key)

#System Message
system_message = {
    "role": "system",
    "content": "You are helpful assistant that can analyze the sentiment of text from user and classify it as positive, negative and neutral."
}

#agent
while True:
    user_input = input("Enter the text you want to analyze (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    message = [
        system_message,
        {"role": "user",
         "content": f"analyze the sentiment of the following text and classify it as positive, negative, or neutral:\n\n{user_input}\n\nSentiment:"}
    ]
    

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=message,
        temperature=0.3,
        max_completion_tokens=50,
        top_p=1,
        stream=True,
        stop=None,
  )

    print("Sentiment Analysis Result:", end=" ")
    for chunk in completion:
      print(chunk.choices[0].delta.content or "", end="")
    print("\n")
