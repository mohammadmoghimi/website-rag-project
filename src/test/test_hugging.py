import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACE_API_KEY_FINEGRAINED")
)

response = client.chat_completion(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {
            "role": "user",
            "content": "What is 2+2?"
        }
    ],
)

print(response.choices[0].message.content)