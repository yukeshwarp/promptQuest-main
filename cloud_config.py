import os
from openai import AzureOpenAI

ENDPOINT = os.getenv("DB_ENDPOINT")
KEY =os.getenv("DB_KEY")
DATABASE_NAME = os.getenv("DB_NAME")
CONTAINER_NAME = os.getenv("DB_CONTAINER_NAME")

# LLM setup
llmclient = AzureOpenAI(
    azure_endpoint=os.getenv("LLM_ENDPOINT"),
    api_key=os.getenv("LLM_KEY"),
    api_version="2024-10-01-preview",
)