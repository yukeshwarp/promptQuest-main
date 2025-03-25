import os
from openai import AzureOpenAI

ENDPOINT = os.getenv("DB_ENDPOINT")
KEY =os.getenv("DB_KEY")
DATABASE_NAME = os.getenv("DB_NAME")
CONTAINER_NAME = os.getenv("DB_CONTAINER_NAME")
# Redis connection details (replace with your actual values)
REDIS_HOST = os.getenv("REDIS_HOST") # Example: 'my-cluster.redisenterprise.azure.com'
REDIS_PORT = 10000  # The port (default is 6379, but it may vary)
REDIS_PASSWORD = os.getenv("REDIS_KEY")  # The password/key you got from Azure

# The connection string for Celery
redis_url = f'rediss://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}'
# LLM setup
llmclient = AzureOpenAI(
    azure_endpoint=os.getenv("LLM_ENDPOINT"),
    api_key=os.getenv("LLM_KEY"),
    api_version="2024-10-01-preview",
)