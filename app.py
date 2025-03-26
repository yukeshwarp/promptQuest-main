import streamlit as st
import pandas as pd
import plotly.express as px
from azure.cosmos import CosmosClient
from topicmodelling_dev import extract_topics_from_text
from cloud_config import *
from cloud_config import redis_url
import time
from datetime import datetime, timedelta
import re
from wordcloud import STOPWORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import dcc
import concurrent.futures
from celery import Celery
import random

# Page configuration
st.set_page_config(
    page_title="promptQuest",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create Celery instance and configure it
app = Celery(
    'summarization_tasks',
    broker=redis_url,  # Set the broker URL (used for sending tasks)
    backend=redis_url,  # Optional: if you want to store task results in Redis
    include=['summarization_tasks']  # Module where your tasks are defined
)

# Optional Celery configuration
app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    broker_transport_options={
        'visibility_timeout': 3600,  # Task visibility timeout (in seconds)
        'max_connections': 100,  # Number of Redis connections
    },
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .chat-container {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 20px;
        height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "text_content" not in st.session_state:
    st.session_state["text_content"] = ""
if "topics" not in st.session_state:
    st.session_state["topics"] = []
if "topic_data" not in st.session_state:
    st.session_state["topic_data"] = None
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = []
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"
if "time_filter" not in st.session_state:
    st.session_state["time_filter"] = "All Time"

# Sidebar navigation
with st.sidebar:
    # st.image("https://via.placeholder.com/150x150.png?text=DBminer", width=150)
    st.markdown("### Navigation")
    
    # if st.button("📊 Dashboard", key="nav_dashboard"):
    #     st.session_state["current_page"] = "Dashboard"
    
    # if st.button("💬 Chat Analysis", key="nav_chat"):
    #     st.session_state["current_page"] = "Chat Analysis"
    
    # if st.button("🔍 Topic Explorer", key="nav_topics"):
    #     st.session_state["current_page"] = "Topic Explorer"
    
    st.markdown("---")
    st.markdown("### Data Filters")
    
    # time_options = ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
    # selected_time = st.selectbox("Time Period", time_options, index=time_options.index(st.session_state["time_filter"]))

    # Sidebar Time Filter Options
    time_options = ["Last 24 Hours", "Last 7 Days", "Last 30 Days","Quarterly", "Monthly", "Half-Yearly", "Yearly", "All Time"]
    selected_time = st.selectbox("Time Period", time_options, index=time_options.index(st.session_state["time_filter"]))

    # New: Option for selecting Year and Quarter if the user selects "Quarterly"
    if selected_time == "Quarterly":
        year_options = [str(year) for year in range(2020, datetime.now().year + 1)]  # Adjust the year range as necessary
        selected_year = st.selectbox("Select Year", year_options, index=year_options.index(str(datetime.now().year)))
        selected_quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])
        st.session_state["selected_year"] = selected_year
        st.session_state["selected_quarter"] = selected_quarter

    
    if selected_time != st.session_state["time_filter"]:
        st.session_state["time_filter"] = selected_time
        # We'll refresh data when filter changes
        st.session_state["refresh_data"] = True
    
    limit = st.slider("Number of Records", min_value=150, max_value=12000, value=150, step=50)
    
    if st.button("Refresh Data", key="refresh_button"):
        st.session_state["refresh_data"] = True

def trend_analysis(mode=st.session_state["time_filter"]):
    prompt = ""
    text_content = st.session_state["text_content"]
    # st.write(text_content)
    topics = st.session_state["topics"]
    if mode == "All Time":
        prompt = f"""Analyse the trend over monthly under 40 words, with bullets of key points (For display in the dashboard of application) for the database and return the output in a redable analysis format
                    Database of application usage by users: {text_content}
                    Highlighted topics: {topics}"""
    elif mode == "Quaterly":
        prompt = f"""Analyse the trend over monthly under 40 words, with bullets of key points (For display in the dashboard of application) for the database and return the output in a redable analysis format
                    Database of application usage by users: {text_content}
                    Highlighted topics: {topics}"""
    else:
        prompt = f"""Analyse the trend over monthly under 40 words, with bullets of key points (For display in the dashboard of application) for the database and return the output in a redable analysis format
                    Database of application usage by users: {text_content}
                    Highlighted topics: {topics}"""
    
    response = llmclient.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in analysing trend in application usage based on application log database."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content

@app.task
def summarize(text):
    max_tries = 5
    base_delay = 1
    for attempt in range(max_tries):
        try:
            response = llmclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant expert in summarizing."},
                    {"role": "user", "content": f"Summarize the following chat in less than 25 words with understanding of intent in the chat: {text}"}
                ],
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Summarizing DB resulted in error after try{attempt}. \nError Details: {e}")
            if attempt<max_tries:
                delay = base_delay*(2**attempt) + random.uniform(1,0)
                time.sleep(delay)
            else:
                raise e


# Function to fetch data from Cosmos DB with extended time ranges
def fetch_chat_titles(limit=250, time_filter="All Time"):
    try:
        query = "SELECT c.id, c.TimeStamp, c.AssistantName, c.ChatTitle, c.category FROM c"
        params = []
        
        # Add time filter if needed
        if time_filter != "All Time":
            current_time = datetime.now()
            
            if time_filter == "Last 24 Hours":
                filter_time = current_time - timedelta(days=1)
            elif time_filter == "Last 7 Days":
                filter_time = current_time - timedelta(days=7)
            elif time_filter == "Last 30 Days":
                filter_time = current_time - timedelta(days=30)
            elif time_filter == "Monthly":
                filter_time = datetime(current_time.year, current_time.month, 1)
            elif time_filter == "Half-Yearly":
                half = 1 if current_time.month <= 6 else 2
                start_month = 1 if half == 1 else 7
                filter_time = datetime(current_time.year, start_month, 1)
            elif time_filter == "Yearly":
                filter_time = datetime(current_time.year, 1, 1)
            
            # Add logic for Quarterly selection
            elif time_filter == "Quarterly":
                selected_year = st.session_state["selected_year"]
                selected_quarter = st.session_state["selected_quarter"]

                if selected_quarter == "Q1":
                    filter_time = datetime(int(selected_year), 1, 1)
                elif selected_quarter == "Q2":
                    filter_time = datetime(int(selected_year), 4, 1)
                elif selected_quarter == "Q3":
                    filter_time = datetime(int(selected_year), 7, 1)
                elif selected_quarter == "Q4":
                    filter_time = datetime(int(selected_year), 10, 1)
                
            query += " WHERE c.TimeStamp >= @filter_time"
            params.append({"name": "@filter_time", "value": filter_time.isoformat()})
        
        query += " ORDER BY c.TimeStamp DESC OFFSET 0 LIMIT @limit"
        params.append({"name": "@limit", "value": limit})
        
        # Initialize Cosmos DB Client
        client = CosmosClient(ENDPOINT, KEY)
        database = client.get_database_client(DATABASE_NAME)
        container = database.get_container_client(CONTAINER_NAME)

        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        
        # Use ThreadPoolExecutor to parallelize summarization
        with concurrent.futures.ThreadPoolExecutor() as executor:
            summaries = list(executor.map(summarize, [item["ChatTitle"] if item.get("AssistantName") != "Summarize" else item["ChatTitle"][:100] for item in items]))

        # Combine summaries with the other relevant data
        return [{"title": summary, "timestamp": item.get("TimeStamp"), "assistant": item.get("AssistantName")} for item, summary in zip(items, summaries)]
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return []
    
text_content = ""
# Function to analyze topics
def analyze_topics(chat_titles):
    if not chat_titles:
        return "", []
    
    topics = []
    
    titles = [item["title"] for item in chat_titles]
    text_content = " ".join(titles)
    topics = (extract_topics_from_text(text_content))
    
    topic_data = topics.split(",")
    print(topic_data)
    return text_content, topic_data

# Refresh data if needed
if st.session_state.get("refresh_data", True):
    with st.spinner("Fetching data from database..."):
        st.session_state["chat_titles"] = fetch_chat_titles(limit, st.session_state["time_filter"])
        
        if st.session_state["chat_titles"]:
            st.session_state["text_content"], topic_data = analyze_topics(st.session_state["chat_titles"])
            # st.write(st.session_state["text_content"] )
            st.session_state["topics"] = topic_data
            
            # Create dataframe for visualization
            if topic_data:
                df = pd.DataFrame(topic_data)
                st.session_state["topic_data"] = df
        
        st.session_state["refresh_data"] = False

col1, col2 = st.columns([0.8,0.2])

with col1:
    st.markdown('<div class="main-header">promptQuest Dashboard</div>', unsafe_allow_html=True)
with col2:
    with st.popover("💬"):
        # Chat Analysis Page
        st.markdown('<div class="main-header">Chat Analysis</div>', unsafe_allow_html=True)

        st.markdown('<div class="sub-header">Ask Questions About Your Data</div>', unsafe_allow_html=True)

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

        # User input for questions
        if prompt := st.chat_input("Ask a question about your emails"):
            text_content = st.session_state["text_content"]
            topics = st.session_state["topics"]
            bot_response = ""
            if text_content and topics:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.spinner("Thinking..."):
                    time.sleep(1) 
                    topic_names = [t for t in topics]
                    response_stream = llmclient.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert product analyst who analyses software products based on the user statistics from user database."},
                            {"role": "user", "content": f"""
                                Answer the user's prompt based on the following data from the database. 
                                The database contains usage history of user questions and AI responses from an AI-assisted chatbot interface, specifically used for legal advice.
                                
                                User Database: {text_content}
                                Highlighted Topics: {topics}
                                
                                ---
                                Prompt: {prompt}
                                
                                ---
                                Intelligently analyze the user's intent in the prompt and provide an insightful answer, utilizing relevant data and context from the database.
                                """
                            }
                        ],
                        temperature=0.7,
                        stream=True,
                    )
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response_stream:
                        if chunk.choices:
                            bot_response += chunk.choices[0].delta.content or ""
                            message_placeholder.markdown(bot_response)
                st.session_state["messages"].append({"role": "assistant", "content": bot_response})
                
            else:
                st.warning("Please fetch and analyze topics first.")

st.title("Quick Trend Analysis")
st.markdown("---")
st.markdown(trend_analysis())
prompt_assistant_data = {}
for chat in st.session_state["chat_titles"]:
    assistant_name = chat.get("assistant", "Unknown")
    if assistant_name in prompt_assistant_data:
        prompt_assistant_data[assistant_name] += 1
    else:
        prompt_assistant_data[assistant_name] = 1
assistant_df = pd.DataFrame(list(prompt_assistant_data.items()), columns=["Assistant", "Chat Count"])


st.markdown('<div class="sub-header">Prompt Assistant Analysis</div>', unsafe_allow_html=True)

if not assistant_df.empty:
    # Create a bar chart for PromptAssistant
    fig = px.bar(
        assistant_df.sort_values("Chat Count", ascending=False),
        x="Assistant",
        y="Chat Count",
        color="Chat Count",
        color_continuous_scale="Blues",
        title="Top 10 Prompt Assistants by Chat Count"
    )
    fig.update_layout(xaxis_title="Prompt Assistant", yaxis_title="Chat Count")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for Prompt Assistant analysis.")


# Topic Explorer Page
# elif st.session_state["current_page"] == "Topic Explorer":
#     st.markdown('<div class="main-header">Topic Explorer</div>', unsafe_allow_html=True)
    
#     if st.session_state["topic_data"] is not None and not st.session_state["topic_data"].empty:
        
#         wordcloud = WordCloud(
#             background_color='#140012',
#             width=512,
#             height=224, margin=True).generate(' '.join(df for df in st.session_state["topics"]))
        
#         # Display WordCloud in Streamlit
#         fig, ax = plt.subplots()
#         ax.imshow(wordcloud, interpolation='bilinear')
#         ax.axis('off')  # Hide axes
        
#         st.pyplot(fig)

#     if st.session_state["topics"]:
#         # Topic selection
#         topic_names = [t for t in st.session_state["topics"]]
#         selected_topic = st.selectbox("Select a topic to explore", topic_names)
        
#         # Find the selected topic details
#         selected_topic_data = next((t for t in st.session_state["topics"] if t == selected_topic), None)
        
#     else:
#         st.info("No topics available. Try refreshing the data.")

# Footer
st.markdown("---")
st.markdown("promptQuest v1.0 | Data last refreshed: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Add a floating refresh button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh All Data"):
    st.session_state["refresh_data"] = True
    st.rerun()
