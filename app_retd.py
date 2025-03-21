import streamlit as st
import pandas as pd
import plotly.express as px
from azure.cosmos import CosmosClient
import logging
from topic_modelling import extract_topics_from_text
from cloud_config import *
import time
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="promptQuest",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    if st.button("📊 Dashboard", key="nav_dashboard"):
        st.session_state["current_page"] = "Dashboard"
    
    if st.button("💬 Chat Analysis", key="nav_chat"):
        st.session_state["current_page"] = "Chat Analysis"
    
    if st.button("🔍 Topic Explorer", key="nav_topics"):
        st.session_state["current_page"] = "Topic Explorer"
    
    st.markdown("---")
    st.markdown("### Data Filters")
    
    time_options = ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
    selected_time = st.selectbox("Time Period", time_options, index=time_options.index(st.session_state["time_filter"]))
    
    if selected_time != st.session_state["time_filter"]:
        st.session_state["time_filter"] = selected_time
        # We'll refresh data when filter changes
        st.session_state["refresh_data"] = True
    
    limit = st.slider("Number of Records", min_value=50, max_value=200, value=150, step=50)
    
    if st.button("Refresh Data", key="refresh_button"):
        st.session_state["refresh_data"] = True

# Function to fetch data from Cosmos DB
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
                
            query += " WHERE c.TimeStamp >= @filter_time"
            params.append({"name": "@filter_time", "value": filter_time.isoformat()})
        
        query += " ORDER BY c.TimeStamp DESC OFFSET 0 LIMIT @limit"
        params.append({"name": "@limit", "value": limit})
        
        # Initialize Cosmos DB Client
        client = CosmosClient(ENDPOINT, KEY)
        database = client.get_database_client(DATABASE_NAME)
        container = database.get_container_client(CONTAINER_NAME)
        
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        return [{"title": item["ChatTitle"], "timestamp": item.get("TimeStamp"), "assistant": item.get("AssistantName")} for item in items]
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
    topics = (extract_topics_from_text(text_content).split(","))
    
    # Convert topics to a format suitable for visualization
    topic_data = []
    for topic in topics:
        print(topic)
        if isinstance(topic, dict):
            topic_data.append({"topic": topic.get("name", ""), "score": topic.get("score", 0), "keywords": topic.get("keywords", [])})
        elif isinstance(topic, str):
            # If topics are just strings, create a simple structure
            topic_data.append({"topic": topic, "score": 1, "keywords": []})
    
    return text_content, topic_data

# Refresh data if needed
if st.session_state.get("refresh_data", True):
    with st.spinner("Fetching data from database..."):
        st.session_state["chat_titles"] = fetch_chat_titles(limit, st.session_state["time_filter"])
        
        if st.session_state["chat_titles"]:
            st.session_state["text_content"], topic_data = analyze_topics(st.session_state["chat_titles"])
            st.session_state["topics"] = topic_data
            
            # Create dataframe for visualization
            if topic_data:
                df = pd.DataFrame(topic_data)
                st.session_state["topic_data"] = df
        
        st.session_state["refresh_data"] = False

# Dashboard Page
if st.session_state["current_page"] == "Dashboard":
    st.markdown('<div class="main-header">promptQuest Dashboard</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(st.session_state["chat_titles"])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Chats</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(st.session_state["topics"])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Identified Topics</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{st.session_state["time_filter"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Time Period</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Extracting PromptAssistant data
    prompt_assistant_data = {}
    for chat in st.session_state["chat_titles"]:
        assistant_name = chat.get("assistant", "Unknown")
        if assistant_name in prompt_assistant_data:
            prompt_assistant_data[assistant_name] += 1
        else:
            prompt_assistant_data[assistant_name] = 1
    # Convert to DataFrame
    assistant_df = pd.DataFrame(list(prompt_assistant_data.items()), columns=["Assistant", "Chat Count"])

    # Topic visualization
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


# Chat Analysis Page
elif st.session_state["current_page"] == "Chat Analysis":
    st.markdown('<div class="main-header">Chat Analysis</div>', unsafe_allow_html=True)
    
    # Chat interface
    st.markdown('<div class="sub-header">Ask Questions About Your Data</div>', unsafe_allow_html=True)
    
    # Display chat history
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
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
                # Simulate response streaming for demonstration
                # In real code, this would use the llmclient as in the original code
                time.sleep(1)  # Simulating API call
                
                # For demonstration, create a simple response
                topic_names = [t["topic"] for t in topics]
                # bot_response = f"Based on the analysis of {len(st.session_state['chat_titles'])} chat titles, I found these main topics: {', '.join(topic_names[:5])}. Your question about '{prompt}' relates to several of these topics."
                
                # In the real implementation, you would use:
                response_stream = llmclient.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant who answers questions based on data from the database."},
                        {"role": "user", "content": f"Answer the user question based on the following data from the database:\n\nText Content: {text_content}\n\nHighlighted Topics: {topics}\n\nQuestion: {prompt}"}
                    ],
                    temperature=0.5,
                    stream=True,
                )
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Simulate streaming for demonstration
                # for word in bot_response.split():
                #     full_response += word + " "
                #     message_placeholder.markdown(full_response)
                #     time.sleep(0.05)
                
                # In the real implementation, you would use:
                # stream = ""
                for chunk in response_stream:
                    if chunk.choices:
                        bot_response += chunk.choices[0].delta.content or ""
                        message_placeholder.markdown(bot_response)
            st.session_state["messages"].append({"role": "assistant", "content": bot_response})
            
        else:
            st.warning("Please fetch and analyze topics first.")

# Topic Explorer Page
elif st.session_state["current_page"] == "Topic Explorer":
    st.markdown('<div class="main-header">Topic Explorer</div>', unsafe_allow_html=True)
    
    if st.session_state["topic_data"] is not None and not st.session_state["topic_data"].empty:
        df = st.session_state["topic_data"]
        
        # Create a bar chart of topics
        fig = px.bar(
            df.sort_values("score", ascending=False).head(30),
            x="topic",
            y="score",
            color="score",
            color_continuous_scale="Blues",
            title="Top 30 Topics by Relevance Score"
        )
        fig.update_layout(xaxis_title="Topic", yaxis_title="Relevance Score")
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state["topics"]:
        # Topic selection
        topic_names = [t["topic"] for t in st.session_state["topics"]]
        selected_topic = st.selectbox("Select a topic to explore", topic_names)
        
        # Find the selected topic details
        selected_topic_data = next((t for t in st.session_state["topics"] if t["topic"] == selected_topic), None)
        
        if selected_topic_data:
            st.markdown(f"### Topic: {selected_topic}")
            
            # Display topic details
            col1, col2 = st.columns(2)
            
            # with col1:
            st.markdown("#### Relevance Score")
            st.markdown(f"<div class='metric-value'>{selected_topic_data['score']:.2f}</div>", unsafe_allow_html=True)
            
            if selected_topic_data.get("keywords"):
                st.markdown("#### Keywords")
                for keyword in selected_topic_data["keywords"]:
                    st.markdown(f"- {keyword}")
            
            # with col2:
                # Find chats related to this topic
            related_chats = []
            for chat in st.session_state["chat_titles"]:
                if selected_topic.lower() in chat["title"].lower():
                    related_chats.append(chat)
            
            st.markdown("#### Related Chats")
            if related_chats:
                for chat in related_chats[:10]:  # Show top 10
                    st.markdown(f"- {chat['title']}")
            else:
                st.info("No directly related chats found.")
            
            # Topic trend over time (if timestamp data is available)
            st.markdown("#### Topic Trend")
            st.info("Topic trend visualization would be displayed here based on timestamp data.")
            
            # Generate insights about the topic
            st.markdown("#### AI Insights")
            st.markdown(f"""
            Based on the analysis of this topic:
            - This topic appears in approximately {len(related_chats)} chats
            - It has a relevance score of {selected_topic_data['score']:.2f} out of 1.0
            - It's frequently discussed alongside other topics like {', '.join(topic_names[:3])}
            """)
    else:
        st.info("No topics available. Try refreshing the data.")

# Footer
st.markdown("---")
st.markdown("promptQuest v1.0 | Data last refreshed: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Add a floating refresh button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh All Data"):
    st.session_state["refresh_data"] = True
    st.rerun()
