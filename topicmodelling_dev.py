from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from cloud_config import llmclient

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Fixed: 'punkt' instead of 'punkt_tab'


def preprocess_text(text):
    """Cleans text by removing special characters, stopwords, and lemmatizing."""
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.lower().split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return " ".join(cleaned_words)

def extract_topics_from_text(text, max_topics=5, max_top_words=10):
    """Extract topics using NMF and return structured topic data."""
    try:
        cleaned_text = preprocess_text(text)
        if len(cleaned_text.split()) < 10:
            logging.warning("Text too short for meaningful topic extraction")
            return []
        
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.85,
            min_df=2,
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # Split text into sentences or chunks
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 3:
            sentences = [text[i:i+100] for i in range(0, len(text), 100) if len(text[i:i+100].strip()) > 0]
        
        tfidf = vectorizer.fit_transform(sentences)
        
        if tfidf.shape[1] < 2:
            logging.warning("Not enough features extracted for NMF")
            return []
        
        n_topics = min(max_topics, min(5, tfidf.shape[1]-1))
        
        nmf = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=500,
            l1_ratio=0.5
        )
        
        nmf_result = nmf.fit_transform(tfidf)
        feature_names = vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_features_ind = topic.argsort()[:-max_top_words-1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            
            weights = topic[top_features_ind]
            weights = weights / weights.sum()
            
            weighted_terms = [{"term": feature, "weight": float(weight)} for feature, weight in zip(top_features, weights)]
            
            topics.append({
                "topic": f"Topic {topic_idx + 1}",
                "score": float(sum(weights)),
                "keywords": weighted_terms
            })
        
        
        topic_analysis = ""
        for topicss in topics:
            for topic in topicss['keywords']:
                topic_analysis += f"{topic['term']} with weight {topic['weight']}\n "
        
        return interpret_topics_with_llm(text, topic_analysis)
    
    except Exception as e:
        logging.error(f"Error extracting topics: {e}")
        return []

def interpret_topics_with_llm(text, raw_topics):
    """
    Use LLM to interpret raw topics and return structured interpretations.
    This function should be called separately from extract_topics_from_text
    if LLM interpretation is needed.
    """
    try:
        prompt = f"""
        I need you to analyze the following text and the extracted topic keywords to identify 
        the main themes and topics. For each theme, provide a concise label and a brief description.
        
        Text excerpt: {text[:1000]}... (truncated for brevity)
        
        Raw extracted topics:
        {raw_topics}
        
        Please respond with:
        1. Main themes you identify from the text and keywords
        2. For each theme, provide a concise label and a 1-2 sentence description
        3. Any notable subtopics or related concepts
        
        Return the identifies topics by separating with commas. Return only the topics strictly with no additional texts.
        """
        
        response = llmclient.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system", "content": "You are a topic analysis expert who can identify meaningful themes and topics from text."
            }, {
                "role": "user", "content": prompt
            }],
            temperature=0.3
        )

        interpreted_content = response.choices[0].message.content

        return interpreted_content

    except Exception as e:
        logging.error(f"Error interpreting topics with LLM: {e}")
        return []

def parse_interpreted_topics(interpreted_content):
    """
    Converts raw LLM response into structured topic data (list of dicts).
    """
    topics = []
    
    lines = interpreted_content.split("\n")
    current_topic = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for numbered list items that likely indicate new topics
        if re.match(r'^\d+\.', line):
            if current_topic and "label" in current_topic:  # Save the previous topic
                topics.append(current_topic)
            current_topic = {}
            
            # Extract label and description if they're on the same line
            parts = line.split(":", 1)
            if len(parts) > 1:
                label = parts[0].strip()
                # Remove the number prefix
                label = re.sub(r'^\d+\.\s*', '', label)
                current_topic["label"] = label
                current_topic["description"] = parts[1].strip()
            else:
                # Just store the label for now
                label = line.strip()
                label = re.sub(r'^\d+\.\s*', '', label)
                current_topic["label"] = label
        
        # If we're in a topic and this line has a description
        elif current_topic and ":" in line and "label" in current_topic and "description" not in current_topic:
            parts = line.split(":", 1)
            current_topic["description"] = parts[1].strip()
        
        # If this is a continuation of a description
        elif current_topic and "label" in current_topic:
            if "description" in current_topic:
                current_topic["description"] += " " + line
            else:
                current_topic["description"] = line

    # Add the last topic if it exists
    if current_topic and "label" in current_topic:
        topics.append(current_topic)
    
    return topics
