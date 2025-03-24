import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import time
import logging
import numpy as np
import streamlit as st
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_step(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


# Modify the data loading section
try:
    log_step("Attempting to load CSV file from drive...")

    # Try to load from Google Drive if secrets are available
    file_id = st.secrets["data"]["gdrive_file_id"]
    # url = f"https://drive.google.com/uc?id={file_id}"
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"

    df = pd.read_csv(url)
    log_step("Loaded DataFrame from Google Drive")
except KeyError:
    log_step(
        "Failed to load CSV file from drive.  ")

log_step(f"Loaded DataFrame with shape: {df.shape}")

# LLM setup
log_step("Initializing OpenAI LLM...")
llm = OpenAI(
    api_token=st.secrets["openai"]["api_key"],
    options={
        "model": "gpt-4",
        "temperature": 0.1,  # Lower temperature for more focused responses
        "max_tokens": 1000
    }
)

# Create SmartDataframe with minimal configuration
log_step("Creating SmartDataframe...")
sdf = SmartDataframe(
    df,
    config={
        "llm": llm,
        "enable_cache": False,
        "use_error_correction_framework": False,
        "custom_whitelisted_dependencies": [],
        "save_charts": False,
        "verbose": True,
        "enforce_privacy": False,
        "max_retries": 3,
        "execution_mode": "local"  # Use local execution mode
    }
)
log_step("SmartDataframe created successfully")

# Add these functions at the top of your file
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

def main():
    st.title("Midori - Oppkey Assistant")
    
    # Add login protection
    if not check_password():
        st.stop()  # Do not continue if check_password is not True
    
    # Create two columns for image and text
    col1, col2 = st.columns([1, 2])  # [1, 2] sets the width ratio between columns
    
    # Add image in the first column
    with col1:
        st.image("images/midori.png", width=300)
    
    # Add text in the second column
    with col2:
        st.markdown("""
        ### Midori Masuda is a fake person. She is a data analyst helping Oppkey partners make strategic business decisions
        """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your dataset"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            
            response = execute_query(prompt)
            
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

def execute_query(query):
    """Function to safely execute query"""
    log_step(f"Sending query: '{query}'")
    start_time = time.time()
    
    try:
        if "list countries" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Get country distribution, excluding NaN values
            country_counts = clean_df['last_ip_location'].dropna().value_counts()
            total_users = len(clean_df)
            
            response = "### 🌎 List of Countries\n\n"
            response += "| Country | Number of Users | Percentage |\n"
            response += "|---------|----------------|------------|\n"
            
            for country, count in country_counts.items():
                percentage = round((count / total_users * 100), 2)
                response += f"| {str(country)} | {int(count)} | {percentage}% |\n"
            
            # Add information about users with unknown location if any exist
            unknown_count = clean_df['last_ip_location'].isna().sum()
            if unknown_count > 0:
                unknown_percentage = round((unknown_count / total_users * 100), 2)
                response += f"| Unknown Location | {int(unknown_count)} | {unknown_percentage}% |\n"
            
            return response
            
        elif "breakdown by country" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Get country distribution
            country_counts = clean_df['last_ip_location'].value_counts()
            total_users = len(clean_df)
            
            # Create a DataFrame for the chart
            chart_data = pd.DataFrame({
                'Country': country_counts.index,
                'Users': country_counts.values,
                'Percentage': (country_counts.values / total_users * 100).round(2)
            })
            
            # Limit to top 15 countries for better visualization
            top_15 = chart_data.head(15)
            
            response = "### 🌎 User Distribution by Country\n\n"
            
            # Create bar chart using Streamlit
            st.bar_chart(
                data=top_15.set_index('Country')['Users'],
                use_container_width=True
            )
            
            # Add detailed breakdown in text
            response += "#### Detailed Breakdown:\n\n"
            response += "| Country | Users | Percentage |\n"
            response += "|---------|--------|------------|\n"
            
            for _, row in chart_data.iterrows():
                response += f"| {row['Country']} | {row['Users']} | {row['Percentage']}% |\n"
            
            # Add summary for remaining countries
            if len(chart_data) > 15:
                others_count = chart_data[15:]['Users'].sum()
                others_pct = (others_count / total_users * 100).round(2)
                response += f"\n*Other countries: {others_count} users ({others_pct}%)*"
            
            return response
            
        elif any(x in query.lower() for x in ["show users in", "users from", "list users in"]):
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Extract location name from query
            query_lower = query.lower()
            location = None
            
            # Handle common US variations and major cities
            us_variants = ["usa", "united states", "us "]
            us_cities = {
                "san francisco": "San Francisco",
                "new york": "New York",
                "seattle": "Seattle",
                "los angeles": "Los Angeles",
                "chicago": "Chicago",
                "boston": "Boston",
                "austin": "Austin",
                # Add more cities as needed
            }
            
            # Check for US variants first
            if any(variant in query_lower for variant in us_variants):
                # For US, use partial match to include all US cities
                country_users = clean_df[clean_df['last_ip_location'].fillna('').str.contains("United States", na=False)]
            else:
                # Check for specific cities
                location = None
                for city_name, full_name in us_cities.items():
                    if city_name in query_lower:
                        location = full_name
                        break
                
                # If no city found, extract location name after "in" or "from"
                if not location:
                    for prefix in ["show users in ", "users from ", "list users in "]:
                        if prefix in query_lower:
                            extracted = query_lower.split(prefix)[-1].strip().title()
                            if extracted:  # Only set location if we extracted something
                                location = extracted
                                break
                
                if location:
                    # Use partial match for cities
                    country_users = clean_df[clean_df['last_ip_location'].fillna('').str.contains(location, na=False)]
                else:
                    return "Please specify a valid location (country or city)"
            
            if len(country_users) == 0:
                return f"No users found in {location or 'United States'}."
            
            response = f"### 👥 Users in {location or 'United States'} ({len(country_users)} users)\n\n"
            
            # Add user details
            for _, user in country_users.iterrows():
                name = str(user['name']) if pd.notna(user['name']) else 'No name'
                org = str(user['organization']) if pd.notna(user['organization']) else 'No organization'
                city = str(user['last_ip_location']) if pd.notna(user['last_ip_location']) else 'Unknown location'
                response += f"- {name} ({org}) - {city}\n"
            
            return response
            
        else:
            # Use PandasAI for other queries
            result = sdf.chat(query)
            if not isinstance(result, str):
                result = str(result)
            return result
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        log_step(f"Error occurred after {elapsed_time:.2f} seconds")
        logger.error(f"Error: {str(e)}", exc_info=True)
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    main()
