import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import time
import logging
# import numpy as np
import streamlit as st
# import hashlib


# Add this near the top of the file with other constants/configurations
SYSTEM_PROMPT = """You are Midori Masuda, a mid-20s female business analyst who helps 
companies make data-driven decisions. You work for Oppkey.
You are friendly and professional, with a knack for explaining 
complex data in simple terms. 
When analyzing data, you focus on practical business insights 
and actionable recommendations.

You are helping managers expand sales in different regions.
"""

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
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"

    df = pd.read_csv(url)
    df.loc[df['organization'].isin(['x', '-', '_', 'none', 'na', 'xxx']), 'organization'] = ''
    log_step("Loaded DataFrame from Google Drive")
except KeyError:
    log_step(
        "Failed to load CSV file from drive. Attempting "
        "local file")

    # Fallback to local file if secrets are not available
    df = pd.read_csv("./data/camera360_users.csv")
    df.loc[df['organization'].isin(['x', '-', '_', 'none', 'na']), 'organization'] = ''

    log_step("Loaded DataFrame from local file")

log_step(f"Loaded DataFrame with shape: {df.shape}")

# LLM setup
log_step("Initializing OpenAI LLM...")
llm = OpenAI(
    api_token=st.secrets["openai"]["api_key"],
    options={
        "model": "gpt-4",
        "temperature": 0.1,  # Lower temperature for more focused responses
        "max_tokens": 3000,
        "system_prompt": SYSTEM_PROMPT  # Add the system prompt here
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
        ### Midori Masuda is a data analyst helping Oppkey partners make strategic business decisions
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
        if "countries" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Function to standardize country names
            def standardize_country(location):
                if pd.isna(location):
                    return 'Unknown'
                location = str(location).strip()
                
                # Extract country name (usually after the last comma)
                parts = location.split(',')
                country = parts[-1].strip() if parts else location
            
            
                # Remove any state/region information
                if ' - ' in country:
                    country = country.split(' - ')[0].strip()
                
                return country
            
            # Clean and standardize country names
            clean_df['last_ip_location'] = clean_df['last_ip_location'].apply(standardize_country)
            
            # Get unique countries and their counts
            country_counts = clean_df['last_ip_location'].value_counts()
            total_users = len(clean_df)
            
            response = "### 🌎 Countries in the Dataset\n\n"
            response += "| Country | Number of Users |\n"
            response += "|---------|----------------|\n"
            
            # Sort by count and display
            for country, count in country_counts.items():
                percentage = float(count) / total_users * 100
                response += f"| {country} | {int(count):,} ({percentage:.1f}%) |\n"
            
            response += f"\n**Total Countries:** {len(country_counts):,}"
            response += f"\n**Total Users:** {total_users:,}"
            
            return response
            
        elif "breakdown by country" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Get country distribution and convert to integers
            country_counts = clean_df['last_ip_location'].value_counts().astype(int)
            total_users = len(clean_df)
            
            # Create a DataFrame for the chart with proper types
            chart_data = pd.DataFrame({
                'Country': country_counts.index,
                'Users': country_counts.values.astype(int),
                'Percentage': (country_counts.values / total_users * 100).round(2)
            })
            
            # Sort by number of users (ensuring integer comparison)
            chart_data = chart_data.sort_values('Users', ascending=False)
            
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
                response += f"| {row['Country']} | {row['Users']} | {row['Percentage']:.2f}% |\n"
            
            # Add summary for remaining countries
            if len(chart_data) > 15:
                others_count = chart_data[15:]['Users'].sum()
                others_pct = (others_count / total_users * 100).round(2)
                response += f"\n*Other countries: {others_count} users ({others_pct:.2f}%)*"
            
            return response
        elif "organizations by industry" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Define 7 main industry categories and their keywords
            industry_keywords = {
                "Technology & Digital": ['tech', 'software', 'digital', 'cyber', 'ai', 'data', 'cloud', 'web', 'app', 'it', 'computing'],
                "Construction & Real Estate": ['construct', 'build', 'real estate', 'property', 'housing', 'development', 'architect', 'realty', 'home'],
                "Automotive & Transportation": ['car', 'auto', 'vehicle', 'motor', 'dealer', 'transport', 'automotive', 'truck', 'fleet'],
                "Healthcare & Education": ['health', 'medical', 'pharma', 'care', 'clinic', 'hospital', 'edu', 'school', 'university', 'college', 'academy'],
                "Financial Services": ['bank', 'finance', 'capital', 'invest', 'trading', 'insurance', 'consult', 'advisory', 'wealth'],
                "Media & Marketing": ['media', 'entertainment', 'game', 'studio', 'film', 'tv', 'broadcast', 'news', 'advertis', 'brand', 'agency', 'pr'],
                "Other Industries": ['manufacturing', 'industrial', 'retail', 'shop', 'store', 'commerce', 'product', 'service', 'group']
            }
            
            def categorize_industry(org):
                if pd.isna(org) or org == '' or org == 'No organization':
                    return 'Uncategorized'
                    
                org_lower = str(org).lower()
                for industry, keywords in industry_keywords.items():
                    if any(keyword in org_lower for keyword in keywords):
                        return industry
                return 'Other Industries'
            
            # Add industry category
            clean_df['industry'] = clean_df['organization'].apply(categorize_industry)
            
            # Get industry counts
            industry_counts = clean_df['industry'].value_counts()
            total_orgs = len(clean_df[clean_df['organization'].notna() & (clean_df['organization'] != '')])
            
            response = "### 🏢 Organizations by Industry\n\n"
            response += "| Industry | Number of Users | Percentage |\n"
            response += "|----------|----------------|------------|\n"
            
            # Sort by count and display
            for industry, count in industry_counts.items():
                percentage = (count / total_orgs * 100)
                response += f"| {industry} | {int(count):,} | {percentage:.1f}% |\n"
            
            # Create bar chart using Streamlit
            st.bar_chart(
                data=industry_counts,
                use_container_width=True
            )
            
            response += f"\n**Total Organizations:** {total_orgs:,}"
            
            # Add sample organizations for each industry
            response += "\n\n### Sample Organizations by Industry:\n"
            for industry in industry_counts.index:
                if industry != 'Uncategorized':
                    sample_orgs = clean_df[clean_df['industry'] == industry]['organization'].unique()[:3]
                    if len(sample_orgs) > 0:
                        response += f"\n**{industry}**: " + ", ".join(sample_orgs)
            
            return response
        elif "organizations" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Get organization counts, handling NaN and empty values
            org_counts = clean_df['organization'].fillna('No organization').replace('', 'No organization').value_counts()
            total_users = len(clean_df)
            
            response = "### 🏢 Organizations in the Dataset\n\n"
            response += "| Organization | Number of Users | Percentage |\n"
            response += "|--------------|----------------|------------|\n"
            
            # Sort by count and display
            for org, count in org_counts.items():
                percentage = (count / total_users * 100)
                response += f"| {str(org)} | {int(count):,} | {percentage:.1f}% |\n"
            
            response += f"\n**Total Organizations:** {len(org_counts):,}"
            response += f"\n**Total Users:** {total_users:,}"
            
            return response
            
        elif any(x in query.lower() for x in ["show users", "users from", "list users in"]):
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Extract location name from query
            query_lower = query.lower()
            location = None
            
            # Define location patterns to check
            location_patterns = [
                "in ", "from ", "at ", "located in ", "based in ",
                "show users in ", "users from ", "list users in ",
                "show me users in ", "find users in ", "get users from "
            ]
            
            # Extract location using the longest matching pattern
            matched_patterns = [p for p in location_patterns if p in query_lower]
            if matched_patterns:
                # Use the longest matching pattern to avoid partial matches
                longest_pattern = max(matched_patterns, key=len)
                parts = query_lower.split(longest_pattern)
                if len(parts) > 1:
                    # Take the text after the pattern and clean it
                    location = parts[-1].strip().title()
                    # Remove any trailing words or punctuation
                    location = location.split(' and ')[0].split(' or ')[0].split(',')[0].strip()
            
            # Handle US variations
            us_variants = {
                "usa": "United States",
                "us": "United States",
                "united states": "United States",
                "america": "United States",
                "u.s.": "United States",
                "u.s.a.": "United States"
            }
            
            # Check if the extracted location is a US variant
            if location and location.lower() in us_variants:
                location = "United States"
            
            if location:
                # Try both location and organization matches (case-insensitive)
                location_lower = location.lower()
                country_users = clean_df[
                    # Match location and ensure name is not empty/null
                    (
                        (clean_df['last_ip_location'].fillna('').str.lower() == location_lower) |  # exact match
                        (clean_df['last_ip_location'].fillna('').str.lower().str.endswith(f", {location_lower}"))  # city, country match
                    ) &
                    (clean_df['name'].notna()) &  # name is not null
                    (clean_df['name'].str.strip() != '')  # name is not empty string
                ]
                
                # Sort by city name to group users by location
                users_to_show = country_users.sort_values('last_ip_location')
                
                if len(users_to_show) == 0:
                    return f"No users with names found in {location}"
                
                response = f"### 👥 Named Users in {location} ({len(users_to_show)} users)\n\n"
                
                # Add table headers
                response += "| Name | Organization | Location |\n"
                response += "|------|--------------|----------|\n"
                
                # Add user details in table format
                for _, user in users_to_show.iterrows():
                    name = str(user['name']) if pd.notna(user['name']) else 'No name'
                    org = str(user['organization']) if pd.notna(user['organization']) and user['organization'] != '' else 'No organization'
                    city = str(user['last_ip_location']) if pd.notna(user['last_ip_location']) else 'Unknown location'
                    response += f"| {name} | {org} | {city} |\n"
                
                return response
            else:
                return "Please specify a valid location or organization name"
            
        elif "users who read the most posts" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Sort by post reads and get top 10
            top_readers = clean_df.nlargest(10, 'posts_read')
            
            response = "### 📚 Top 10 Users by Posts Read\n\n"
            response += "| Rank | Username | Posts Read | Location | Registration Location |\n"
            response += "|------|----------|------------|----------|---------------------|\n"
            
            # Add user details in table format
            for rank, (_, user) in enumerate(top_readers.iterrows(), 1):
                username = str(user['username']) if pd.notna(user['username']) else 'No username'
                posts_read = int(user['posts_read'])
                location = str(user['last_ip_location']) if pd.notna(user['last_ip_location']) else 'Unknown'
                reg_location = str(user['registration_ip_location']) if pd.notna(user['registration_ip_location']) else 'Unknown'
                
                response += f"| {rank} | {username} | {posts_read:,} | {location} | {reg_location} |\n"
            
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