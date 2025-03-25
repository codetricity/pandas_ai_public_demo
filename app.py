import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import time
import logging
import random
# import numpy as np
import streamlit as st
# import hashlib
import threading
import math


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
    st.title("Midori Masuda - Oppkey Assistant")

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
        ### Hello, I am a data analyst in Tokyo helping Oppkey partners make strategic business decisions
        """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                # Randomly select between the three Midori poses
                midori_pose = random.choice([
                    "images/midori_poses/midori_1.png",
                    "images/midori_poses/midori_2.png",
                    "images/midori_poses/midori_3.png"
                ])
                st.image(midori_pose, width=100)
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your dataset"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Show Midori's image while processing
            midori_pose = random.choice([
                "images/midori_poses/midori_1.png",
                "images/midori_poses/midori_2.png",
                "images/midori_poses/midori_3.png"
            ])
            st.image(midori_pose, width=100)

            # Get the response
            response = execute_query(prompt)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


def execute_query(query):
    """Function to safely execute query"""
    log_step(f"Sending query: '{query}'")
    start_time = time.time()
    
    try:
        if "most active users" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Sort by post reads and get top 20
            top_readers = clean_df.nlargest(20, 'posts_read')
            
            response = "### 📚 Most Active Users\n\n"
            response += "| Rank | Username | Posts Read | Location | Organization |\n"
            response += "|------|----------|------------|----------|--------------|\n"
            
            # Add user details in table format
            for rank, (_, user) in enumerate(top_readers.iterrows(), 1):
                username = str(user['username']) if pd.notna(user['username']) else 'No username'
                posts_read = int(user['posts_read'])
                location = str(user['last_ip_location']) if pd.notna(user['last_ip_location']) else 'Unknown'
                org = str(user['organization']) if pd.notna(user['organization']) and user['organization'] != '' else 'No organization'
                
                response += f"| {rank} | {username} | {posts_read:,} | {location} | {org} |\n"
            
            # Add summary statistics
            total_posts = clean_df['posts_read'].sum()
            avg_posts = clean_df['posts_read'].mean()
            response += f"\n**Total Posts Read:** {total_posts:,}"
            response += f"\n**Average Posts per User:** {avg_posts:.1f}"
            
            # Create analysis DataFrame
            analysis_df = pd.DataFrame({
                'Username': top_readers['username'],
                'Posts Read': top_readers['posts_read'],
                'Location': top_readers['last_ip_location'],
                'Organization': top_readers['organization']
            })
            
            # Add analysis section
            response += "\n\n### 📊 Analysis\n"
            analysis = analyze_table_with_openai(analysis_df, "Most active users in the dataset")
            response += analysis
            
            return response
            
        elif "organizations" in query.lower():
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Extract location if specified
            location = None
            if "in" in query.lower():
                location = query.lower().split("in")[-1].strip().title()
                
                # Handle US variations
                us_variants = {
                    "usa": "United States",
                    "us": "United States",
                    "the united states": "United States",
                    "united states": "United States",
                    "america": "United States",
                    "u.s.": "United States",
                    "u.s.a.": "United States"
                }
                
                if location.lower() in us_variants:
                    location = "United States"
                
                # Debug: Print unique location values
                log_step("Unique location values in data:")
                log_step(clean_df['last_ip_location'].unique())
                
                # Filter for location if specified
                if location:
                    # More flexible location matching
                    location_mask = (
                        (clean_df['last_ip_location'].fillna('').str.lower().str.contains('united states')) |
                        (clean_df['last_ip_location'].fillna('').str.lower().str.contains('usa')) |
                        (clean_df['last_ip_location'].fillna('').str.lower().str.contains('u.s.')) |
                        (clean_df['last_ip_location'].fillna('').str.lower().str.contains('america'))
                    )
                    
                    # Debug: Print matching locations
                    log_step("Matching locations:")
                    log_step(clean_df[location_mask]['last_ip_location'].unique())
                    
                    clean_df = clean_df[location_mask]
            
            # Define organizations to exclude (case-insensitive)
            exclude_orgs = {
                'oppkey', 'self', 'personal', 'individual', 'private', 'none', 
                'n/a', 'na', '', 'unknown', 'unemployed', 'student', 'freelance',
                'independent', 'retired', 'other', 'test', 'demo', 'example'
            }
            
            # Filter out excluded organizations
            clean_df = clean_df[
                ~clean_df['organization'].fillna('').str.lower().isin(exclude_orgs)
            ]
            
            # Get organization counts
            org_counts = clean_df['organization'].value_counts()
            total_users = len(clean_df)
            
            if total_users == 0:
                return f"No organizations found{f' in {location}' if location else ''}. This might be due to location matching or data filtering."
            
            response = "### 🏢 Organizations in the Dataset\n\n"
            response += "| Organization | Number of Users |\n"
            response += "|--------------|----------------|\n"
            
            # Create a DataFrame for analysis
            analysis_df = pd.DataFrame({
                'Organization': org_counts.index,
                'Number of Users': org_counts.values,
                'Percentage': (org_counts.values / total_users * 100).round(1)
            })
            
            # Sort by count and display
            for org, count in org_counts.items():
                percentage = float(count) / total_users * 100
                response += f"| {str(org)} | {int(count):,} ({percentage:.1f}%) |\n"
            
            response += f"\n**Total Organizations:** {len(org_counts):,}"
            response += f"\n**Total Users:** {total_users:,}"
            
            # Add analysis section
            context = f"Organization distribution{f' in {location}' if location else ''}"
            response += f"\n\n### 📊 Analysis\n"
            analysis = analyze_table_with_openai(analysis_df, context)
            response += analysis
            
            return response
            
        elif "countries" in query.lower():
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
            
            # Create a DataFrame for analysis
            analysis_df = pd.DataFrame({
                'Country': country_counts.index,
                'Number of Users': country_counts.values,
                'Percentage': (country_counts.values / total_users * 100).round(1)
            })
            
            # Sort by count and display
            for country, count in country_counts.items():
                percentage = float(count) / total_users * 100
                response += f"| {country} | {int(count):,} ({percentage:.1f}%) |\n"
            
            response += f"\n**Total Countries:** {len(country_counts):,}"
            response += f"\n**Total Users:** {total_users:,}"
            
            # Add analysis section
            response += "\n\n### 📊 Analysis\n"
            analysis = analyze_table_with_openai(analysis_df, "Country distribution of users in the dataset")
            response += analysis
            
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
        elif any(x in query.lower() for x in ["show users", "users from", "list users in"]):
            # Create a clean DataFrame for analysis
            clean_df = df.copy()
            
            # Debug: Print all unique locations at the start
            all_locations = clean_df['last_ip_location'].dropna().unique()
            log_step("All unique locations in dataset:")
            log_step(str(all_locations))
            
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
                    # Remove leading "The " if present
                    if location.lower().startswith('the '):
                        location = location[4:]
            
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
                # Debug: Print extracted location
                log_step(f"Searching for location: {location}")
                
                # More comprehensive US location matching
                if location == "United States":
                    # Debug: Print sample of data before matching
                    sample_locations = clean_df['last_ip_location'].dropna().head(20)
                    log_step("Sample of location data:")
                    log_step(str(sample_locations))
                    
                    location_mask = (
                        clean_df['last_ip_location'].fillna('').str.lower().str.contains('united states', case=False, na=False) |
                        clean_df['last_ip_location'].fillna('').str.lower().str.contains('usa', case=False, na=False) |
                        clean_df['last_ip_location'].fillna('').str.lower().str.contains('u.s.', case=False, na=False) |
                        clean_df['last_ip_location'].fillna('').str.lower().str.contains('america', case=False, na=False) |
                        clean_df['last_ip_location'].fillna('').str.lower().str.contains(' us', case=False, na=False) |
                        clean_df['last_ip_location'].fillna('').str.lower().str.contains(', us', case=False, na=False)
                    )
                else:
                    # For other locations, use more flexible matching
                    location_mask = clean_df['last_ip_location'].fillna('').str.contains(location.lower(), case=False, na=False)
                
                # Debug: Print matching results
                matching_locations = clean_df[location_mask]['last_ip_location'].unique()
                log_step(f"Found {len(matching_locations)} matching locations:")
                log_step(str(matching_locations))
                
                # Debug: Print counts at each step
                total_before_name = len(clean_df[location_mask])
                log_step(f"Total users before name filter: {total_before_name}")
                
                # Check name column
                log_step("Sample of name column data:")
                log_step(str(clean_df['name'].head(20)))
                
                country_users = clean_df[
                    location_mask &
                    (clean_df['name'].notna()) &  # name is not null
                    (clean_df['name'].str.strip() != '')  # name is not empty string
                ]
                
                # Debug: Print counts after name filter
                total_after_name = len(country_users)
                log_step(f"Total users after name filter: {total_after_name}")
                
                # Sort by city name to group users by location
                users_to_show = country_users.sort_values('last_ip_location')
                
                if len(users_to_show) == 0:
                    # Debug: Show total users before filtering
                    return f"No users with names found in {location}. Found {total_before_name} total users but none had names. Debug info: {len(matching_locations)} matching locations found."
                
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
                
                # Add organization analysis
                if len(users_to_show) > 0:
                    # Create organization analysis DataFrame
                    org_analysis = pd.DataFrame({
                        'Organization': users_to_show['organization'].value_counts().index,
                        'Number of Users': users_to_show['organization'].value_counts().values,
                        'Percentage': (users_to_show['organization'].value_counts().values / len(users_to_show) * 100).round(1)
                    })
                    
                    # Add analysis section
                    response += f"\n\n### 📊 Organization Analysis for {location}\n"
                    analysis = analyze_table_with_openai(org_analysis, f"Organization distribution in {location}")
                    response += analysis
                
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

def analyze_table_with_openai(table_data, context):
    """Analyze table data using OpenAI with sales-focused insights"""
    try:
        # Format the DataFrame in a more readable way
        if isinstance(table_data, pd.DataFrame):
            # Convert DataFrame to a formatted string with proper alignment
            formatted_data = table_data.to_string(index=False)
            
            # Create context-appropriate summary
            if "most active users" in context.lower():
                summary = f"""
Summary Statistics:
- Total Users Analyzed: {len(table_data)}
- Top 5 Users by Posts Read: {', '.join(table_data.nlargest(5, 'Posts Read')['Username'].tolist())}
- Average Posts per User: {table_data['Posts Read'].mean():.1f}
- Users with >100 Posts: {len(table_data[table_data['Posts Read'] > 100])}
"""
            elif "organization" in context.lower():
                summary = f"""
Summary Statistics:
- Total Organizations: {len(table_data)}
- Top 5 Organizations by Users: {', '.join(table_data.nlargest(5, 'Number of Users')['Organization'].tolist())}
- Average Users per Organization: {table_data['Number of Users'].mean():.1f}
- Organizations with >10 Users: {len(table_data[table_data['Number of Users'] > 10])}
"""
            else:
                summary = f"""
Summary Statistics:
- Total Entries: {len(table_data)}
- Top 5 Countries by Users: {', '.join(table_data.nlargest(5, 'Number of Users')['Country'].tolist())}
- Average Users per Country: {table_data['Number of Users'].mean():.1f}
"""
        else:
            formatted_data = str(table_data)
            summary = ""
        
        # Create context-appropriate analysis prompt
        if "most active users" in context.lower():
            analysis_prompt = f"""As Midori Masuda, analyze this user activity data and provide sales-focused insights and recommendations:

Context: {context}

{summary}

Detailed Data:
{formatted_data}

Please provide a concise analysis focusing on:

1. User Engagement Insights:
   - Which users show the highest engagement levels?
   - What patterns exist in user activity across different locations?
   - Which organizations have the most active users?

2. Market Insights:
   - Geographic distribution of highly engaged users
   - Organization patterns among top users
   - Engagement trends across different regions

3. Strategic Recommendations:
   - How to leverage highly engaged users for growth
   - Suggested engagement strategies for different regions
   - Potential partnership opportunities with active users' organizations
   - Ways to increase engagement in less active regions

Keep the analysis practical and actionable, focusing on concrete steps to drive user engagement and sales growth."""
        elif "organization" in context.lower():
            analysis_prompt = f"""As Midori Masuda, analyze this organization data and provide sales-focused insights and recommendations:

Context: {context}

{summary}

Detailed Data:
{formatted_data}

Please provide a concise analysis focusing on:

1. Sales Opportunities:
   - Which organizations show the highest potential for expansion? (excluding internal/individual organizations)
   - Where are the largest untapped customer bases?
   - Which organizations have the most engaged users?
   - Identify organizations that could be high-value sales targets

2. Market Insights:
   - Key organization concentration patterns
   - Industry distribution and trends
   - Organization size and engagement patterns
   - Market segments with the most potential

3. Strategic Recommendations:
   - Priority organizations for sales team focus (excluding internal/individual organizations)
   - Suggested resource allocation
   - Specific growth opportunities
   - Potential partnership or expansion targets
   - Recommended sales approach for different organization types

Keep the analysis practical and actionable, focusing on concrete steps to drive sales growth. Exclude any internal organizations or individual users from the analysis."""
        else:
            analysis_prompt = f"""As Midori Masuda, analyze this data table and provide sales-focused insights and recommendations:

Context: {context}

{summary}

Detailed Data:
{formatted_data}

Please provide a concise analysis focusing on:

1. Sales Opportunities:
   - Which regions/countries show the highest potential for growth?
   - Where are the largest untapped markets?
   - Which areas have the most engaged users?

2. Market Insights:
   - Key market concentration patterns
   - Geographic distribution of user engagement
   - Regional market maturity indicators

3. Strategic Recommendations:
   - Priority regions for sales team focus
   - Suggested resource allocation
   - Specific growth opportunities
   - Potential partnership or expansion targets

Keep the analysis practical and actionable, focusing on concrete steps to drive sales growth."""

        # Get analysis from OpenAI with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use OpenAI client directly
                import openai
                openai.api_key = st.secrets["openai"]["api_key"]
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                analysis = response.choices[0].message.content
                if analysis and len(analysis.strip()) > 0:
                    return analysis
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retrying
        
        return "Unable to generate analysis at this time. Please try again."
        
    except Exception as e:
        logger.error(f"Error in table analysis: {str(e)}", exc_info=True)
        return f"Unable to generate analysis at this time. Error: {str(e)}"

if __name__ == "__main__":
    main()