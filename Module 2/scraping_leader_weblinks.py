import pandas as pd
import wikipediaapi

# Load the CSV file
file_path = '/Users/niels/code/414/module 2/cleaned_leaders_data.csv'  # Replace with the actual file path
leaders_df = pd.read_csv(file_path)

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(user_agent="nielsworldleaders")

def get_wikipedia_link(name):
    # Search for the page
    page = wiki_wiki.page(name)
    
    # Check if the page exists and return the URL
    if page.exists():
        return page.fullurl
    else:
        return None

# Create a new column for Wikipedia links
leaders_df['Wikipedia_Link'] = leaders_df['head of government'].apply(get_wikipedia_link)
print(leaders_df.head())
leaders_df.to_csv('linked_cleaned_leaders_data.csv', index=False)

