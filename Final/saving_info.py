import pandas as pd
import requests
from bs4 import BeautifulSoup
import networkx as nx
from community import community_louvain

# Original file to build the graph
input_file_path = 'linked_cleaned_leaders_data.csv'
df_original = pd.read_csv(input_file_path)

# Output file
output_file_path = 'df_leaders.csv'
df_leaders = pd.read_csv(output_file_path)

G = nx.DiGraph()

# Function to scrape Wikipedia page and find links before the "References" section
def find_leader_links(wikipedia_url, df_names):
    response = requests.get(wikipedia_url, headers={'User-Agent': 'Your User Agent'})
    soup = BeautifulSoup(response.content, 'html.parser')
    body_content = soup.find(id="bodyContent")
    references_section = body_content.find(id="References") if body_content else None
    links = []

    if references_section:
        for element in references_section.find_all_previous():
            if element.name == 'a' and 'href' in element.attrs and element['href'].startswith('/wiki/'):
                links.append(element['href'])
    else:
        links = [a['href'] for a in body_content.find_all('a', href=True) if a['href'].startswith('/wiki/')]

    matching_links = []
    for link in links:
        linked_leader_name = link.split('/')[-1].replace('_', ' ').title()
        if linked_leader_name in df_names and linked_leader_name not in matching_links:
            matching_links.append(linked_leader_name)

    return matching_links

# Map leaders to countries
leader_to_country = dict(zip(df_original['head of government'], df_original['Country']))

# Get list of world leaders
df_names = df_original['head of government'].values

# Add edges to graph
for index, row in df_original.iterrows():
    current_leader = row['head of government']
    current_link = row['Wikipedia_Link']
    
    if pd.isna(current_link):
        continue
    
    matching_links = find_leader_links(current_link, df_names)
    for linked_leader_name in matching_links:
        if current_leader != linked_leader_name:
            G.add_edge(current_leader, linked_leader_name)

# Community detection
G_undirected = G.to_undirected()
partition = community_louvain.best_partition(G_undirected)

# Centrality metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

# Create a dictionary to store the updated metrics
updated_data = []
for leader in G.nodes:
    country = leader_to_country.get(leader, 'Unknown Country')
    updated_data.append({
        'Country': country,
        'Community': partition.get(leader, -1),
        'Degree Centrality': degree_centrality.get(leader, 0),
        'Betweenness Centrality': betweenness_centrality.get(leader, 0),
        'Eigenvector Centrality': eigenvector_centrality.get(leader, 0)
    })

updated_df = pd.DataFrame(updated_data)

# Update existing columns in df_leaders with the new values
df_leaders.set_index('Country', inplace=True)
updated_df.set_index('Country', inplace=True)

# Update or add the new columns
for column in ['Community', 'Degree Centrality', 'Betweenness Centrality', 'Eigenvector Centrality']:
    df_leaders[column] = updated_df[column]

df_leaders.reset_index(inplace=True)

# Save the updated DataFrame back to the same file
df_leaders.to_csv(output_file_path, index=False)
print("\nUpdated df_leaders.csv with community and centrality metrics.")
