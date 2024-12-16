from community import community_louvain
import pandas as pd
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('linked_cleaned_leaders_data.csv')

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
leader_to_country = dict(zip(df['head of government'], df['Country']))

# Get the list of world leaders
df_names = df['head of government'].values

# Add edges to the graph
for index, row in df.iterrows():
    current_leader = row['head of government']
    current_link = row['Wikipedia_Link']
    
    if pd.isna(current_link):
        continue
    
    matching_links = find_leader_links(current_link, df_names)
    for linked_leader_name in matching_links:
        if current_leader != linked_leader_name:
            G.add_edge(current_leader, linked_leader_name)

# Centrality metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

# Community detection
G_undirected = G.to_undirected()
partition = community_louvain.best_partition(G_undirected)

# Visualization
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_edges(G, pos, edge_color='green')
nx.draw_networkx_labels(G, pos, labels={node: leader_to_country.get(node, node) for node in G.nodes()}, font_size=8)
plt.title("World Leaders Network")
plt.show()
