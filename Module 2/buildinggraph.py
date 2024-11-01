import pandas as pd
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('/Users/niels/code/414/module 2/linked_cleaned_leaders_data.csv')

G = nx.DiGraph()

# Function to scrape Wikipedia page and find links before the "References" section
def find_leader_links(wikipedia_url, df_names):
    response = requests.get(wikipedia_url, headers={'User-Agent': 'Your User Agent'})
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Focus only on the main content of the Wikipedia page
    body_content = soup.find(id="bodyContent")

    # Find the "References" section within the body content
    references_section = body_content.find(id="References") if body_content else None

    # Collect links
    links = []
    
    if references_section:
        # Get all elements before the "References" section
        for element in references_section.find_all_previous():
            if element.name == 'a' and 'href' in element.attrs and element['href'].startswith('/wiki/'):
                links.append(element['href'])
    else:
        # If "References" section is not found, get all 'a' tags in the main content
        links = [a['href'] for a in body_content.find_all('a', href=True) if a['href'].startswith('/wiki/')]

    # Process the links to get the linked name and check if it matches any in the dataframe
    matching_links = []
    for link in links:
        linked_leader_name = link.split('/')[-1].replace('_', ' ').title()
        if linked_leader_name in df_names and linked_leader_name not in matching_links:
            matching_links.append(linked_leader_name)

    return matching_links

# Get the list of world leaders
df_names = df['head of government'].values

# Dictionaries to count outgoing and incoming edges for each leader
outgoing_edges_count = {}
incoming_edges_count = {}

# Iterate over each world leader and add to the graph
for index, row in df.iterrows():
    current_leader = row['head of government']
    current_link = row['Wikipedia_Link']
    
    if pd.isna(current_link):
        print(f"Skipping {current_leader} due to missing Wikipedia link.")
        continue
    
    # Scrape Wikipedia page of the current leader
    matching_links = find_leader_links(current_link, df_names)

    # Add edges between the current leader and the linked leaders
    for linked_leader_name in matching_links:
        # Prevent self-references
        if current_leader != linked_leader_name:
            G.add_edge(current_leader, linked_leader_name, weight=1)

            # Count outgoing edges for the current leader
            outgoing_edges_count[current_leader] = outgoing_edges_count.get(current_leader, 0) + 1
            
            # Count incoming edges for the linked leader
            incoming_edges_count[linked_leader_name] = incoming_edges_count.get(linked_leader_name, 0) + 1

# Calculate centrality metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

# Identify top 20 most central leaders
top_20_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
top_20_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
top_20_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:20]

print("\nTop 20 Leaders by Degree Centrality:")
for leader, centrality in top_20_degree:
    print(f"{leader}: {centrality:.4f}")

print("\nTop 20 Leaders by Betweenness Centrality:")
for leader, centrality in top_20_betweenness:
    print(f"{leader}: {centrality:.4f}")

print("\nTop 20 Leaders by Eigenvector Centrality:")
for leader, centrality in top_20_eigenvector:
    print(f"{leader}: {centrality:.4f}")

# Optional: Display the graph summary
print(f"\nNumber of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Visualize the graph using matplotlib
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k = .5)  # Adjusted k value for layout
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_edges(G, pos, edge_color='green')
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("World Leaders Network")
plt.show()