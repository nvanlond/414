import folium

m = folium.Map(location=[39.0458, -76.6413], zoom_start=10)  # Coordinates for Maryland

m.save('basic_map.html')
