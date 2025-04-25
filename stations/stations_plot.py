from obspy.clients.fdsn import Client

# Get station info using ObsPy's FDSN client (Raspberry Shake)
client = Client("https://data.raspberryshake.org")
inventory = client.get_stations(network="AM", station="RFA80", level="channel")

# Plot the inventory and capture the figure object
fig = inventory.plot()

# Save the figure as a PNG file
fig.savefig("station_inventory_plot.png", dpi=300, figsize=(16, 9), bbox_inches='tight')

# Optionally, show the plot
fig.show()
