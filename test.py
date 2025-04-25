from obspy import read
import matplotlib.pyplot as plt

RAW_PATH = "data/raw/SAC_Data.sac"
PROCESSED_PATH = "processed_plots/001Hours-250Hz-1.0s-GPU/data/processed/processed_trace.sac"

# st = read (RAW_PATH)
st = read (PROCESSED_PATH)

print(st)

tr = st[0]
# tr_id = tr.id

data = tr.data
t =  tr.times()

fig, ax = plt.subplots(figsize = (16, 9), dpi = 300)
ax.plot(t, data, color="black")
ax.set_title(f"Trace: ")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.grid(True)
plt.tight_layout()
plt.savefig("SAC_Data_Plot_Processed.png")

# print("Trace ID:")
# print(tr_id)
# print("Network:", tr.stats.network)
# print("Station:", tr.stats.station)
# print("Location:", tr.stats.location)
# print("Channel:", tr.stats.channel)
# # print("Trace stats:", tr.stats)

# tr.stats.network = "MY_NETWORK"
# tr.stats.station = "MY_STATION"
# tr.stats.location = "MY_LOCATION"
# tr.stats.channel = "MY_CHANNEL"


# tr.write("SAC_Data.mseed", format="MSEED")

# mseed_st = read("./SAC_Data.mseed")
# mseed_tr = mseed_st[0]

# data = mseed_tr.data
# t =  mseed_tr.times()


# fig, ax = plt.subplots(figsize = (16, 9), dpi = 300)
# ax.plot(t, data, color="black")
# ax.set_title(f"Trace: {trace_id}")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Amplitude")
# ax.grid(True)
# plt.tight_layout()
# plt.savefig("SAC_Data_Plot.png")

# sac_header = tr.stats.sac
# print(dict(sac_header))  # returns a dict_keys object
