# **Description:**
# This Jupyter notebook is developed to show case the use of JSON header files for faster data access to NWM files. Here's a brief overview of the imported modules:
#
# - `joblib`: Used for parallel processing and efficient caching.
# - `numpy`: A fundamental library for numerical operations.
# - `xarray`: A library for working with labeled multi-dimensional arrays, often used in scientific data analysis.
# - `fsspec`: Provides a common interface for working with various filesystem-like protocols.
# - `ujson`: A fast JSON encoder and decoder for handling JSON data.
# - `matplotlib.pyplot`: Used for creating data visualizations.
# - `psutil`: Provides information on system resource utilization.
# - `concurrent.futures`: A module for asynchronously executing functions using threads or processes.
# - `multiprocessing`: A library for parallel and concurrent computing.
# - `urlgennwm`: An external library/module for generating URLs specific to the NWM (National Water Model).
#
#

import re
from datetime import datetime
import joblib
import numpy as np
import xarray as xr
import fsspec
import ujson
import matplotlib.pyplot as plt
import psutil
import concurrent.futures
import multiprocessing
import urlgennwm

from show_util import show, clear_display


#
# This code snippet retrieves and calculates system information, including the number of CPU cores and available memory in gigabytes (GB). It also computes a memory limit per worker process based on available resources. These metrics are essential for optimizing and parallelizing computational tasks.
#

# Get the number of available CPU cores
num_cores = psutil.cpu_count(logical=False)  # Use logical=True for hyperthreading

# Calculate the available memory in GB
available_memory_gb = psutil.virtual_memory().available / (1024**3)  # Bytes to GB

# Calculate a memory limit per worker based on available memory
# Adjust this factor based on your memory usage requirements
memory_per_worker_gb = available_memory_gb / num_cores

# This code defines a set of functions for working with remote datasets in JSON format. It includes functions to load remote JSON content, open datasets from JSON, select streamflow data, and select time values. The `process_file` function combines these operations to process a single file for a given feature ID, making it useful for retrieving and working with specific data from remote sources.


# Define a function to load remote JSON content
def load_remote_json(file_url):
    of = fsspec.open(file_url)
    with of as f:
        return ujson.load(f)


# Define a function to load a remote dataset from JSON content
def load_remote_ds(json_obj):
    backend_args = {
        "consolidated": False,
        "storage_options": {
            "fo": json_obj,
        },
    }
    return xr.open_dataset("reference://", engine="zarr", backend_kwargs=backend_args)


# Define a function to select streamflow data from a dataset
def select_flow(ds, feature_id):
    cords = ds.streamflow.sel(feature_id=feature_id)
    return cords.values


# Define a function to select time
def select_time(ds):
    dstime = ds.time
    return dstime.values


# Define a function to process a single file for a given feature ID
def _process_file(file_url, feature_id):
    json_obj = load_remote_json(file_url)
    ds = load_remote_ds(json_obj)
    streamflow_value = select_flow(ds, feature_id)
    time_value = select_time(ds)
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    return ds.to_dataframe(), streamflow_value, time_value


def process_file(file_url, feature_id):
    """
    Processes a single NWM Zarr reference file to extract streamflow,
    data timestamp, initialization timestamp, and initialization time label.

    Args:
        file_url (str): The URL to the NWM Zarr reference JSON file.
        feature_id (int): The ID of the river segment (feature) to extract data for.

    Returns:
        tuple: A tuple containing:
            - streamflow_value (np.array): Streamflow data for the specified feature_id.
            - data_timestamp (np.datetime64): The valid time of the forecast data.
            - init_timestamp (np.datetime64): The initialization time of the model run.
            - init_time_label (str): A string label for the initialization time (e.g., 't08z').
    """
    json_obj = load_remote_json(file_url)
    ds = load_remote_ds(json_obj)

    # Extract the streamflow value for the given feature_id
    streamflow_value = select_flow(ds, feature_id)

    # Extract the valid time of the data value
    data_timestamp = select_time(ds)

    # Extract initialization date and hour from the file_url
    # You have got to be kidding me -- is the only way to get
    # the initialization time from the filename???
    # Example URL: .../nwm.20250704/short_range/nwm.t08z.short_range...
    date_match = re.search(r"nwm\.(\d{8})", file_url)
    time_match = re.search(r"nwm\.t(\d{2})z", file_url)

    init_timestamp = None
    init_time_label = "unknown"  # Default label if parsing fails

    if date_match and time_match:
        date_str = date_match.group(1)  # e.g., '20250704'
        hour_str = time_match.group(1)  # e.g., '08'

        try:
            # Construct the initialization datetime object
            init_timestamp_dt = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")
            init_timestamp = np.datetime64(
                init_timestamp_dt
            )  # Convert to numpy datetime64
            init_time_label = f"t{hour_str}z"
        except ValueError:
            print(
                f"Warning: Could not parse date/time for initialization from URL: {file_url}"
            )

    return streamflow_value, data_timestamp, init_timestamp, init_time_label


#
# This code defines a list of feature IDs, which are typically used as identifiers for specific geographic or data features. In this case, the list includes feature IDs `8153461`, `8153027`, and `18210860`, representing specific features of interest. These feature IDs are used in subsequent data retrieval. Feature IDs could for different locations could be found at https://water.noaa.gov/map#forecast-chart
#

# Define a list of feature IDs
feature_ids = [3589508, 3585872, 3587620, 3586280, 3585816, 3585750, 3585606, 3585620]
# 2589508, Location of highest flood extent in HAND maps
# 3585872, Cypress River outlet
# 3587620, Cypress River upstream
# 3586280, South Fork Guadalupe just above confluence with Cypress Creek
# 3585816, South Fork Guadalupe just below confluence with Cypress Creek
# 3585750  South Fork Guadalupe River just above confluence with North Fork
# 3585606  North Fork Guadalupe River just above confluence with South Fork
# 3585620  Guadalupe River at Hunt (just below North and South forks confluence; Gage site: https://waterdata.usgs.gov/nwis/inventory/?site_no=08165500)

legend_ncols = len(feature_ids)  # Number of columns desired for the plot legends

# **Description:**
# This code block sets input variables to specify the parameters required for generating National Water Model (NWM) JSON header URLs. These URLs are utilized to access data related to the NWM for various configurations. Here's an overview of the input variables:
#
# - `start_date`: A string representing the starting date in the format "YYYYMMDDHHMM."
# - `end_date`: A string representing the ending date in the same format.
# - `fcst_cycle`: A list of integers specifying forecast cycle numbers, e.g., `[0, 1, 2, 3, 4]`. These cycles represent specific points in time for which URLs will be generated.
# - `lead_time`: A list of integers indicating lead times in hours for forecasts. It determines the time ahead of the forecast start, e.g., `[1, 2, 3, 4]`.
# - `varinput`: An integer or string representing the variable of interest within the NWM data. Available options include:
#   - `1` or `"channel_rt"` for channel routing data.
#   - `2` or `"land"` for land data.
#   - `3` or `"reservoir"` for reservoir data.
#   - `4` or `"terrain_rt"` for terrain routing data.
#   - `5` or `"forcing"` for forcing data.
# - `geoinput`: An integer or string specifying the geographic region of interest. Options include:
#   - `1` or `"conus"` for the continental United States.
#   - `2` or `"hawaii"` for Hawaii.
#   - `3` or `"puertorico"` for Puerto Rico.
# - `runinput`: An integer or string representing the NWM run configuration. Available options include:
#   - `1` or `"short_range"` for short-range forecasts.
#   - `2` or `"medium_range"` for medium-range forecasts.
#   - `3` or `"medium_range_no_da"` for medium-range forecasts without data assimilation.
#   - `4` or `"long_range"` for long-range forecasts.
#   - `5` or `"analysis_assim"` for analysis-assimilation runs.
#   - `6` or `"analysis_assim_extend"` for extended analysis-assimilation runs.
#   - `7` or `"analysis_assim_extend_no_da"` for extended analysis-assimilation runs without data assimilation.
#   - `8` or `"analysis_assim_long"` for long analysis-assimilation runs.
#   - `9` or `"analysis_assim_long_no_da"` for long analysis-assimilation runs without data assimilation.
#   - `10` or `"analysis_assim_no_da"` for analysis-assimilation runs without data assimilation.
#   - `11` or `"short_range_no_da"` for short-range forecasts without data assimilation.
#
# After defining these parameters, the code calls the `urlgennwm.generate_urls` function, passing in these variables as arguments. This function generates a list of URLs tailored to the specified parameters, allowing the retrieval of specific NWM data for further analysis or modeling.
#


# # Setting input variables to generate NWM JSON header urls
# start_date = "202507040000"
# end_date   = "202507040000"
# # fcst_cycle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# fcst_cycle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# # fcst_cycle = [8,10]
# lead_time = lead_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# # lead_time = lead_time = [1, 2, 3, 4,]
# varinput = 1
# geoinput = 1
# runinput = 1

# Define the name of the file containing the URLs
filename = "filenamelist.txt"
file_dest = f"dist/{filename}"

# urlgennwm.generate_urls(start_date, end_date, fcst_cycle, lead_time, varinput, geoinput, runinput, target_file=file_dest)
urlgennwm.generate_urls(
    start_date="202507040000",
    end_date="202507040000",
    fcst_cycle=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    lead_time=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    varinput=1,
    geoinput=1,
    runinput=1,
    target_file=file_dest,
)

# This code snippet initializes an empty list called `files` to store URLs. It then specifies the name of a file, "filenamelist.txt," which contains a list of URLs. The code opens this file in read mode, reads the URLs line by line, removes leading and trailing whitespace from each line, and appends each URL to the `files` list. Finally, it initializes an empty dictionary called `extracted_values_dict`,  intended to store extracted values associated with specific feature IDs.

# Initialize an empty list to store the URLs
files = []

# Open the file in read mode and read the URLs line by line
with open(file_dest, "r") as file:
    for line in file:
        # Remove leading and trailing whitespace and append the URL to the list
        url = line.strip()
        files.append(url)

# Define the name of the file containing the URLs
forcing_filename = "filenamelist_forcing.txt"
forcing_file_dest = f"dist/{forcing_filename}"

# # Setting input variables to generate NWM JSON header urls FOR FORCING/PRECIP
# start_date = "202507040000"
# end_date   = "202507040000"
# # fcst_cycle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# fcst_cycle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# lead_time = lead_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# # lead_time = lead_time = [1, 2, 3, 4,]
# varinput = 5
# geoinput = 1
# runinput = 1

# urlgennwm.generate_urls(start_date, end_date, fcst_cycle, lead_time, varinput, geoinput, runinput, target_file=file_dest)
urlgennwm.generate_urls(
    start_date="202507040000",
    end_date="202507040000",
    fcst_cycle=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    lead_time=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    varinput=5,
    geoinput=1,
    runinput=1,
    target_file=forcing_file_dest,
)


# Initialize an empty list to store the URLs
files_forcing = []

# Open the file in read mode and read the URLs line by line
with open(forcing_file_dest, "r") as file:
    for line in file:
        # Remove leading and trailing whitespace and append the URL to the list
        url = line.strip()
        files_forcing.append(url)
# # Create a dictionary to store the extracted values for each feature ID
# extracted_values_dict = {}

show(files_forcing[0:10])  # Display the first 10 URLs for brevity

show(files[0:10])  # Display the first 10 URLs for brevity

# This code snippet utilizes the `joblib` library to create a parallel processing pool using threading for dynamic settings. It processes a list of `files` using the `process_file` function in parallel, passing each `file_url` and the `feature_ids` list as arguments. The results are stored in `result_values` and sorted based on the first element of each result tuple. The extracted streamflow values are then stored in the `Streamflow` list, and the associated timestamps are stored in the `Timestamp` list, which can be further used for analysis or visualization.

# file = files[0]
# print(file)
# print(feature_ids)
# process_file(file, feature_ids)

import pickle
from pathlib import Path
from time import perf_counter

t0 = perf_counter()
result_values_cache = Path("dist/result_values.pkl")
try:
    result_values = pickle.load(result_values_cache.open("rb"))
    print(
        f"Loaded {len(result_values)} result values from pickle file in {perf_counter() - t0:.2f} seconds."
    )
except FileNotFoundError:
    t1 = perf_counter()
    print(
        f"Failed to load result values from pickle file in {perf_counter() - t0:.2f} seconds. Processing files directly..."
    )

    # Create a Parallel processing pool using joblib with dynamic settings
    with joblib.parallel_backend(
        "threading", n_jobs=num_cores
    ) as p:  # Use threads for parallel processing
        print(f"Created {type(p).__name__} with {num_cores} jobs.")
        result_values = joblib.Parallel()(
            joblib.delayed(process_file)(file_url, feature_ids) for file_url in files
        )

    result_values_cache.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the directory exists
    with result_values_cache.open("wb") as f:
        pickle.dump(result_values, f)
    t2 = perf_counter()
    print(
        f"Processed {len(result_values)} files in {t2 - t1:.2f} seconds ({t2 - t0:.2f} total). Results saved to {result_values_cache}."
    )

# # Create a Parallel processing pool using joblib with dynamic settings
# with joblib.parallel_backend("threading", n_jobs=num_cores):  # Use threads for parallel processing
#     result_values = joblib.Parallel()(joblib.delayed(process_file)(file_url, feature_ids) for file_url in files)

# Sort and process the result values
result_values = sorted(result_values, key=lambda x: x[1][0])
Streamflow = [item[0] for item in result_values]
Timestamp = [item[1] for item in result_values]


# result_values

# This code snippet creates separate plots for each feature ID in the `feature_ids` list. For each feature ID, it plots the time series of the extracted streamflow values using Matplotlib. The streamflow values are stored in the `Streamflow` list, and each plot is labeled with the corresponding feature ID. The x-axis represents time in hours, and the y-axis represents streamflow values. Finally, the code displays all the separate plots.
#

# Create separate plots for each feature ID
# Plot the time series of the extracted streamflow values for the current feature ID using Matplotlib
plt.figure()
plt.plot(Streamflow, label=[f"Feature ID {feature_id}" for feature_id in feature_ids])
plt.xlabel(f"Time(hours after {min(Timestamp)})")
# plt.xlabel("Time(hours)")
plt.ylabel("Streamflow")
plt.title(f"Streamflow Time Series")
plt.legend()  # This one doesn't need immediate adjustment
# Show all the separate plots
show(plt)

result_values = sorted(result_values, key=lambda x: x[1][0])  # Sort by data timestamp

# --- Data Separation ---

# Dynamically find all unique initialization time labels
all_init_time_labels = sorted(list(set(item[3] for item in result_values)))

# Initialize a dictionary to store data separated by initialization time and then by feature ID
# Structure: {init_time_label: {feature_id: [{'data_timestamp': dt, 'streamflow_value': val}, ...]}}
separated_data = {
    init_label: {fid: [] for fid in feature_ids} for init_label in all_init_time_labels
}
import itertools

# separated_data = dict(itertools.islice(separated_data.items(), 7))

# Iterate through the result_values and separate based on the initialization time label
# The structure of each item is: (streamflow_values_array, data_timestamp, init_timestamp, init_time_label)
for (
    streamflow_values_array,
    data_timestamp_array,
    init_timestamp_value,
    init_time_label,
) in result_values:
    # Convert numpy.datetime64 to a standard Python datetime object for plotting
    data_timestamp_dt = data_timestamp_array[0].astype(datetime)

    if init_time_label in separated_data:
        # Assuming streamflow_values_array contains data for all feature_ids in order
        for i, feature_id in enumerate(feature_ids):
            if i < len(streamflow_values_array):  # Ensure index is within bounds
                data_entry = {
                    "data_timestamp": data_timestamp_dt,
                    "streamflow_value": streamflow_values_array[i],
                    "init_timestamp": init_timestamp_value,
                    "init_time_label": init_time_label,
                }
                separated_data[init_time_label][feature_id].append(data_entry)
            else:
                print(
                    f"Warning: Streamflow array for {init_time_label} at {data_timestamp_dt} has fewer elements than feature_ids list."
                )
    else:
        print(
            f"Warning: Unknown initialization time label '{init_time_label}' encountered for data at {data_timestamp_dt}."
        )

# Ensure the separated lists are sorted by data timestamp (important for plotting)
for init_label in separated_data:
    for fid in separated_data[init_label]:
        separated_data[init_label][fid] = sorted(
            separated_data[init_label][fid], key=lambda x: x["data_timestamp"]
        )

# --- Create the plots for all data in a single window ---

plt.figure(figsize=(15, 8))  # Set a single figure size

# Define a color palette for different feature IDs
colors = plt.cm.get_cmap(
    "tab10", len(feature_ids)
)  # Using a colormap for distinct colors

# Define a list of linestyles and markers to cycle through for initialization times
# This allows handling an arbitrary number of initialization times
linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]  # More styles
markers = ["o", "x", "s", "^", "D", "P"]  # More markers

# Iterate through each feature ID and then each initialization time to plot
for i, feature_id in enumerate(feature_ids):
    feature_color = colors(i)  # Get a unique color for each feature ID

    for j, init_label in enumerate(all_init_time_labels):
        data_for_plot = separated_data[init_label][feature_id]

        # Get linestyle and marker for the current initialization time, cycling if needed
        current_linestyle = linestyles[j % len(linestyles)]
        current_marker = markers[j % len(markers)]

        data_timestamps_plot = [item["data_timestamp"] for item in data_for_plot]
        streamflows_plot = [item["streamflow_value"] for item in data_for_plot]

        plt.plot(
            data_timestamps_plot,
            streamflows_plot,
            color=feature_color,
            linestyle=current_linestyle,
            marker=current_marker,
            label=f"Feature ID {feature_id} ({init_label} Init)",
        )

# Add plot labels and title
plt.xlabel("Valid Time (UTC)", fontsize=12)
plt.ylabel("Streamflow (cms)", fontsize=12)
plt.title("Streamflow Forecasts by Initialization Time and Feature ID", fontsize=14)

# Add a grid for better readability
plt.grid(True, linestyle=":", alpha=0.7)

# Add a legend to distinguish the lines
# Place the legend outside the plot area if it gets too crowded
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10, ncol=legend_ncols)
# The legend when stacked via columns is manageably tall now, but still over doubles the width of the plot.
# This would be more efficient if it was placed directly below the plot, so that it doesn't take up so much horizontal space.
plt.legend(
    bbox_to_anchor=(0, -0.1),
    loc="upper left",
    borderaxespad=0.0,
    fontsize=10,
    ncol=legend_ncols,
)

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha="right")

# Adjust layout to prevent labels/legend from being cut off
# plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for the legend
# Above line produces the following warning:
## UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
# Need to adjust the layout to prevent labels/legend from being cut off somehow.
# The opposite of tight_layout?


# Display the plot
show(plt)

# Updated feature IDs and their descriptive names
feature_names = {
    3585872: "Cypress River outlet",
    3587620: "Cypress River upstream",
    3586280: "South Fork Guadalupe just above confluence with Cypress Creek",
    3585816: "South Fork Guadalupe just below confluence with Cypress Creek",
    3585750: "South Fork Guadalupe River just above confluence with North Fork",
    3585606: "North Fork Guadalupe River just above confluence with South Fork",
    3585620: "Guadalupe River at Hunt (just below North and South forks confluence; Gage site: https://waterdata.usgs.gov/nwis/inventory/?site_no=08165500)",
}
# --- Data Separation ---

# Dynamically find all unique initialization time labels
all_init_time_labels = sorted(list(set(item[3] for item in result_values)))

# Initialize a dictionary to store data separated by initialization time and then by feature ID
# Structure: {init_time_label: {feature_id: [{'data_timestamp': dt, 'streamflow_value': val}, ...]}}
separated_data = {
    init_label: {fid: [] for fid in feature_ids} for init_label in all_init_time_labels
}

# Iterate through the result_values and separate based on the initialization time label
# The structure of each item is: (streamflow_values_array, data_timestamp, init_timestamp, init_time_label)
for (
    streamflow_values_array,
    data_timestamp_array,
    init_timestamp_value,
    init_time_label,
) in result_values:
    # Convert numpy.datetime64 to a standard Python datetime object for plotting
    data_timestamp_dt = data_timestamp_array[0].astype(datetime)

    if init_time_label in separated_data:
        # Assuming streamflow_values_array contains data for all feature_ids in order
        for i, feature_id in enumerate(feature_ids):
            if i < len(streamflow_values_array):  # Ensure index is within bounds
                data_entry = {
                    "data_timestamp": data_timestamp_dt,
                    "streamflow_value": streamflow_values_array[i],
                    "init_timestamp": init_timestamp_value,
                    "init_time_label": init_time_label,
                }
                separated_data[init_time_label][feature_id].append(data_entry)
            else:
                print(
                    f"Warning: Streamflow array for {init_time_label} at {data_timestamp_dt} has fewer elements than feature_ids list."
                )
    else:
        print(
            f"Warning: Unknown initialization time label '{init_time_label}' encountered for data at {data_timestamp_dt}."
        )

# Ensure the separated lists are sorted by data timestamp (important for plotting)
for init_label in separated_data:
    for fid in separated_data[init_label]:
        separated_data[init_label][fid] = sorted(
            separated_data[init_label][fid], key=lambda x: x["data_timestamp"]
        )

# --- Create the plots for all data in a single window ---

plt.figure(figsize=(15, 8))  # Set a single figure size

# Define a color palette for different feature IDs
colors = plt.cm.get_cmap(
    "tab10", len(feature_ids)
)  # Using a colormap for distinct colors

# Define a list of linestyles and markers to cycle through for initialization times
# This allows handling an arbitrary number of initialization times
linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]  # More styles
markers = ["o", "x", "s", "^", "D", "P"]  # More markers

# Iterate through each feature ID and then each initialization time to plot
for i, feature_id in enumerate(feature_ids):
    feature_color = colors(i)  # Get a unique color for each feature ID
    feature_name = feature_names.get(feature_id, f"Unknown Location ({feature_id})")

    for j, init_label in enumerate(all_init_time_labels):
        data_for_plot = separated_data[init_label][feature_id]

        # Get linestyle and marker for the current initialization time, cycling if needed
        current_linestyle = linestyles[j % len(linestyles)]
        current_marker = markers[j % len(markers)]

        data_timestamps_plot = [item["data_timestamp"] for item in data_for_plot]
        streamflows_plot = [item["streamflow_value"] for item in data_for_plot]

        plt.plot(
            data_timestamps_plot,
            streamflows_plot,
            color=feature_color,
            linestyle=current_linestyle,
            marker=current_marker,
            label=f"{feature_name} ({init_label} Init)",
        )

# Add plot labels and title
plt.xlabel("Valid Time (UTC)", fontsize=12)
plt.ylabel("Streamflow (cms)", fontsize=12)
plt.title(
    "Streamflow Forecasts by Initialization Time and Feature Location", fontsize=14
)

# Add a grid for better readability
plt.grid(True, linestyle=":", alpha=0.7)

# Add a legend to distinguish the lines
# Place the legend outside the plot area if it gets too crowded
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=10)

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha="right")

# Adjust layout to prevent labels/legend from being cut off
# plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for the legend

# Display the plot
show(plt)
