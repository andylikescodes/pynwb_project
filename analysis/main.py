from analysis.behavior import *
from analysis.single_neuron import *
import logging

logging.basicConfig(filename='errorfiles.log')

# Plot the behavioral graphs
#plot_behavioral_graphs()

# Plot the single neuron analysis
# Find all filenames in the data directory
filenames = get_nwbfile_names("../data")
neurons = []

# Get all neurons from the nwbfiles
for filename in filenames:
    try:
        nwbfile = read(filename)
    except ValueError as e:
        print('Problem opening the file: ' + str(e))
        logging.warning('Error opening file: ' + filename)
        continue
    try:
        temp = extract_neuron_data_from_nwb(nwbfile)
    except IndexError as e:
        print("Somehow catch this index error: " + str(e))
        logging.warning('Error in extracting events, filename: '+ filename)
        continue
    neurons = neurons + extract_neuron_data_from_nwb(nwbfile)


# Find visually selective (VS) neurons and memory selective (MS) neurons
vs_neurons = []
ms_neurons = []
for neuron in neurons:
    if neuron.vs_test() < 0.05:
        vs_neurons.append(neuron)
    if neuron.ms_test(10000) < 0.05:
        ms_neurons.append(neuron)

# Plot the raster/psth first five VS neurons and first five MS neurons
for i in range(5):
    vs_neurons[i].raster_psth(cell_type='visual')

for i in range(5):
    ms_neurons[i].raster_psth(cell_type='memory')