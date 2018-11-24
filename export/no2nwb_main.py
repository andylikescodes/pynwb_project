import urllib.request
import zipfile
import os.path
from export import no2nwb, data

from pynwb import NWBHDF5IO


# Download the NO dataset from the website
if not os.path.exists('../RecogMemory_MTL_release_v2'):
    os.path.isfile('../RecogMemory_MTL_release_v2.zip')

    urllib.request.urlretrieve('https://datadryad.org/bitstream/handle/10255/dryad.163179/RecogMemory_MTL_release_v2.zip', \
                               'RecogMemory_MTL_release_v2.zip')

    zip_ref = zipfile.ZipFile('../RecogMemory_MTL_release_v2.zip', 'r')
    zip_ref.extractall('../RecogMemory_MTL_release_v2')
    zip_ref.close()

    os.remove('../RecogMemory_MTL_release_v2.zip')

# Set data path
path_to_data = '../RecogMemory_MTL_release_v2/Data'

# Create the NWB file and extract data from the original data format
NOdata = data.NOData(path_to_data)
# nwbfile = no2nwb.no2nwb(NOdata, 6)

# # Export and write the nwbfile
session_name = NOdata.sessions[6]['session']
# io = NWBHDF5IO('data/' + '/' + session_name + '.nwb', mode='w')
# io.write(nwbfile)
# io.close()

cell1 = NOdata.pop_cell(5, (1, 1))
#print(NOdata.ls_cells(5))
cell2 = NOdata.pop_cell(5, (2, 1))
cell3 = NOdata.pop_cell(5, (2, 2))

def extract_trials(cell):
    new_old = []
    spikes = []
    for trial in cell1.trials:
        new_old.append(trial.new_old_recog)
        spikes.append(trial.trial_timestamps_recog)

    return [spikes, new_old]

mat1 = extract_trials(cell1)
mat2 = extract_trials(cell2)
mat3 = extract_trials(cell3)

matfile = {"cell1": mat1, "cell2": mat2, "cell3": mat3}

import scipy.io as sio
sio.savemat("cells.mat", matfile)


