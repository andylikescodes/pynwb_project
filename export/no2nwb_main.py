import urllib.request
import zipfile
import os.path
from export import no2nwb, data
import pandas as pd
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

# Read in subject data
subjects = pd.read_csv('export/subjects.csv')

# Create the NWB file and extract data from the original data format
NOdata = data.NOData(path_to_data)

for session_nr in NOdata.sessions.keys():
    nwbfile = no2nwb.no2nwb(NOdata, session_nr, subjects)

    # Export and write the nwbfile
    session_name = NOdata.sessions[session_nr]['session']

    io = NWBHDF5IO('data/' + '/' + session_name + '_' + str(session_nr) + '.nwb', mode='w')
    io.write(nwbfile)
    io.close()


