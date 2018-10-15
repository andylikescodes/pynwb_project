import urllib.request
import zipfile
import os.path
import data
import no2nwb

from pynwb import NWBHDF5IO


# Download the NO dataset from the website
if not os.path.exists('./RecogMemory_MTL_release_v2'):
    os.path.isfile('RecogMemory_MTL_release_v2.zip')

    urllib.request.urlretrieve('https://datadryad.org/bitstream/handle/10255/dryad.163179/RecogMemory_MTL_release_v2.zip', \
                               'RecogMemory_MTL_release_v2.zip')

    zip_ref = zipfile.ZipFile('./RecogMemory_MTL_release_v2.zip', 'r')
    zip_ref.extractall('RecogMemory_MTL_release_v2')
    zip_ref.close()

    os.remove('./RecogMemory_MTL_release_v2.zip')

# Set data path
path_to_data = './RecogMemory_MTL_release_v2/Data'

# Create the NWB file and extract data from the original data format
NOdata = data.NOData(path_to_data)
nwbfile = no2nwb.no2nwb(NOdata, 5)

# Export and write the nwbfile
session_name = NOdata.sessions[6]['session']
io = NWBHDF5IO(session_name + '.nwb', mode='w')
io.write(nwbfile)
io.close()

