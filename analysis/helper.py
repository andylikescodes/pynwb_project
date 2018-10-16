from pynwb import NWBHDF5IO
import numpy as np
import re


def read(file_path):
    """
    read in files
    :param file_path:
    :return:
    """
    io = NWBHDF5IO(file_path)
    nwbfile = io.read()
    return nwbfile


def get_event_data(nwbfile):
    """
    Get event data from the nwbfile
    :param nwbfile:
    :return:
    """
    events = nwbfile.get_acquisition('events')
    experiment_id_list = np.asarray(nwbfile.get_acquisition('experiment_ids').data)
    events_data = np.asarray(events.data)
    events_timestamps = np.asarray(events.timestamps)

    experiment_description = nwbfile.experiment_description

    experiment_ids = re.findall(r'\d+', experiment_description)
    experiment_id_learn = int(experiment_ids[0])
    experiment_id_recog = int(experiment_ids[1])

    ind_learn = np.where(experiment_id_list == experiment_id_learn)
    ind_recog = np.where(experiment_id_list == experiment_id_recog)

    events_learn = events_data[ind_learn]
    timestamps_learn = events_timestamps[ind_learn]

    events_recog = events_data[ind_recog]
    timestamps_recog = events_timestamps[ind_recog]

    return events_learn, timestamps_learn, events_recog, timestamps_recog

def calcroc(nwbfile):
    """
    calculate the true positives, true negatives, false positives and false negatives
    :param nwbfile:
    :return:
    """
    events_learn, timestamps_learn, events_recog, timestamps_recog = get_event_data(nwbfile)
    response_recog_ind = np.where((events_recog >= 30) & (events_recog <= 36))
    response_recog = events_recog[response_recog_ind] - 30

    labels = np.asarray(nwbfile.trials['new_old_labels_recog'])
    new_old_labels = np.delete(labels, np.where(labels == 'NA')).astype(int)
    print(new_old_labels)