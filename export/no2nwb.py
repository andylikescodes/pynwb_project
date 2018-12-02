import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.image import ImageSeries
from pynwb.ecephys import Clustering, ClusterWaveforms, FeatureExtraction
from pynwb.file import Subject
from pynwb.device import Device
import datetime
import cv2

def no2nwb(NOData, session_use, subjects):

    # Prepare the NO data that will be coverted to the NWB format

    session = NOData.sessions[session_use]
    events = NOData._get_event_data(session_use, experiment_type='All')
    cell_ids = NOData.ls_cells(session_use)
    experiment_id_learn = session['experiment_id_learn']
    experiment_id_recog = session['experiment_id_recog']
    task_descr = session['task_descr']

    # Get the metadata for the subject
    df_session = subjects[subjects['session_id'] == session_use]

    print('session_use')
    print(session_use)
    print('age')
    print(str(df_session['age'].values[0]))
    print('epilepsy_diagnosis')
    print(str(df_session['epilepsy_diagnosis'].values[0]))

    nwb_subject = Subject(age=str(df_session['age'].values[0]), description=df_session['epilepsy_diagnosis'].values[0],
                          sex=df_session['sex'].values[0], subject_id=df_session['subject_id'].values[0])

    # Create the NWB file
    nwbfile = NWBFile(
        #source='https://datadryad.org/bitstream/handle/10255/dryad.163179/RecogMemory_MTL_release_v2.zip',
        session_description='RecogMemory dataset session use 5' + session['session'],
        identifier=session['session_id'],
        session_start_time=datetime.datetime.now(),# TODO: need to check out the time for session start
        file_create_date=datetime.datetime.now(),
        experiment_description="learning: " + str(experiment_id_learn) + ", " + \
                               "recognition: " + \
                               str(experiment_id_recog),
        subject=nwb_subject
    )


    # Add event and experiment_id acquisition
    # event_ts = TimeSeries(name='events', source='NA', unit='NA', data=np.asarray(events[1].values),
    #                       timestamps=np.asarray(events[0].values))

    event_ts = TimeSeries(name='events', unit='NA', data=np.asarray(events[1].values),
                          timestamps=np.asarray(events[0].values))
    # experiment_ids = TimeSeries(name='experiment_ids', source='NA', unit='NA', data=np.asarray(events[2]),
    #                             timestamps=np.asarray(events[0].values))
    experiment_ids = TimeSeries(name='experiment_ids', unit='NA', data=np.asarray(events[2]),
                                timestamps=np.asarray(events[0].values))
    nwbfile.add_acquisition(event_ts)
    nwbfile.add_acquisition(experiment_ids)

    # Add stimuli to the NWB file2
    # Get the first cell from the cell list
    cell = NOData.pop_cell(session_use, NOData.ls_cells(session_use)[0])
    trials = cell.trials
    stimuli_recog_path = [trial.file_path_recog for trial in trials]
    stimuli_learn_path = [trial.file_path_learn for trial in trials]

    # Add stimuli recog
    counter = 1
    for path in stimuli_recog_path:
        folders = path.split('\\')
        path = os.path.join('./RecogMemory_MTL_release_v2', 'Stimuli', folders[0], folders[1], folders[2])
        img = cv2.imread(path)
        name = 'stimuli_recog_' + str(counter)
        stimulus_recog = ImageSeries(
            name=name,
            data=img,
            unit='NA',
            format='',
            timestamps=[0.0])

        nwbfile.add_stimulus(stimulus_recog)
        counter += 1

    # Add stimuli learn
    counter = 1
    for path in stimuli_learn_path:
        if path == 'NA':
            continue
        folders = path.split('\\')

        path = os.path.join('./RecogMemory_MTL_release_v2', 'Stimuli', folders[0], folders[1], folders[2])
        img = cv2.imread(path)

        name = 'stimuli_learn_' + str(counter)

        stimulus_learn = ImageSeries(
            name=name,
            data=img,
            unit='NA',
            format='',
            timestamps=[0.0])

        nwbfile.add_stimulus(stimulus_learn)

        counter += 1

    # Add epochs and trials: storing start and end times for a stimulus

    # First extract the category ids and names that we need
    # The metadata for each trials will be store in a trial table

    cat_id_recog = [trial.category_recog for trial in trials]
    cat_name_recog = [trial.category_name_recog for trial in trials]
    cat_id_learn = [trial.category_learn for trial in trials]
    cat_name_learn = [trial.category_name_learn for trial in trials]

    # Extract the event timestamps
    events_learn_stim_on = events[(events[2] == experiment_id_learn) & (events[1] == NOData.markers['stimulus_on'])]
    events_learn_stim_off = events[(events[2] == experiment_id_learn) & (events[1] == NOData.markers['stimulus_off'])]
    events_learn_delay1_off = events[(events[2] == experiment_id_learn) & (events[1] == NOData.markers['delay1_off'])]
    events_learn_delay2_off = events[(events[2] == experiment_id_learn) & (events[1] == NOData.markers['delay2_off'])]
    
    events_recog_stim_on = events[(events[2] == experiment_id_recog) & (events[1] == NOData.markers['stimulus_on'])]
    events_recog_stim_off = events[(events[2] == experiment_id_recog) & (events[1] == NOData.markers['stimulus_off'])]
    events_recog_delay1_off = events[(events[2] == experiment_id_recog) & (events[1] == NOData.markers['delay1_off'])]
    events_recog_delay2_off = events[(events[2] == experiment_id_recog) & (events[1] == NOData.markers['delay2_off'])]

    # Extract new_old label
    new_old_recog = [trial.new_old_recog for trial in trials]

    # Create the trial tables
    nwbfile.add_trial_column('stim_on', 'the time when the stimulus is shown')
    nwbfile.add_trial_column('stim_off', 'the time when the stimulus is off')
    nwbfile.add_trial_column('delay1_off', 'the time when delay1 is off')
    nwbfile.add_trial_column('delay2_off', 'the time when delay2 is off')
    nwbfile.add_trial_column('stim_phase', 'learning/recognition phase during the trial')
    nwbfile.add_trial_column('category_id', 'the category id of the stimulus')
    nwbfile.add_trial_column('category_name', 'the category name of the stimulus')
    nwbfile.add_trial_column('external_image_file', 'the file path to the stimulus')
    nwbfile.add_trial_column('new_old_labels_recog', 'labels for new or old stimulus')

    range_recog = np.amin([len(events_recog_stim_on), len(events_recog_stim_off), len(events_recog_delay1_off),
                           len(events_recog_delay2_off)])
    range_learn = np.amin([len(events_learn_stim_on), len(events_learn_stim_off), len(events_learn_delay1_off),
                           len(events_learn_delay2_off)])

    # Iterate the event list and add information into each epoch and trial table
    for i in range(range_learn):
        # nwbfile.create_epoch(start_time=events_learn_stim_on.iloc[i][0],
        #                      stop_time=events_learn_stim_off.iloc[i][0],
        #                      timeseries=[event_ts, experiment_ids],
        #                      tags='stimulus_learn',
        #                      description='learning phase stimulus')

        # nwbfile.add_trial({'start': events_learn_stim_on.iloc[i][0],
        #                    'end': events_learn_delay2_off.iloc[i][0],
        #                    'stim_on': events_learn_stim_on.iloc[i][0],
        #                    'stim_off': events_learn_stim_off.iloc[i][0],
        #                    'delay1_off': events_learn_delay1_off.iloc[i][0],
        #                    'delay2_off': events_learn_delay2_off.iloc[i][0],
        #                    'stim_phase': 'learn',
        #                    'category_id': cat_id_learn[i],
        #                    'category_name': cat_name_learn[i],
        #                    'external_image_file': stimuli_learn_path[i],
        #                    'new_old_labels_recog': -1})

        nwbfile.add_trial(start_time=events_learn_stim_on.iloc[i][0],
                          stop_time=events_learn_delay2_off.iloc[i][0],
                          stim_on=events_learn_stim_on.iloc[i][0],
                          stim_off=events_learn_stim_off.iloc[i][0],
                          delay1_off=events_learn_delay1_off.iloc[i][0],
                          delay2_off=events_learn_delay2_off.iloc[i][0],
                          stim_phase='learn',
                          category_id=cat_id_learn[i],
                          category_name=cat_name_learn[i],
                          external_image_file=stimuli_learn_path[i],
                          new_old_labels_recog='NA'
                         )

    for i in range(range_recog):
        # nwbfile.create_epoch(start_time=events_recog_stim_on.iloc[i][0],
        #                      stop_time=events_recog_stim_off.iloc[i][0],
        #                      timeseries=[event_ts, experiment_ids],
        #                      tags='stimulus_recog',
        #                      description='recognition phase stimulus')

        nwbfile.add_trial(start_time=events_recog_stim_on.iloc[i][0],
                          stop_time=events_recog_delay2_off.iloc[i][0],
                          stim_on=events_recog_stim_on.iloc[i][0],
                          stim_off=events_recog_stim_off.iloc[i][0],
                          delay1_off=events_recog_delay1_off.iloc[i][0],
                          delay2_off=events_recog_delay2_off.iloc[i][0],
                          stim_phase='recog',
                          category_id=cat_id_recog[i],
                          category_name=cat_name_recog[i],
                          external_image_file=stimuli_recog_path[i],
                          new_old_labels_recog=new_old_recog[i])

    # Add the waveform clustering and the spike data.
    # Create necessary processing modules for different kinds of waveform data
    clustering_processing_module = ProcessingModule('Spikes', 'The spike data contained')
    clusterWaveform_learn_processing_module = ProcessingModule('MeanWaveforms_learn',
                                                               'The mean waveforms for the clustered raw signal for learning phase')
    clusterWaveform_recog_processing_module = ProcessingModule('MeanWaveforms_recog',
                                                               'The mean waveforms for the clustered raw signal for recognition phase')
    IsolDist_processing_module = ProcessingModule('IsoDist',
                                                  'The IsolDist')
    SNR_processing_module = ProcessingModule('SNR',
                                             'SNR (signal-to-noise)')
    # Get the unique channel id that we will be iterate over
    channel_ids = np.unique([cell_id[0] for cell_id in cell_ids])

    # Interate the channel list
    for channel_id in channel_ids:
        cell_name = 'A' + str(channel_id) + '_cells.mat'
        file_path = os.path.join('RecogMemory_MTL_release_v2', 'Data', 'sorted', session['session'],
                                 task_descr, cell_name)
        try:
            cell_mat = loadmat(file_path)
        except FileNotFoundError:
            print("File not found")
            continue
        spikes = cell_mat['spikes']
        meanWaveform_recog = cell_mat['meanWaveform_recog']
        meanWaveform_learn = cell_mat['meanWaveform_learn']
        IsolDist_SNR = cell_mat['IsolDist_SNR']

        spike_id = np.asarray([spike[0] for spike in spikes])
        spike_cluster_id = np.asarray([spike[1] for spike in spikes])
        spike_timestamps = np.asarray([spike[2]/1000000 for spike in spikes])
        clustering = Clustering(description='Spikes of the channel detected',
                                num=spike_id, peak_over_rms=np.asarray([0]), times=spike_timestamps,
                                name='channel'+str(channel_id))
        clustering_processing_module.add_data_interface(clustering)

        for i in range(len(meanWaveform_learn[0][0][0][0])):
            waveform_mean_learn = ClusterWaveforms(clustering_interface=clustering,
                                                   waveform_filtering='NA',
                                                   waveform_sd=np.asarray([[0]]),
                                                   waveform_mean=np.asarray([meanWaveform_learn[0][0][1][i]]),
                                                   name='waveform_learn_cluster_id_'+str(meanWaveform_learn[0][0][0][0][i]))
            try:
                clusterWaveform_learn_processing_module.add_data_interface(waveform_mean_learn)
            except ValueError as e:
                print('Catch an error in adding waveform interface to the recog processing module:' + str(e))
                continue

        # Adding mean waveform recognition into the processing module
        for i in range(len(meanWaveform_recog[0][0][0][0])):
            waveform_mean_recog = ClusterWaveforms(clustering_interface=clustering,
                                                   waveform_filtering='NA',
                                                   waveform_sd=np.asarray([[0]]),
                                                   waveform_mean=np.asarray([meanWaveform_recog[0][0][1][i]]),
                                                   name='waveform_recog_cluster_id_'+str(meanWaveform_recog[0][0][0][0][i]))
            try:
                clusterWaveform_recog_processing_module.add_data_interface(waveform_mean_recog)
            except ValueError as e:
                print('Catch an error in adding waveform interface to the recog processing module:' + str(e))
                continue

        # Adding IsolDist_SNR data into the processing module
        # Here I use feature extraction to store the IsolDist_SNR data because
        # they are extracted from the original signals.
        # print(IsolDist_SNR[0][0][0])
        for i in range(len(IsolDist_SNR[0][0][1][0])):
            isoldist_data_interface = TimeSeries(data=[IsolDist_SNR[0][0][1][0][i]],
                                                 unit='NA',
                                                 timestamps=[0],
                                                 name='IsolDist_' + str(IsolDist_SNR[0][0][0][0][i]))
            try:
                IsolDist_processing_module.add_data_interface(isoldist_data_interface)
            except ValueError as e:
                print('Catch an error in adding IsolDist to the processing module:' + str(e))
                continue

            SNR_data_interface = TimeSeries(unit='NA',
                                            description='The SNR data',
                                            data=[IsolDist_SNR[0][0][2][0][i]],
                                            timestamps=[0],
                                            name='SNR_' + str(IsolDist_SNR[0][0][0][0][i]))

            try:
                SNR_processing_module.add_data_interface(SNR_data_interface)
            except ValueError as e:
                print('Catch an error in adding SNR to the processing module:' + str(e))
                continue

    nwbfile.add_processing_module(clustering_processing_module)
    nwbfile.add_processing_module(clusterWaveform_learn_processing_module)
    nwbfile.add_processing_module(clusterWaveform_recog_processing_module)
    nwbfile.add_processing_module(IsolDist_processing_module)
    nwbfile.add_processing_module(SNR_processing_module)

    return nwbfile



def create_stimulus():
    pass

# So the problem is, we will have 20 - 30 cells per sessions, these cells will have their own
# spike timestamps.
# Event data for the learning phase the recognition phase.

# So the event timeseries should contain the entire event time series, with the experiment id,
# event markers, and the timestamps

# We can actually just add trials.

