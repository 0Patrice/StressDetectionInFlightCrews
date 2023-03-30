# Stress Detection in FlightCrews
This is the repository to my master's thesis with the title: "Measurements of Physiological Parameters in Flight Simulator Studies for Stress Detection and Analysis through Deep Learning Applications" at the University of Stuttgart in cooperation with the Institute of Flight Guidance at the German Aerospace Centre (Deutsches Zentrum für Luft- und Raumfahrt e. V. (DLR)) Brunswick, Germany.

## Data
Consisting of CSV files from both flight simulator studies containing an ECG signal and corresponding stress level based on self evaluation of the participants.

### Single Pilot Operation Study

    ['Time', 'Value', 'Stress']
| Header | Description                         |
|--------|-------------------------------------|
| Time   | from record start in seconds        |
| Value  | ECG limb lead II                    |
| Stress | A/D converted stress level feedback |


### Limits of Human Performance Study

#### Resting ECG Header

    ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']

| Header            | Description                           |
|-------------------|---------------------------------------|
| I, II, II         | ECG limb leads according to Einthoven |
| aVR, aVL, aVF, C1 | Augmented ECG leads                   |
| Resp              | Respiration Curve                     |

#### Scenario Header

    ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp', 'UTC', 'DA', 'Conf']

| Header                             | Description                |
|------------------------------------|----------------------------|
| I, II, II, aVR, aVL, aVF, C1, Resp | Same as Resting ECG        |
| UTC                                | UTC time of signal         |
| DA                                 | Slider values from OSAT    |
| Conf                               | Confirmed Values from OSAT |

## OSAT
OSAT is a tool to track stress in pilots and was used first during the LoHP study. It utilizes a slider asking participants to assess their momentary stress level on a scale between 0 (resting) and 10 (maximum individual stress). It was designed using Ploty and Dash.


## SimplECG
Simple software tool developed using tkinter, matplotlib, nummpy, pandas and other python frameworks (see requirements). It can load raw data from the study as well as multi-record-CSV files. 

## LSTM
Contains source code of the deep learning system used in this thesis. Calculated losses and model states are included in the *losses* folder. Predictions output on the acquired data made by this system are included in the respective folder.

## Notebooks
Collection of Jupyter Notebooks used in order to explore the data, plot relevant data, and merge the SPO study's data.

## Debriefing Tool

A software tool that was developed in preparation of the LoHP study in order to discuss all caputred data with flight crews after their simulator flights. It displays various aircraft, physiological, and stress data at once and can be zoomed in particular time areas for discussion. 