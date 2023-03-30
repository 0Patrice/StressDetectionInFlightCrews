from dash import Dash, html, dcc, html, Input, Output, ctx, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from flask_caching import Cache

import pandas as pd
import numpy as np
import os

from PIL import Image

import wfdb
from wfdb import processing

################################################
##      Set File Names
################################################
dirnameSim = "G:\Simulator\Crew6\SC2"

filenameSim = ""

for root, subdirectories, files in os.walk(dirnameSim, topdown=True):
    for file in files:
        if file.endswith('.reca'):
            if not file.startswith('.'):
                filenameSim = os.path.join(root, file)

fileNameECG_FO = "G:\EKG\Crew6\VP14_Scenario2.log"
fileNameECG_CP = "G:\EKG\Crew6\VP13_Scenario2.log"

fileNameECGRuhe_CP = "G:\EKG\Crew6\RuheEKG_VP13.log"
fileNameECGRuhe_FO = "G:\EKG\Crew6\RuheEKG_VP14.log"

# Define File Names for Stress Plots
crew = "Crew6"
scenario = "SC2"
timeslot = "10_18_9_36"

fileDataArrayCP = f"G:\Stressverlauf\{crew}\{scenario}\F_{timeslot}_CP_DA.csv"
fileDataArrayConfirmedCP = f"G:\Stressverlauf\{crew}\{scenario}\F_{timeslot}_CP_DConf.csv"

timeslotFO = "10_18_9_36"
fileDataArrayFO = f"G:\Stressverlauf\{crew}\{scenario}\F_{timeslotFO}_FO_DA.csv"
fileDataArrayConfirmedFO = f"G:\Stressverlauf\{crew}\{scenario}\F_{timeslotFO}_FO_DConf.csv"

################################################
##      App Setup
################################################

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
# app = Dash(external_stylesheets=[dbc.themes.FLATLY])
load_figure_template('BOOTSTRAP')


################################################
##      Flight Data Plots
################################################
def getData(filename, position):
    print(filename)
    myFile = np.genfromtxt(filename, delimiter=',', skip_header=126)
    return myFile[:, position]


# Load Data from Filename
myFile = np.genfromtxt(filenameSim, delimiter=',', skip_header=126)
sysTime_dirty = myFile[:, 0]
sysTime = sysTime_dirty - sysTime_dirty[0]
sysTime = np.round(sysTime, 2)

print(f'len Systime {len(sysTime)}')

currentSimPos = myFile[0, 0]
print(currentSimPos)

# print(len(sysTime) / 100)
# print(sysTime[0])

spdTAS = myFile[:, 15]
spdIAS = myFile[:, 14]

altitudeGPS = myFile[:, 20]
altitudeBaro = myFile[:, 16]
altitudeRA1 = myFile[:, 21]
altitudeRA2 = myFile[:, 22]

localizer = myFile[:, 3]
glideslope = myFile[:, 5]

stickFOPitch = myFile[:, 8]
stickFORoll = myFile[:, 9]

stickCPPitch = myFile[:, 6]
stickCPRoll = myFile[:, 7]

thrustLeverLH = myFile[:, 78]
thrustLeverRH = myFile[:, 79]

flapPos = myFile[:, 31]
speadbrake = myFile[:, 106]
gearCmd = myFile[:, 45]

# Map GPS Data
myLat = myFile[:, 112] * 180 / np.pi
myLong = myFile[:, 114] * 180 / np.pi


def setData(filenameSD):
    global filenameSim, sysTime, spdIAS, spdTAS, altitudeGPS, altitudeBaro, altitudeRA2, altitudeRA1
    if filenameSim == filenameSD:
        print(True)
    sysTime = getData(filenameSD, 0)
    spdTAS = getData(filenameSD, 14)
    spdIAS = getData(filenameSD, 13)

    altitudeGPS = getData(filenameSD, 18)
    altitudeBaro = getData(filenameSD, 16)
    altitudeRA1 = getData(filenameSD, 20)
    altitudeRA2 = getData(filenameSD, 21)


reduced_Latitude = myLat[0::100]
reduced_Longitude = myLong[0::100]
print(f'Lat SIze {len(reduced_Latitude)}')

global figMap
figMap = go.Figure()
figMap.add_trace(go.Scattermapbox(
    mode="markers",
    lat=reduced_Latitude,
    lon=reduced_Longitude,
    name="FlightPath",
    marker={'color': 'green', 'size': 4},
    showlegend=False))

tracker = go.Scattermapbox(
    mode="markers",
    lat=[myLat[0]],
    lon=[myLong[0]],
    name="Pos",
    marker={'size': 15},
    showlegend=False)

figMap.add_trace(tracker)

figMap.update_layout(
    mapbox_style="open-street-map",
    margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
    mapbox={
        'center': {'lon': myLong.mean(), 'lat': myLat.mean()},
        'style': "stamen-terrain",
        'center': {'lon': myLong.mean(), 'lat': myLat.mean()},
        'zoom': 6})

figMap.update_layout(transition_duration=5)
figMap.update_layout(uirevision='constant')

# Create Plots
figureFlightData = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.005)

# Altitude
figAltGPS = go.Scatter(x=sysTime, y=altitudeGPS, name="GPS Altitude")
figAltBaro = go.Scatter(x=sysTime, y=altitudeBaro, name="Baro Altitude")
figAltRA1 = go.Scatter(x=sysTime, y=altitudeRA1, name="RA1 Altitude")
figAltRA2 = go.Scatter(x=sysTime, y=altitudeRA2, name="RA2 Altitude")

# Speeds
figTAS = go.Scatter(x=sysTime, y=spdTAS, name="True Air Speed")
figIAS = go.Scatter(x=sysTime, y=spdIAS, name="Indicated Air Speed")

# Sidestick Position
figStickCPPitch = go.Scatter(x=sysTime, y=stickCPPitch, name="CP Ptich")
figStickCPRoll = go.Scatter(x=sysTime, y=stickCPRoll, name="CP Roll")

figStickFOPitch = go.Scatter(x=sysTime, y=stickFOPitch, name="FO Ptich")
figStickFORoll = go.Scatter(x=sysTime, y=stickFORoll, name="FO Roll")

# Thrust Lever Position
figStickThrLevRH = go.Scatter(x=sysTime, y=thrustLeverLH, name="Thrust Lever RH")
figStickThrLevLH = go.Scatter(x=sysTime, y=thrustLeverRH, name="Thrust Lever LH")

# Localizer
figLoc = go.Scatter(x=sysTime, y=localizer, name="Localizer Deviation")

# Glideslope
figGS = go.Scatter(x=sysTime, y=glideslope, name="Glideslope Deviation")

# Flap Lever
figFlapLever = go.Scatter(x=sysTime, y=flapPos, name="Flap Pos")
# Landing Gear Command
figGear = go.Scatter(x=sysTime, y=gearCmd, name="Gear Pos")
# Spead Brakes
figSpdBrake = go.Scatter(x=sysTime, y=speadbrake, name="Spead Brakes")

# Subplots

# figAltitude = make_subplots()
figureFlightData.add_trace(figAltBaro, row=1, col=1)
figureFlightData.add_trace(figAltGPS, row=1, col=1)
figureFlightData.add_trace(figAltRA1, row=1, col=1)
figureFlightData.add_trace(figAltRA2, row=1, col=1)

figureFlightData.add_trace(figTAS, row=2, col=1)
figureFlightData.add_trace(figIAS, row=2, col=1)

figureFlightData.add_trace(figStickCPPitch, row=3, col=1)
figureFlightData.add_trace(figStickCPRoll, row=3, col=1)

figureFlightData.add_trace(figStickFOPitch, row=3, col=1)
figureFlightData.add_trace(figStickFORoll, row=3, col=1)

figureFlightData.add_trace(figLoc, row=4, col=1)
figureFlightData.add_trace(figGS, row=4, col=1)

figureFlightData.add_trace(figStickThrLevRH, row=5, col=1)
figureFlightData.add_trace(figStickThrLevLH, row=5, col=1)

figureFlightData.add_trace(figFlapLever, row=6, col=1)
figureFlightData.add_trace(figGear, row=6, col=1)
figureFlightData.add_trace(figSpdBrake, row=6, col=1)

figureFlightData.add_vline(
    x=sysTime[0], line_width=1.5, line_dash="dash",
    line_color="green")

# edit axis labels
figureFlightData['layout']['yaxis']['title'] = 'Altitude'
figureFlightData['layout']['yaxis2']['title'] = 'Speed'
figureFlightData['layout']['yaxis3']['title'] = 'Stick'
figureFlightData['layout']['yaxis4']['title'] = 'LOC/GS'
figureFlightData['layout']['yaxis5']['title'] = 'Thrustlever'
figureFlightData['layout']['yaxis6']['title'] = 'Flaps/Gear'

figureFlightData.update_layout(
    legend=dict(
        orientation="h",
        # yanchor="bottom",
        # y=1.02,
        # xanchor="right",
        # x=1.05
    ),
    margin=dict(
        l=0.001,
        r=0.001,
        b=0.001,
        t=0.001,
        pad=0.001
    ))

l = [-20 * np.pi / 180, 0, 27 * np.pi / 180, 36 * np.pi / 180, 45 * np.pi / 180]
thr_txt = ['Max Rev', 'Idle', 'CLB', 'FLX/MCT', 'TO/GA']

figStickCPPositionCurrent = go.Scatter(mode="markers",
                                       x=[stickCPRoll[0]], y=[stickCPPitch[0]],
                                       marker_symbol="star-diamond",
                                       marker_size=40,
                                       name="CurrentStickCP")

figStickFOPositionCurrent = go.Scatter(mode="markers",
                                       x=[stickFORoll[0]], y=[stickFOPitch[0]],
                                       marker_symbol="star-diamond",
                                       marker_size=40,
                                       name="CurrentStickFO")

figThrustLHCurrent = go.Scatter(mode="markers",
                                x=[1.34], y=[thrustLeverLH[0]],
                                marker_symbol="arrow-left",
                                marker_size=40,
                                name="LHCurrentThrust")

figThrustRHCurrent = go.Scatter(mode="markers",
                                x=[1.66], y=[thrustLeverRH[0]],
                                marker_symbol="arrow-right",
                                marker_size=40,
                                name="RHCurrentThrust")

figStickCurrentCP = make_subplots()
figStickCurrentCP.add_trace(figStickCPPositionCurrent)

figStickCurrentCP.update_layout(
    yaxis_range=[-10, 10],
    xaxis_range=[-10, 10],
    showlegend=False,
    height=250,
    # width=250,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=4,
        pad=1
    ),
)

figStickCurrentFO = make_subplots()
figStickCurrentFO.add_trace(figStickFOPositionCurrent)

figStickCurrentFO.update_layout(
    yaxis_range=[-10, 10],
    xaxis_range=[-10, 10],
    showlegend=False,
    height=250,
    # width=250,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=4,
        pad=1
    ),
)

figThrustCurrent = make_subplots()
figThrustCurrent.add_trace(figThrustLHCurrent)
figThrustCurrent.add_trace(figThrustRHCurrent)

figThrustCurrent.update_layout(
    yaxis_range=[-0.5, 1],
    xaxis_range=[1.2, 1.8],
    showlegend=False,
    height=250,
    # width=250,
    yaxis=dict(
        tickmode='array',
        tickvals=l,
        ticktext=thr_txt
    ),

    margin=dict(
        l=1,
        r=1,
        b=1,
        t=4,
        pad=1
    ),
)


################################################
##      ECG Data Plots
################################################

def analyze_qrs(dataframe, xqrs):
    xqrs.detect()
    rr = np.diff(xqrs.qrs_inds)

    heart_rate = processing.compute_hr(sig_len=len(dataframe), qrs_inds=xqrs.qrs_inds, fs=300)

    heart_rate_r2 = 60 * 1000 / rr * (300 / 1000)

    # Mean RR
    mean_rr = np.mean(rr)
    print(f'> Mean RR: \t{mean_rr} ms')

    # RMSSD: take square root of mean square of differences
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    print(f'> RMSSD: \t{rmssd} ms')

    # SDNN
    sdnn = np.std(rr)
    print(f'> SDNN: \t{sdnn} ms')

    # Mean HR
    mean_hr2 = np.mean(heart_rate_r2)
    print(f'> Mean HR: \t{mean_hr2} beats/min')

    # MIN HR
    min_hr2 = np.min(heart_rate_r2)
    print(f'> Min HR: \t{min_hr2} beats/min')

    # Max HR
    max_hr2 = np.max(heart_rate_r2)
    print(f'> Max HR: \t{max_hr2} beats/min')

    # MNNxx: Sum absolute differences that are larger than 50ms
    nnxx = np.sum(np.abs(np.diff(rr)) > 50) * 1
    print(f'> NNXX: \t{nnxx}')

    # pNNx:  fraction of nxx of all rr-intervals
    pnnx = 100 * nnxx / len(rr)
    print(f'> pNNxx: \t{pnnx} %')

    qrs_inds_plot = np.divide(xqrs.qrs_inds, sampling_rate)

    return rr, heart_rate, heart_rate_r2, qrs_inds_plot


columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']

sampling_rate = 300

# Captain Side Data
df_cp = pd.read_csv(fileNameECG_CP, usecols=columns, skiprows=1, delimiter="\t")
xqrs_cp = processing.XQRS(sig=df_cp['II'], fs=sampling_rate)
rr_cp, hr_cp, hr2_cp, qrs_cp_plot = analyze_qrs(df_cp['II'], xqrs_cp)

# First Officer Side
df_fo = pd.read_csv(fileNameECG_FO, usecols=columns, skiprows=1, delimiter="\t")
xqrs_fo = processing.XQRS(sig=df_fo['II'], fs=sampling_rate)
rr_fo, hr_fo, hr2_fo, qrs_fo_plot = analyze_qrs(df_fo['II'], xqrs_fo)

# Ruhe EKG Daten CP
df_ruhe_cp = pd.read_csv(fileNameECGRuhe_CP, usecols=columns, skiprows=1, delimiter="\t")
xqrs_ruhe_cp = processing.XQRS(sig=df_ruhe_cp['II'], fs=sampling_rate)
rr_ruhe_cp, hr_ruhe_cp, hr2_ruhe_cp, qrs_ruhe_cp_plot = analyze_qrs(df_ruhe_cp['II'], xqrs_ruhe_cp)

# Ruhe EKG Daten FO
df_ruhe_fo = pd.read_csv(fileNameECGRuhe_FO, usecols=columns, skiprows=1, delimiter="\t")
xqrs_ruhe_fo = processing.XQRS(sig=df_ruhe_fo['II'], fs=sampling_rate)
rr_ruhe_fo, hr_ruhe_fo, hr2_ruhe_fo, qrs_ruhe_fo_plot = analyze_qrs(df_ruhe_fo['II'], xqrs_ruhe_fo)

hr_cp_rounded = np.round(hr2_cp, 0)
hr_fo_rounded = np.round(hr2_fo, 0)

hr_ruhe_cp_rounded = np.round(hr2_ruhe_cp, 0)
hr_ruhe_fo_rounded = np.round(hr2_ruhe_fo, 0)

hr_histo_fig = go.Figure()
hr_histo_fig.add_trace(go.Histogram(x=hr_cp_rounded, name="HR Dist CP"))
hr_histo_fig.add_trace(go.Histogram(x=hr_fo_rounded, name="HR Dist FO"))

hr_histo_fig.add_trace(go.Histogram(x=hr_ruhe_fo_rounded, name="HR Ruhe CP"))
hr_histo_fig.add_trace(go.Histogram(x=hr_ruhe_fo_rounded, name="HR Ruhe FO"))

hr_histo_fig.update_layout(barmode='overlay')
hr_histo_fig.update_traces(opacity=0.75)
hr_histo_fig.update_layout(
    legend=dict(
        orientation="h",
    ),
    margin=dict(
        l=0.001,
        r=0.001,
        b=0.001,
        t=0.01,
        pad=0.01
    ))


def createECGFig(channels):
    print(channels)

    channel_length = len(channels)

    y_axis_array = []
    for channel in channels:
        y_axis_array.append([{"secondary_y": False}])
    y_axis_array.append([{"secondary_y": True}])
    y_axis_array.append([{"secondary_y": False}])

    for channel in channels:
        y_axis_array.append([{"secondary_y": False}])
    y_axis_array.append([{"secondary_y": True}])
    y_axis_array.append([{"secondary_y": False}])

    figure = make_subplots(rows=channel_length * 2 + 4, cols=1, shared_xaxes=True, vertical_spacing=0.005,
                           specs=y_axis_array)

    ecg_time_cp = np.arange(0.0, len(df_cp['I']) / sampling_rate, 1 / sampling_rate)
    ecg_time_fo = np.arange(0.0, len(df_fo['I']) / sampling_rate, 1 / sampling_rate)

    hr1Plot_cp = go.Scatter(x=ecg_time_cp, y=hr_cp, name="Heart Rate WFDB")
    hr2Plot_cp = go.Scatter(x=qrs_cp_plot, y=np.insert(hr2_cp, 0, 0), mode='markers', name="Heart Rate")
    rrPlot_cp = go.Scatter(x=qrs_cp_plot, y=np.insert(rr_cp, 0, 0), mode='markers', name="RR-Distance")

    # phys_data_fo_figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.005, specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]])

    hr1Plot_fo = go.Scatter(x=ecg_time_fo, y=hr_fo, name="Heart Rate WFDB")
    hr2Plot_fo = go.Scatter(x=qrs_fo_plot, y=np.insert(hr2_fo, 0, 0), mode='markers', name="Heart Rate")
    rrPlot_fo = go.Scatter(x=qrs_fo_plot, y=np.insert(rr_fo, 0, 0), mode='markers', name="RR-Distance")

    row = 1

    for channel in channels:
        plot_channel_cp = go.Scatter(x=ecg_time_cp, y=df_cp[channel], name=f'Channel {channel}', fillcolor='blue')
        plot_channel_fo = go.Scatter(x=ecg_time_fo, y=df_fo[channel], name=f'Channel {channel}', fillcolor='blue')

        figure.add_trace(plot_channel_cp, row=row, col=1)
        figure.add_trace(plot_channel_fo, row=row + len(channels) + 2, col=1)
        if row == 1:
            figure['layout'][f'yaxis']['title'] = f'{channel} CP'
            figure['layout'][f'yaxis{row + channel_length + 3}']['title'] = f'{channel} FO'
        else:
            figure['layout'][f'yaxis{row}']['title'] = f'{channel} CP'
            figure['layout'][f'yaxis{row + channel_length + 3}']['title'] = f'{channel} FO'
        row = row + 1

    figure.add_trace(hr1Plot_cp, row=row, col=1, secondary_y=False)
    figure.add_trace(hr2Plot_cp, row=row, col=1, secondary_y=False)
    figure.add_trace(rrPlot_cp, row=row, col=1, secondary_y=True)

    figure.add_trace(hr1Plot_fo, row=row + channel_length + 2, col=1, secondary_y=False)
    figure.add_trace(hr2Plot_fo, row=row + channel_length + 2, col=1, secondary_y=False)
    figure.add_trace(rrPlot_fo, row=row + channel_length + 2, col=1, secondary_y=True)
    ################################################
    ##      Stress Tracker Plots
    ################################################
    # # Load Data from Captain
    dataCP = np.loadtxt(fileDataArrayCP, skiprows=0, delimiter=',')
    dataConfirmedCP = np.loadtxt(fileDataArrayConfirmedCP, skiprows=0, delimiter=',')

    # Load Data from First Officer
    dataFO = np.loadtxt(fileDataArrayFO, skiprows=0, delimiter=',')
    dataConfirmedFO = np.loadtxt(fileDataArrayConfirmedFO, skiprows=0, delimiter=',')

    stresslevel_cp = go.Scatter(x=dataCP[:, 0] - dataCP[0, 0], y=dataCP[:, 1], name="Slider Change CP")
    stresslevelConf_cp = go.Scatter(x=dataConfirmedCP[:, 0] - dataConfirmedCP[0, 0], y=dataConfirmedCP[:, 1],
                                    name="Confirmed Values CP")

    stresslevel_fo = go.Scatter(x=dataFO[:, 0] - dataFO[0, 0], y=dataFO[:, 1], name="Slider Change FO")
    stresslevelConf_fo = go.Scatter(x=dataConfirmedFO[:, 0] - dataConfirmedFO[0, 0], y=dataConfirmedFO[:, 1],
                                    name="Confirmed Values FO")

    figure.add_trace(stresslevel_cp, row=row + 1, col=1)
    figure.add_trace(stresslevelConf_cp, row=row + 1, col=1)

    figure.add_trace(stresslevel_fo, row=row + channel_length + 3, col=1)
    figure.add_trace(stresslevelConf_fo, row=row + channel_length + 3, col=1)

    figure['layout'][f'yaxis{row}']['title'] = f'HR CP'
    figure['layout'][f'yaxis{row + channel_length + 3}']['title'] = f'HR FO'

    if len(ecg_time_cp) > len(ecg_time_fo):
        x_axis_limit = ecg_time_cp[-1]
    elif len(ecg_time_fo) > len(ecg_time_cp):
        x_axis_limit = ecg_time_fo[-1]

    figure.update_layout(
        xaxis_range=[0, x_axis_limit],
        legend=dict(
            orientation="h",
        ),
        margin=dict(
            l=0.001,
            r=0.001,
            b=0.001,
            t=0.001,
            pad=0.001
        ))

    return figure


def create_card(card_id, title, value, description):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, id=f"{card_id}-title"),
                html.H6(children=value, id=f"{card_id}-value"),
                # html.P(description, id=f"{card_id}-description")
            ],
        )
    )



################################################
##      HTML Layout
################################################
app.layout = html.Div(children=[
    html.H1(children='AVES Flight Data'),

    dcc.Interval(id="animate", disabled=True, interval=1000),

    html.Div([

        html.Div(children=[

            html.H2(children='Flight Data'),
            html.Div(id='flightDataReadOut', children=[
                dbc.Row([
                    dbc.Col([create_card('card_baro_alt', 'Baro', f'{np.round(altitudeBaro[0], 1)} m',
                                         'Barometric Altitude')]),
                    dbc.Col([create_card('card_gps_alt', 'GPS', f'{np.round(altitudeGPS[0], 1)} m', 'GPS Altitude')]),
                    dbc.Col([create_card('card_ra1_alt', 'RA 1', f'{np.round(altitudeRA1[0], 1)} m', 'RA1 Altitude')]),
                    dbc.Col([create_card('card_ra2_alt', 'RA 2', f'{np.round(altitudeRA2[0], 1)} m', 'RA2 Altitude')])
                ], className="g-0"),
                dbc.Row([
                    dbc.Col([create_card('card_ias_spd', 'IAS', f'{np.round(spdIAS[0], 1)} kts', 'Indicated Air Speed')]),
                    dbc.Col([create_card('card_tas_spd', 'TAS', f'{np.round(spdTAS[0], 1)} kts', 'True Air Speed')])
                ], className="g-0"),
                dbc.Row([
                    dbc.Col([create_card('card_lheng_spd', 'ENG 1', f'{np.round(thrustLeverLH[0] * 100, 1)} %',
                                         'Thrust Lever ENG 1')]),
                    dbc.Col([create_card('card_rheng_spd', 'ENG 2', f'{np.round(thrustLeverRH[0] * 100, 1)} %',
                                         'Thrust Lever ENG 2')])
                ], className="g-0"),
                dbc.Row([
                    dbc.Col([create_card('card_loc_spd', 'LOC', f'{np.round(localizer[0], 1)} %', 'Localizer Deviation')]),
                    dbc.Col([create_card('card_gs_spd', 'GS', f'{np.round(glideslope[0], 1)} %', 'Glideslope Deviation')])
                ], className="g-0"),
                dbc.Row([
                    dbc.Col([create_card('card_flps_spd', 'Flaps', f'{np.round(flapPos[0], 1)}', 'Flaps Position')]),
                    dbc.Col([create_card('card_gear_spd', 'Gear', f'{np.round(gearCmd[0], 1)}', 'Gear Up/Down')]),
                    dbc.Col([create_card('card_spdbrks_spd', 'Spoiler', f'{np.round(speadbrake[0], 1)}', 'Speed Brakes')])
                ], className="g-0")
            ]),

            # html.Div(id='flightDataReadOut'),

            html.Br(),

            html.H2(children='Simulator Time Slider'),
            html.Div([
                dcc.Slider(
                    min=sysTime[0] + 1,
                    max=sysTime[-1],
                    # marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},
                    value=sysTime[0],
                    id='timeSlider',
                    marks={(i): {'label': str(np.round(sysTime[i] * 100)),
                                 'style': {'color': 'grey', 'writing-mode': 'vertical-rl',
                                           'text-orientation': 'use-glyph-orientation'}} for i in range(len(sysTime)) if
                           i % 100 == 0},
                ),

            ], style={'text-orientation': 'sideways', "margin-bottom": "20px"}),

            html.Div(id='container_text_sim_time',
                     children='Current Sim Time: [ms]'),
            html.Button('«', id='pl_Rev'),
            html.Button('▶', id='pl_Play'),
            html.Button('⏸', id='pl_Pause'),
            html.Button('»', id='pl_Back'),
            html.Div(id='container-button-timestamp'),

            html.Br(),

            html.H3('Physiology'),
            html.Label('ECG Channel Select'),
            dcc.Dropdown(columns, id='ecg_channel', value=['I', 'Resp'], multi=True),

            html.Br(),

            html.H3('File Upload'),
            html.Div(id='subtext'),
            dcc.Upload(html.Button('Upload File'), id='fileSelector'),

        ], style={'padding': 2, 'flex': 2, 'width': '6%'}),

        html.Div([
            html.Div(children=[
                dcc.Graph(
                    figure=figureFlightData,
                    id='graphFligthData',

                    style={'height': '90vh'}
                ),
            ], style={'padding': 2, 'flex': 1}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),

        html.Div([
            html.Div(children=[

                # dcc.Graph(
                #     figure=imageFig,
                #     id='CockpitImage',
                #     style={"margin-bottom": "1px"}
                # ),

                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='stickCPPosCurrent',
                            style={'width': '100%'}
                        )
                    ], style={'flex': 1}),

                    html.Div([
                        dcc.Graph(
                            id='thrustPosCurrent',
                            style={'width': '100%'}
                        )
                    ], style={'flex': 1}),

                    html.Div([
                        dcc.Graph(
                            id='stickFOPosCurrent',
                            style={'width': '100%'}
                        )
                    ], style={'flex': 1}),

                ], style={'display': 'flex', 'flex-direction': 'row'}),

                dcc.Graph(
                    figure=figMap,
                    id='MapPlot',
                    style={'height': '30vh'}
                ),

                dcc.Graph(
                    figure=hr_histo_fig,
                    id='hr_histo_fig',
                    style={'width': '100%'}
                ),

            ], style={'padding': 2, 'flex': 1, 'width': '25vw'}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),

        html.Div([
            html.Div(id='display-selected-ecg-channel', style={"margin-bottom": "1px", 'height': '90%'}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),

    ], style={'margin-bottom': '1px', 'display': 'flex', 'flex-direction': 'row'})
])


###############################


##############################################
# Callbacks
##############################################
@app.callback(
    Output(component_id='display-selected-ecg-channel', component_property='children'),
    Input('ecg_channel', 'value')
)
def update_ecg_plt(values):
    plots = []

    fig = createECGFig(values)
    graph = dcc.Graph(figure=fig, style={'height': '90vh'})
    plots.append(graph)

    children = html.Div(plots)
    return children


@app.callback(
    Output("timeSlider", "value"),
    Input("animate", "n_intervals"),
    State("timeSlider", "value"),
    # prevent_initial_call=True,
)
def update_animation(n, selected_time):
    # print(f'Selected: {selected_time}')
    new_time = np.round(selected_time + 2, 2)
    return new_time


@app.callback([
    Output('container_text_sim_time', 'children'),
    Output('stickCPPosCurrent', 'figure'),
    Output('stickFOPosCurrent', 'figure'),
    Output('thrustPosCurrent', 'figure'),
    Output('graphFligthData', 'figure'),
    Output('MapPlot', 'figure'),
    Output('card_gps_alt-value', 'children'),
    Output('card_baro_alt-value', 'children'),
    Output('card_ra1_alt-value', 'children'),
    Output('card_ra2_alt-value', 'children'),
    Output('card_ias_spd-value', 'children'),
    Output('card_tas_spd-value', 'children'),
    Output('card_lheng_spd-value', 'children'),
    Output('card_rheng_spd-value', 'children'),
    Output('card_loc_spd-value', 'children'),
    Output('card_gs_spd-value', 'children'),
    Output('card_flps_spd-value', 'children'),
    Output('card_gear_spd-value', 'children'),
    Output('card_spdbrks_spd-value', 'children'),
],
    Input("timeSlider", "value"),
)
def update_figure(timeSlider):
    location = np.where(sysTime == timeSlider)

    # print(f'Update SimTime: {location[0][0]}')
    # print(f"Current Sim Time: {timeSlider}")

    textSilder = f"Current Sim Time: {timeSlider}"

    figMap.update_traces(selector=dict(name="Pos"), lat=[myLat[location[0]][0]])
    figMap.update_traces(selector=dict(name="Pos"), lon=[myLong[location[0]][0]])

    # Stick and Thrust Lever Position
    figureFlightData.update_shapes(selector=dict(type="line"), x0=sysTime[location][0])
    figureFlightData.update_shapes(selector=dict(type="line"), x1=sysTime[location][0])

    figThrustCurrent.update_traces(selector=dict(name="RHCurrentThrust"), y=thrustLeverRH[location])
    figThrustCurrent.update_traces(selector=dict(name="LHCurrentThrust"), y=thrustLeverLH[location])

    figStickCurrentFO.update_traces(selector=dict(name="CurrentStickFO"), x=stickFORoll[location])
    figStickCurrentFO.update_traces(selector=dict(name="CurrentStickFO"), y=stickFOPitch[location])

    figStickCurrentCP.update_traces(selector=dict(name="CurrentStickCP"), x=stickCPRoll[location])
    figStickCurrentCP.update_traces(selector=dict(name="CurrentStickCP"), y=stickCPPitch[location])

    card_baro_alt = f'{np.round(altitudeBaro[location][0], 1)} m'
    card_gps_alt = f'{np.round(altitudeGPS[location][0], 1)} m'
    card_ra1_alt = f'{np.round(altitudeRA1[location][0], 1)} m'
    card_ra2_alt = f'{np.round(altitudeRA2[location][0], 1)} m'
    card_ias_spd = f'{np.round(spdIAS[location][0], 1)} kts'
    card_tas_spd = f'{np.round(spdTAS[location][0], 1)} kts'
    card_lheng_spd = f'{np.round(thrustLeverLH[location][0] * 100, 1)} %'
    card_rheng_spd = f'{np.round(thrustLeverRH[location][0] * 100, 1)} %'
    card_loc_spd = f'{np.round(localizer[location][0], 1)} %'
    card_gs_spd = f'{np.round(glideslope[location][0], 1)} %'
    card_flps_spd = f'{np.round(flapPos[location][0], 1)}'
    card_gear_spd = f'{np.round(gearCmd[location][0], 1)}'
    card_spdbrks_spd = f'{np.round(speadbrake[location][0], 1)}'

    return textSilder, figStickCurrentCP, figStickCurrentFO, figThrustCurrent, figureFlightData, figMap, \
           card_gps_alt, card_baro_alt, card_ra1_alt, card_ra2_alt, card_ias_spd, card_tas_spd, \
           card_lheng_spd, card_rheng_spd, card_loc_spd, card_gs_spd, card_flps_spd, card_gear_spd, \
           card_spdbrks_spd


@app.callback(
    Output('subtext', 'children'),
    Input('fileSelector', 'filename'))
def update_filename(fileSelector):
    global filenameSim
    print(fileSelector)
    if fileSelector == None:
        children = f'Current Flight: {filenameSim}'
    else:
        children = f'Current Flight: {fileSelector}'
        filenameSim = fileSelector
        setData(fileSelector)
    return children


@app.callback(
    [Output('container-button-timestamp', 'children'),
     Output("animate", "disabled")],
    Input('pl_Play', 'n_clicks'),
    State("animate", "disabled"),
    Input('pl_Pause', 'n_clicks'),
    Input('pl_Rev', 'n_clicks'),
    Input('pl_Back', 'n_clicks'),
)
def displayClick(btn1, playing, btn2, btn3, btn4):
    msg = "None of the buttons have been clicked yet"
    global currentSimPos

    if "pl_Play" == ctx.triggered_id:
        msg = "Button Play was most recently clicked"

    elif "pl_Pause" == ctx.triggered_id:
        msg = "Button Pause was most recently clicked"

    elif "pl_Rev" == ctx.triggered_id:
        msg = "Button Reverse was most recently clicked"

    elif "pl_Back" == ctx.triggered_id:
        msg = "Button Backend was most recently clicked"

    if btn1:
        print(playing)
        return html.Div(msg), not playing
    return html.Div(msg), playing


################################################
##      Main Call
################################################
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
