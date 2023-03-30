import csv

from dash import Dash, html, dcc, html, Input, Output, ctx, State
import dash_daq as daq

import numpy as np
import time
from datetime import datetime

from PIL import Image
import base64


app = Dash(__name__)

t_update = 60  # Time After Update is requested In Seconds
t_increase = 2 * 60  # Time after Stress level is increased in Seconds
t_step = 4  # Time Step for each increase In Seconds
sl_incr = 0.1  # Stress Level increment

global data_array
data_array = np.array([[time.time(), 0]])
global data_array_confirmed
data_array_confirmed = np.array([[time.time(), 0, 0]])

global simRun
simRun = False


global dataFile
dataFile = ""
global dataFileConfirmed
dataFileConfirmed = ""

ind_col_amber = "#f58b27"
ind_col_green = "#33c41f"
ind_col_red = "#f22020"


def countdown():
    time.sleep(10)
    return False


logo_path = 'DLRFLLogo.png'
pil_img = Image.open(logo_path)


# Using base64 encoding and decoding
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


###############################
# HTML Layout
###############################
app.layout = html.Div(children=[
    html.Div([
        html.Div(children=[
            html.H1(children='Stress Level Feedback'),

            html.Img(src=b64_image(logo_path)),  # using base64 to encode and decode the image file

        ], style={'display': 'flex', 'flex-direction': 'row', "margin-right": "5px",
                  'justify-content': 'space-between'}),
    ], style={'padding': 2, 'flex': 1}),

    html.Div([
        daq.Slider(
            id='slider_stress',
            marks={str(h): {'label': str(h), 'style': {'fontSize': '30px'}} for h in range(11)},
            max=10,
            value=0,
            step=0.1,
            updatemode='drag',
            vertical=False,
            handleLabel={
                "showCurrentValue": True,
                "label": "VALUE",
                'style': {'fontSize': '200px'}
            },
            size=900,
        )
    ], style={'padding': 2, 'flex': 1, "margin-left": "20px", "margin-bottom": "80px"}),

    html.Div([

        html.Div(children=[
            html.Div([
                html.Button('Submit', id='submit-val', n_clicks=0,
                            style={'font-size': '24px',
                                   'width': '240px',
                                   'height': '60px',
                                   'display': 'inline-block',
                                   'margin-bottom': '10px',
                                   'margin-left': '50px',
                                   'verticalAlign': 'top'}),

                daq.Indicator(
                    id='updateIndicator',
                    label="Update",
                    color=ind_col_green,
                    value=True,
                    size=50,
                ),
                daq.Indicator(
                    id='simIndicator',
                    label="Sim Rec",
                    color=ind_col_amber,
                    value=True,
                    size=50,
                    style={"margin-left": "50px"},
                ),
                dcc.Interval(
                    id='interval-UDP',
                    interval=500,  # in milliseconds
                ),

                dcc.Interval(
                    id='interval-component',
                    interval=1 * 1000,  # in milliseconds
                    n_intervals=0,
                ),

            ], style={'display': 'flex', 'flex-direction': 'row'}),
            html.Div(children=[
                html.H1(children='Selected Stress Level: '),

                html.H1(id='container_text_stress', children='None', style={'margin-left': '15px'}),
            ], style={'display': 'flex', 'flex-direction': 'row'}),

            html.Button('Start', id='simStartStopButton', n_clicks=0,
                        style={'font-size': '24px',
                               'width': '240px',
                               'height': '60px',
                               'display': 'inline-block',
                               'margin-bottom': '10px',
                               'margin-left': '15px',
                               'verticalAlign': 'top'}),
            html.Div(id='textContainerClicks', children='Enter a value and press submit'),

        ], style={'padding': 2, 'flex': 1}),

    ], style={'display': 'flex', 'flex-direction': 'row'}),
])


##############################################
# Callbacks
##############################################
@app.callback(
    Output('container_text_stress', 'children'),
    Input('slider_stress', 'value'), prevent_initial_call=True
)
def update(sliderStress):
    text = f'{sliderStress}'

    global data_array
    values = [time.time(), sliderStress]
    data_array = np.append(data_array, [values], axis=0)

    if dataFile != "":
        with open(rf'{dataFile}', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(values)

    # print(data_array.shape)

    return text


@app.callback(
    Output('textContainerClicks', 'children'),
    Output('interval-component', 'n_intervals'),
    Input('submit-val', 'n_clicks'),
    State('slider_stress', 'value'), prevent_initial_call=True
)
def buttonSubmit(n_clicks, value):
    global data_array_confirmed

    values = [time.time(), value, n_clicks]
    data_array_confirmed = np.append(data_array_confirmed, [values], axis=0)
    if dataFileConfirmed != "":
        with open(rf'{dataFileConfirmed}', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(values)

    # print(data_array_confirmed)

    text = f'# Submits: {n_clicks} || Last Submit: {datetime.utcnow()}'
    return text, 0


@app.callback(
    Output('simStartStopButton', 'children'),
    Output('interval-component', 'disabled'),
    Output('simIndicator', 'color'),
    Output('submit-val', 'n_clicks'),
    Input('simStartStopButton', 'n_clicks'),
    State('interval-component', 'disabled'))
def startStopButton(n_clicks, disabled_state):
    print(n_clicks)

    global simRun
    global data_array
    global data_array_confirmed
    global dataFile
    global dataFileConfirmed

    if n_clicks % 2 == 0:
        # Stop Recording
        text = "Start"
        simRun = False
        disabled_state = not simRun
        indicatorColor = ind_col_amber

        if n_clicks != 0:
            print("none")
            # Save Data to csv File
            timeNow = datetime.utcnow()

            np.savetxt(f"F_{dataFile}", data_array,
                       delimiter=",")

            np.savetxt(f"F_{dataFileConfirmed}",
                       data_array_confirmed, delimiter=",")
    else:
        # Start Recording

        # Reset Arrays
        data_array = np.array([[time.time(), 0]])
        data_array_confirmed = np.array([[time.time(), 0, 0]])

        # Save Data to csv File
        timeNow = datetime.utcnow()

        #dataFile = f"DA_FO_{timeNow.month}_{timeNow.day}_{timeNow.hour}_{timeNow.minute}.csv"
        dataFile = f"{timeNow.month}_{timeNow.day}_{timeNow.hour}_{timeNow.minute}_FO_DA.csv"

        dataFileConfirmed = f"{timeNow.month}_{timeNow.day}_{timeNow.hour}_{timeNow.minute}_FO_DConf.csv"

        np.savetxt(dataFile, data_array, delimiter=",")

        np.savetxt(dataFileConfirmed, data_array_confirmed, delimiter=",")

        text = "Stop"
        simRun = True
        disabled_state = not simRun
        indicatorColor = ind_col_green

    n_clicks = 0

    print(f'State: {disabled_state}')
    return text, disabled_state, indicatorColor, n_clicks


@app.callback(
    Output('updateIndicator', 'color'),
    Output('updateIndicator', 'size'),
    Output('updateIndicator', 'value'),
    Output('slider_stress', 'value'),
    Input('interval-component', 'n_intervals'),
    Input('slider_stress', 'value'),
)
def intervalFire(n, value):
    print(n)
    if n >= t_update:
        colorOut = ind_col_red
        sizeOut = 200
        if n % 2 == 0:
            bT = False
        else:
            bT = True

        if n >= t_increase and value < 10:
            if n % t_step == 0:
                print(f'Increase Stresslevel {value} +0.1')
                value = round(value + sl_incr, 1)

    else:
        colorOut = ind_col_green
        sizeOut = 50
        bT = True

    return colorOut, sizeOut, bT, value


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)