import datetime
import os
import numpy as np
import pandas as pd
import sys

import matplotlib
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, RectangleSelector, RangeSlider, Cursor, SpanSelector
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from wfdb import processing

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo, askyesnocancel, askyesno, showerror
from tkinter.filedialog import asksaveasfile

import pyxdf
from scipy import interpolate
from scipy.fftpack import fft

# Styles
LARGE_FONT = ("Verdana", 20)
NORMAL_FONT = ("Verdana", 10)
#style.use("seaborn-deep")
axcolor = 'lightgoldenrodyellow'

# Global Variables
currentOpenFolder = os.getcwd()
currentOpenStressFolder = os.getcwd()

matplotlib.use('TkAgg')

columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']

################################################
##      ECG Data Analytics
################################################
def analyze_qrs(dataframe, xqrs, fs):
    xqrs.detect()

    # XQRS inds in samples -> to seconds divide by sampling rate
    # From seconds to ms multipy by 1000
    rr = np.diff(xqrs.qrs_inds)*1000 / fs

    # From ms to min divide rr by 60000
    heart_rate_r2 = 60 * 1000 / rr

    heart_rate = processing.compute_hr(sig_len=len(dataframe), qrs_inds=xqrs.qrs_inds, fs=fs)

    print(f'> Mean HR 1: \t{np.mean(heart_rate)} beats/min')

    # Mean RR
    mean_rr = np.mean(rr)
    print(f'> Mean RR: \t{mean_rr} ms')

    # MIN RR
    min_rr2 = np.min(rr)
    print(f'> Min RR: \t{min_rr2} ')

    # Max RR
    max_rr2 = np.max(rr)
    print(f'> Max RR: \t{max_rr2}')

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

    # NNxx: Sum absolute differences that are larger than 50ms
    nnxx = np.sum(np.abs(np.diff(rr)) > 50) * 1
    print(f'> NNXX: \t{nnxx}')

    # pNNx:  fraction of nxx of all rr-intervals
    pnnx = 100 * nnxx / len(rr)
    print(f'> pNNxx: \t{pnnx} %')

    qrs_inds_plot = np.divide(xqrs.qrs_inds, fs)

    return rr, heart_rate, heart_rate_r2, qrs_inds_plot, mean_rr, sdnn, mean_hr2, min_hr2, max_hr2, nnxx, pnnx, rmssd

def popupmsg(msg):
    popup = tk.Tk()

    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORMAL_FONT)
    label.pack(side="top", fill="x", padx=60, pady=50)
    B1 = ttk.Button(popup, text="Okay", command=lambda: popup.destroy())

    B1.pack()
    popup.mainloop()

class ECGReadOutFrame(tk.LabelFrame):

    def __init__(self, master, text):
        tk.LabelFrame.__init__(self, master)

        self.config(text=text)

        # Min HR
        self.hr_min_label_label = tk.Label(self, text="HR Min:", font=NORMAL_FONT)
        self.hr_min_label = tk.StringVar()
        self.hr_min_label.set("XX /min")
        self.hr_min_label1 = tk.Label(self, textvariable=self.hr_min_label, font=NORMAL_FONT)

        self.hr_min_label_label.grid(row=0, column=0, pady=10)
        self.hr_min_label1.grid(row=0, column=1, pady=10)

        # Max HR
        self.hr_max_label_label = tk.Label(self, text="HR Max:", font=NORMAL_FONT)
        self.hr_max_label = tk.StringVar()
        self.hr_max_label.set("XX /min")
        self.hr_max_label1 = tk.Label(self, textvariable=self.hr_max_label, font=NORMAL_FONT)

        self.hr_max_label_label.grid(row=2, column=0, pady=10)
        self.hr_max_label1.grid(row=2, column=1, pady=10)

        # Mean HR
        self.hr_mean_label_label = tk.Label(self, text="HR Mean:", font=NORMAL_FONT)
        self.hr_mean_label = tk.StringVar()
        self.hr_mean_label.set("XX /min")
        self.hr_mean_label1 = tk.Label(self, textvariable=self.hr_mean_label, font=NORMAL_FONT)

        self.hr_mean_label_label.grid(row=4, column=0, pady=10)
        self.hr_mean_label1.grid(row=4, column=1, pady=10)

        # Mean RR
        self.rr_mean_label_label = tk.Label(self, text="RR Mean:", font=NORMAL_FONT)
        self.rr_mean_label = tk.StringVar()
        self.rr_mean_label.set("XX /min")
        self.rr_mean_label1 = tk.Label(self, textvariable=self.rr_mean_label, font=NORMAL_FONT)

        self.rr_mean_label_label.grid(row=6, column=0, pady=10)
        self.rr_mean_label1.grid(row=6, column=1, pady=10)

        # SDNN
        self.sdnn_label_label = tk.Label(self, text="SDNN:", font=NORMAL_FONT)
        self.sdnn_label = tk.StringVar()
        self.sdnn_label.set("XX")
        self.sdnn_label1 = tk.Label(self, textvariable=self.sdnn_label, font=NORMAL_FONT)

        self.sdnn_label_label.grid(row=8, column=0, pady=10)
        self.sdnn_label1.grid(row=8, column=1, pady=10)

        # RMSSD
        self.rmssd_label_label = tk.Label(self, text="RMSSD:", font=NORMAL_FONT)
        self.rmssd_label = tk.StringVar()
        self.rmssd_label.set("XX")
        self.rmssd_label1 = tk.Label(self, textvariable=self.rmssd_label, font=NORMAL_FONT)

        self.rmssd_label_label.grid(row=9, column=0, pady=10)
        self.rmssd_label1.grid(row=9, column=1, pady=10)

        # NNXX
        self.nnxx_label_label = tk.Label(self, text="NNxx:", font=NORMAL_FONT)
        self.nnxx_label = tk.StringVar()
        self.nnxx_label.set("XX")
        self.nnxx_label1 = tk.Label(self, textvariable=self.nnxx_label, font=NORMAL_FONT)

        self.nnxx_label_label.grid(row=10, column=0, pady=10)
        self.nnxx_label1.grid(row=10, column=1, pady=10)

        # pNNXX
        self.pnnxx_label_label = tk.Label(self, text="pNNxx:", font=NORMAL_FONT)
        self.pnnxx_label = tk.StringVar()
        self.pnnxx_label.set("XX %")
        self.pnnxxlabel1 = tk.Label(self, textvariable=self.pnnxx_label, font=NORMAL_FONT)

        self.pnnxx_label_label.grid(row=11, column=0, pady=10)
        self.pnnxxlabel1.grid(row=11, column=1, pady=10)


class MainApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        ################
        # Window Setup
        ################
        self.title("SimplECG Analyzer")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        ################################
        # Menu bar Items
        ################################
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save Settings", command=lambda: popupmsg("now command yet"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        tk.Tk.config(self, menu=menubar)

        ################
        # Frames
        ################
        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):
            frame = F(container, self)

            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky=N+E+S+W)

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="SimpleECG Viewer & Analyzer", font=LARGE_FONT)
        label.pack(pady=100, padx=100, side='top', fill='both')

        button1 = ttk.Button(self, text="Test1",
                             command=lambda: controller.show_frame(PageOne))
        button1.pack()

        button2 = ttk.Button(self, text="Test2",
                             command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="SimplECG",
                             command=lambda: controller.show_frame(PageThree))
        button3.pack()


class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.start_time = None
        self.stop_time = None

        ##########
        # Create Center Frames

        ##########
        top_frame = Frame(self)  # , bg='yellow') # yellow bg for debugging
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=0)

        left_frame = Frame(self)  # , bg='grey')
        left_frame.pack(side=tk.LEFT, fill=BOTH)

        right_frame = Frame(self)  # , bg='blue')
        right_frame.pack(side=tk.RIGHT, fill=BOTH)

        ##########
        # Plot
        #

        # Create Figure and Axes
        self.figP1, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(nrows=4, sharex=True)

        self.rraxis = self.ax3.twinx()

        self.ax4.set_ylim([-0.5, 10])

        self.slider_ax = self.figP1.add_axes([0.08, 0.04, 0.80, 0.03])

        self.selectslider_ax = self.figP1.add_axes([0.08, 0.01, 0.80, 0.03])

        # self.aniP1 = animation.FuncAnimation(self.figP1, self.showECG, interval=1000000, blit=False)

        # remove vertical gap between subplots
        self.figP1.subplots_adjust(top=0.995,
                                   bottom=0.1,
                                   left=0.04,
                                   right=0.96,
                                   hspace=0.0,
                                   wspace=0.105)

        # Packing
        canvasP1 = FigureCanvasTkAgg(self.figP1, self)
        canvasP1.draw()
        canvasP1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbarP1 = NavigationToolbar2Tk(canvasP1, self)
        toolbarP1.update()
        canvasP1._tkcanvas.pack(fill=tk.BOTH, expand=True)

        ##########
        # Labels
        #

        # Headline
        label = tk.Label(top_frame, text="ECG & Stress Track Matching", font=LARGE_FONT)
        label.pack(pady=5, padx=5)

        # Current File Label
        # Needs to be known by Class as it's modified later
        self.label2text = tk.StringVar()
        self.label2text.set("Current ECG File:" + currentOpenFolder)
        self.label2 = tk.Label(top_frame, textvariable=self.label2text, font=NORMAL_FONT)
        self.label2.pack(pady=5, padx=5)

        self.label3text = tk.StringVar()
        self.label3text.set("Current Stress File:" + currentOpenStressFolder)
        self.label3 = tk.Label(top_frame, textvariable=self.label3text, font=NORMAL_FONT)
        self.label3.pack(pady=5, padx=5)

        ##########
        # Navigation Frames
        #
        navFrame = LabelFrame(left_frame, text='Navigation')
        navFrame.pack(fill=X, expand=True)

        viewFrame = LabelFrame(left_frame, text='View Control')
        viewFrame.pack(fill=X, expand=True)

        selectionFrame = LabelFrame(left_frame, text='Channel Selection')
        selectionFrame.pack(fill=X, expand=True)

        analyzeFrame = LabelFrame(left_frame, text='Analyze')
        analyzeFrame.pack(fill=X, expand=True)

        exportFrame = LabelFrame(left_frame, text='Export')
        exportFrame.pack(fill=X, expand=True)

        ##########
        # navFrame
        #

        # Button to Get Back to Main Page
        button_Home = tk.Button(navFrame, text="Back Home",
                                 command=lambda: controller.show_frame(StartPage))

        # Button to open Files
        load_folder_button = tk.Button(navFrame, text='Load Data Folder',
                                     command=lambda: self.loadFolder())

        # Button to open MRF
        load_mrf_button = tk.Button(navFrame, text='Load Multi Record File',
                                     command=lambda: self.loadMRF())

        # Button to open Files
        self.load_ecg_button = tk.Button(navFrame, text='Load ECG File',
                                     command=lambda: self.select_ecg_file())

        # Button to show current Folder
        current_button = tk.Button(navFrame, text='Current ECG File',
                                    command=lambda: self.printName())

        # Button to open Files
        self.load_stress_button = tk.Button(navFrame, text='Load Stress Files',
                                        command=lambda: self.select_stress_file())

        # Button to reset Plots
        reset_button = tk.Button(navFrame, text='Reset All',
                                        command=lambda: self.reset_axis())

        ##########
        # selectionFrame
        #
        global columns
        ax1Label = tk.Label(selectionFrame, text="Plot 1:", font=NORMAL_FONT)
        self.ax1Combo = ttk.Combobox(selectionFrame, values=columns)
        self.ax1Combo.set('II')

        ax2Label = tk.Label(selectionFrame, text="Plot 2:", font=NORMAL_FONT)
        self.ax2Combo = ttk.Combobox(selectionFrame, values=columns)
        self.ax2Combo.set('Resp')

        self.ax1Combo.bind("<<ComboboxSelected>>", self.showECG)
        self.ax2Combo.bind("<<ComboboxSelected>>", self.showECG)

        ##########
        # analyzeFrame
        start_label = tk.Label(analyzeFrame, text="Start at:", font=NORMAL_FONT)
        self.start_entry = tk.Entry(analyzeFrame)

        stop_label = tk.Label(analyzeFrame, text="Stop at:", font=NORMAL_FONT)
        self.stop_entry = tk.Entry(analyzeFrame)

        # Button to accpet inout Area
        setselection_button = tk.Button(analyzeFrame, text='Set Selection',
                                         command=lambda: self.setSelection())

        # Button to analyze current Area
        analyze_button = tk.Button(analyzeFrame, text='Analyze Selection',
                                   command=lambda: self.analyzeSelection())

        # Button to Interpolate SLT
        interpol_button = tk.Button(analyzeFrame, text='Create SLT Interpolation',
                                   command=lambda: self.createInterpolation())

        interpLabel = tk.Label(analyzeFrame, text="Interpolation Type:", font=NORMAL_FONT)
        interpolate_types = ['Interp 1D', 'Akima 1D', 'Cubic Spline', 'PCHIP 1D']
        self.interpCombo = ttk.Combobox(analyzeFrame, values=interpolate_types)
        self.interpCombo.set('Akima 1D')

        self.interpCombo.bind("<<ComboboxSelected>>", self.createInterpolation())
        self.plotMeanBool = False
        self.meanCheckButton = tk.Checkbutton(analyzeFrame, text='Da/Conf Mean', variable=self.plotMeanBool, command=lambda: self.changeCombobBoxMean())

        ##########
        # exportFrame
        #

        # Button to Export Selected Data Area
        export_ecg_button = tk.Button(exportFrame, text='Export ECG Selection',
                                   command=lambda: self.exportSelection())

        export_mrf_button = tk.Button(exportFrame, text='Export to MRF',
                                       command=lambda : self.exportMRF())

        ##########
        # viewFrame
        #

        min_label = tk.Label(viewFrame, text="min:", font=NORMAL_FONT)
        self.min_entry = tk.Entry(viewFrame)
        max_label = tk.Label(viewFrame, text="max:", font=NORMAL_FONT)
        self.max_entry = tk.Entry(viewFrame)

        # Button to set Limit Entries
        setEntry_button = tk.Button(viewFrame, text='Set new Limits',
                                     command=lambda: self.setEntry())

        # Button to set Limit Entries
        resetZoom_button = tk.Button(viewFrame, text='Reset Zoom',
                                      command=lambda: self.zoomOut())

        # Button to Shift 100 ticks Left
        shiftL_button = tk.Button(viewFrame, text='<- 1 Seconds',
                                   command=lambda: self.shift_Left())

        # Button to Shift 100 ticks Rigth
        shiftR_button = tk.Button(viewFrame, text='1 Second ->',
                                   command=lambda: self.shift_Right())

        # Button to Shift 1000 Left
        shiftLK_button = tk.Button(viewFrame, text='<- 10 Seconds',
                                    command=lambda: self.shift_LeftK())

        # Button to Shift 1000 Rigth
        shiftRK_button = tk.Button(viewFrame, text='10 Seconds ->',
                                    command=lambda: self.shift_RightK())
        ##########
        # Grid all up
        #
        button_Home.grid(row=0, column=0, pady=10, columnspan=2)  # .pack(side=tk.TOP, expand=True)
        load_folder_button.grid(row=1, column=0, columnspan=2)
        load_mrf_button.grid(row=2, column=0, columnspan=2)
        self.load_ecg_button.grid(row=3, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        current_button.grid(row=3, column=1, padx=10, pady=10)  # .pack(side=tk.TOP, expand=True)
        self.load_stress_button.grid(row=4, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        reset_button.grid(row=5, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)

        min_label.grid(row=0, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        self.min_entry.grid(row=0, column=1, pady=10)  # .pack(side=tk.TOP, expand=True)
        max_label.grid(row=2, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        self.max_entry.grid(row=2, column=1, pady=10)  # .pack(side=tk.TOP, expand=True)
        shiftL_button.grid(row=5, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        shiftR_button.grid(row=5, column=1, pady=10)  # .pack(side=tk.TOP, expand=True)
        shiftLK_button.grid(row=6, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        shiftRK_button.grid(row=6, column=1, pady=10)  # .pack(side=tk.TOP, expand=True)
        setEntry_button.grid(row=4, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        resetZoom_button.grid(row=4, column=1, pady=10)  # .pack(side=tk.TOP, expand=True)

        ax1Label.grid(row=0, column=0, pady=10)
        self.ax1Combo.grid(row=0, column=1, pady=10)
        ax2Label.grid(row=2, column=0, pady=10)
        self.ax2Combo.grid(row=2, column=1, pady=10)

        start_label.grid(row=1, column=0, pady=10)
        self.start_entry.grid(row=1, column=1, pady=10)
        stop_label.grid(row=2, column=0, pady=10)
        self.stop_entry.grid(row=2, column=1, pady=10)
        setselection_button.grid(row=4, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        analyze_button.grid(row=6, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        interpol_button.grid(row=7, column=0, columnspan=2, pady=10)  # .pack(side=tk.TOP, expand=True)
        interpLabel.grid(row=8, column=0, pady=10)
        self.interpCombo.grid(row=8, column=1, pady=10)
        self.meanCheckButton.grid(row=9, column=0, pady=10)

        export_ecg_button.grid(row=0, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        export_mrf_button.grid(row=1, column=0, pady=10)  # .pack(side=tk.TOP, expand=True)
        ##########
        # Analysis Read Out Frame
        #

        # rawDataFrame = LabelFrame(right_frame, text='Original Dataset')
        self.rawDataFrame = ECGReadOutFrame(right_frame, text='Original Dataset')
        self.rawDataFrame.pack(fill=X, expand=True)

        self.selectedDataFrame = ECGReadOutFrame(right_frame, text='Selected Dataset')
        self.selectedDataFrame.pack(fill=X, expand=True)

    # Function to be rendered anytime a slider's value changes
    def setEntry(self):
        minVal = self.min_entry.get()
        maxVal = self.max_entry.get()
        if minVal:
            if maxVal:
                if maxVal > minVal:
                    print(minVal, maxVal)
                    self.span_slider.set_val([float(minVal), float(maxVal)])
                    self.figP1.canvas.draw()
                else:
                    reply = askyesnocancel(title="Wrong Order", message="Min Value higher than max Value. Want switch?")
                    if reply:
                        print(f"Min Value: {maxVal} Max Value: {minVal}")
                        self.min_entry.delete(0, END)
                        self.max_entry.delete(0, END)
                        self.min_entry.insert(0, maxVal)
                        self.max_entry.insert(0, minVal)
                        self.span_slider.set_val([float(maxVal), float(minVal)])
            else:
                showerror(title="No Value", message="Please enter a Max Value")
        else:
            showerror(title="No Value", message="Please enter a Min Value")

    def zoomOut(self):
        # Fixed Values
        global currentOpenFolder
        sampling_rate = 300

        # Length of Signal
        ecg_runtime = len(self.df_ecg['II'])

        stop_time = os.path.getmtime(currentOpenFolder)
        start_time2 = datetime.datetime.fromtimestamp(stop_time) - datetime.timedelta(seconds=ecg_runtime / sampling_rate)

        self.span_slider.set_val([start_time2.timestamp(), stop_time])

    def setSelection(self):
        startVal = self.start_entry.get()
        stoppVal = self.stop_entry.get()

        self.select_slider.set_val([float(startVal), float(stoppVal)])

        self.figP1.canvas.draw()

    def updateSpan(self, val):
        # print(f'Span: {self.span_slider.val}')

        self.ax1.set_xlim(self.span_slider.val[0], self.span_slider.val[1])
        self.min_entry.delete(0, END)
        self.max_entry.delete(0, END)
        self.min_entry.insert(0, self.span_slider.val[0])
        self.max_entry.insert(0, self.span_slider.val[1])
        self.figP1.canvas.draw()

    def updateSelectSlider(self, val):
        print(f'Selected: {val}')

        self.start_entry.delete(0, END)
        self.start_entry.insert(0, val[0])

        self.stop_entry.delete(0, END)
        self.stop_entry.insert(0, val[1])

        self.left_selection.set_width(val[0]-self.start_time)

        self.right_selection.set_x(val[1])
        self.right_selection.set_width(self.stop_time - val[1])

        self.figP1.canvas.draw()

    def printName(self):
        print(currentOpenFolder)

    def reset_axis(self):
        global currentOpenFolder
        global currentOpenStressFolder
        currentOpenFolder = None
        currentOpenStressFolder = None

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.rraxis.clear()
        self.slider_ax.clear()
        self.selectslider_ax.clear()
        self.ax4.clear()

        self.figP1.canvas.draw()

    def changeCombobBoxMean(self):
        self.plotMeanBool = not self.plotMeanBool
        print(f'Plot Mean? {self.plotMeanBool}')

        if self.plotMeanBool:

            if not hasattr(self, 'df_slt_slider'):
                print('No Data yet - return')
                return

            slt_sum = self.df_slt_confirmed['Conf'] + self.df_slt_slider['DA']

            slt_mean = slt_sum/2

            self.interp_mean = self.ax4.plot(self.df_slt_slider['UTC'], slt_mean, color='green')

            self.figP1.canvas.draw()

        else:
            print('remove mean line')
            l1 = self.interp_mean.pop(0)
            l1.remove()

            self.figP1.canvas.draw()

    def createInterpolation(self):
        print("I'm here to create a interpolation")

        # Check if Button is pressed on Empty Data
        if hasattr(self, 'intp_da'):
            print('Data Clear')
            l1 = self.intp_da.pop(0)
            l2 = self.intp_conf.pop(0)

            l1.remove()
            l2.remove()

        if not hasattr(self, 'df_slt_slider'):
            print('No Data yet - return')
            return

        #f_da = interpolate.interp1d(self.df_slt_slider['UTC'], self.df_slt_slider['Value'])
        #f_dconf = interpolate.interp1d(self.df_slt_confirmed['UTC'], self.df_slt_confirmed['Value'])
        interpolate_types = ['Interp 1D', 'Akima 1D', 'Cubic Spline', 'PCHIP 1D']

        interpolate_type = self.interpCombo.get()

        if interpolate_type == 'Akima 1D':
            f_da = interpolate.Akima1DInterpolator(self.df_slt_slider['UTC'], self.df_slt_slider['DA'])
            f_dconf = interpolate.Akima1DInterpolator(self.df_slt_confirmed['UTC'], self.df_slt_confirmed['Conf'])
        elif interpolate_type == 'Interp 1D':
            f_da = interpolate.interp1d(self.df_slt_slider['UTC'], self.df_slt_slider['DA'])
            f_dconf = interpolate.interp1d(self.df_slt_confirmed['UTC'], self.df_slt_confirmed['Conf'])
        elif interpolate_types == 'Cubic Spline':
            f_da = interpolate.CubicSpline(self.df_slt_slider['UTC'], self.df_slt_slider['DA'])
            f_dconf = interpolate.CubicSpline(self.df_slt_confirmed['UTC'], self.df_slt_confirmed['Conf'])
        else:
            return
        sel_range = self.select_slider.val
        print(sel_range)

        # Create Array for x-Axis according to selected Channel and Sampling Rate on UTC base
        start_row = self.df_ecg.iloc[(self.df_ecg['UTC'] - sel_range[0]).abs().argsort()[:1]]
        print(f'Start row: {start_row}')
        print(f'Stop Index: {start_row.index[0]}')

        stop_row = self.df_ecg.iloc[(self.df_ecg['UTC'] - sel_range[1]).abs().argsort()[:1]]
        print(f'Stop row: {stop_row}')
        print(f'Stop Index: {stop_row.index[0]}')

        sel_df = self.df_ecg.iloc[start_row.index[0]:stop_row.index[0]]

        print(f'ECG Interpol Time Len: {len(sel_df)} and orginally {len(self.df_ecg)}')

        ecg_time = np.linspace(sel_range[0], sel_range[1], len(sel_df))
        new_data_array = f_da(ecg_time)
        new_data_confirmed = f_dconf(ecg_time)



        self.intp_da = self.ax4.plot(ecg_time, new_data_array, color='green')
        self.intp_conf = self.ax4.plot(ecg_time, new_data_confirmed, color='violet')


        print("Interpolate done; Redraw")
        self.figP1.canvas.draw()
        return

    def analyzeSelection(self):
        area = self.select_slider.val

        print(area)
        print(currentOpenFolder)

        ################################################################
        ##   ECG Analytics
        #################################################################
        sampling_rate = 300

        print(area[0]-self.start_time)

        lowerlimit = int(np.round((area[0]-self.start_time) * sampling_rate, 0))
        upperlimit = int(np.round((area[1]-self.start_time) * sampling_rate, 0))

        print(f'Limits: {lowerlimit} {upperlimit}')

        selected_data = self.df_ecg.loc[lowerlimit:upperlimit, ['II']]

        xqrs = processing.XQRS(sig=selected_data['II'], fs=sampling_rate)

        rr, heart_rate, heart_rate_r2, qrs_inds_plot, mean_rr, sdnn, mean_hr2, min_hr2, max_hr2, nnxx, pnnx, rmssd = analyze_qrs(
            selected_data['II'], xqrs, sampling_rate)

        ################################################################
        #  Labels
        #

        self.selectedDataFrame.hr_min_label.set(f'{np.round(min_hr2, 2)} /min')
        self.selectedDataFrame.hr_max_label.set(f'{np.round(max_hr2, 2)} /min')
        self.selectedDataFrame.hr_mean_label.set(f'{np.round(mean_hr2, 2)} /min')

        self.selectedDataFrame.rr_mean_label.set(f'{np.round(mean_rr, 2)} ms')

        self.selectedDataFrame.rmssd_label.set(f'{np.round(rmssd, 2)} ms')
        self.selectedDataFrame.nnxx_label.set(f'{np.round(nnxx, 2)}')
        self.selectedDataFrame.pnnxx_label.set(f'{np.round(pnnx, 2)} %')
        self.selectedDataFrame.sdnn_label.set(f'{np.round(sdnn, 2)} ms')

    def exportSelection(self):
        area = self.select_slider.val

        sampling_rate = 300

        start_row = self.df_ecg.iloc[(self.df_ecg['UTC'] - area[0]).abs().argsort()[:1]]
        print(f'Start row: {start_row}')

        stop_row = self.df_ecg.iloc[(self.df_ecg['UTC'] - area[1]).abs().argsort()[:1]]
        print(f'Stop row: {stop_row}')

        f = asksaveasfile(mode='w', defaultextension=".csv")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return

        # saving the DataFrame as a CSV file
        save_df = self.df_ecg.iloc[start_row.index[0]:stop_row.index[0]]
        gfg_csv_data = save_df.to_csv(f.name, index=False)

        f.close()
        return

    def exportMRF(self):
        print("I'm here to do the Export for a Multi Record File")

        selection = self.select_slider.val

        start_row = self.df_ecg.iloc[(self.df_ecg['UTC'] - selection[0]).abs().argsort()[:1]]
        print(f'ECG Start row: {start_row}')

        stop_row = self.df_ecg.iloc[(self.df_ecg['UTC'] - selection[1]).abs().argsort()[:1]]
        print(f'ECG Stop row: {stop_row}')

        # saving the DataFrame as a CSV file
        ecg_df = self.df_ecg.iloc[start_row.index[0]:stop_row.index[0]]

        f_da = interpolate.Akima1DInterpolator(self.df_slt_slider['UTC'], self.df_slt_slider['DA'])
        f_dconf = interpolate.Akima1DInterpolator(self.df_slt_confirmed['UTC'], self.df_slt_confirmed['Conf'])

        ecg_time = np.linspace(ecg_df['UTC'].iat[0], ecg_df['UTC'].iat[-1], len(ecg_df['UTC']))

        new_data_array = f_da(ecg_time)
        new_data_confirmed = f_dconf(ecg_time)

        df_data_array = pd.DataFrame(np.round(new_data_array, 2), columns=['DA'])
        df_data_confirmed = pd.DataFrame(np.round(new_data_confirmed, 2), columns=['Conf'])

        print(df_data_confirmed)

        save_df = pd.concat([ecg_df.reset_index(drop=True), df_data_array.reset_index(drop=True), df_data_confirmed.reset_index(drop=True)], axis=1)
        print(save_df)

        f = asksaveasfile(mode='w', defaultextension=".csv")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        gfg_csv_data = save_df.to_csv(f.name, index=False)

        f.close()
        print("Export MRF: All Done!")
        return

    def updateECG(self):
        print("I'm Here to Update the plot")

        print("All Done!")
    def showECG(self, i):
        print(f'Show ECG iter: {i}')

        # Check if Button is pressed on Empty Data
        if not hasattr(self, 'df_ecg'):
            print('No Data Yet')
            return

        ################################################################
        ##   ECG Analytics
        #################################################################
        sampling_rate = 300
        print(f'Plot1 Selected: {self.ax1Combo.get()}')
        print(f'Plot2 Selected: {self.ax2Combo.get()}')

        current_channel1 = self.ax1Combo.get()
        current_channel2 = self.ax2Combo.get()

        xqrs = processing.XQRS(sig=self.df_ecg[current_channel1], fs=sampling_rate)

        rr, heart_rate, heart_rate_r2, qrs_inds_plot, mean_rr, sdnn, mean_hr2, min_hr2, max_hr2, nnxx, pnnx, rmssd = analyze_qrs(
            self.df_ecg[current_channel1], xqrs, sampling_rate)

        ################################################################
        #  Labels
        #

        self.rawDataFrame.hr_min_label.set(f'{np.round(min_hr2, 2)} /min')
        self.rawDataFrame.hr_max_label.set(f'{np.round(max_hr2, 2)} /min')
        self.rawDataFrame.hr_mean_label.set(f'{np.round(mean_hr2, 2)} /min')

        self.rawDataFrame.rr_mean_label.set(f'{np.round(mean_rr, 2)} ms')

        self.rawDataFrame.rmssd_label.set(f'{np.round(rmssd, 2)} ms')
        self.rawDataFrame.nnxx_label.set(f'{np.round(nnxx, 2)}')
        self.rawDataFrame.pnnxx_label.set(f'{np.round(pnnx, 2)} %')
        self.rawDataFrame.sdnn_label.set(f'{np.round(sdnn, 2)} ms')
        ################################################################
        ##   Plotting
        #################################################################
        # Fixed Values
        global currentOpenFolder

        # Length of Signal
        ecg_runtime = len(self.df_ecg[current_channel1])/sampling_rate

        try:
            self.df_ecg['UTC']
            print('UTC Already exists')
            ecg_time = self.df_ecg['UTC']
            self.start_time = self.df_ecg['UTC'].values[0]
            self.stop_time = self.df_ecg['UTC'].values[-1]

        except KeyError:
            print("No UTC in Data Frame - will be added")
            stop_time = os.path.getmtime(currentOpenFolder)

            print(f'ShowECG: Stop Time: {stop_time}, Runtime: {len(self.df_ecg[current_channel1])}')

            start_time = stop_time - (len(self.df_ecg[current_channel1]) / sampling_rate)

            print(
                f'Start: {datetime.datetime.fromtimestamp(start_time)}, Stop: {datetime.datetime.fromtimestamp(stop_time)}')

            start_time2 = datetime.datetime.fromtimestamp(stop_time) - datetime.timedelta(
                seconds=len(self.df_ecg[current_channel1]) / sampling_rate)
            print(start_time2)

            self.stop_time = stop_time
            self.start_time = start_time2.timestamp()

            # Create Array for x-Axis according to selected Channel and Sampling Rate on UTC base
            ecg_time = np.linspace(start_time2.timestamp(), stop_time, len(self.df_ecg[current_channel1]))

            df_UTC = pd.DataFrame(ecg_time, columns=['UTC'])
            self.df_ecg = pd.concat([self.df_ecg, df_UTC], axis=1)

            print('UTC Added')
            print(self.df_ecg)

        init_range = [self.start_time, self.stop_time]

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.rraxis.clear()

        self.ax1.set_ylabel(f'Ch {current_channel1}')
        self.ax2.set_ylabel(f'Ch {current_channel2}')
        self.ax3.set_ylabel('Heart Rate', color='red')

        self.rraxis.set_ylabel('RR-Interval', color='green')

        #self.ax1.grid()
        #self.ax2.grid()
        self.ax3.grid()

        # Plot Ch. II and Resp Curve
        self.ax1.plot(ecg_time, self.df_ecg[current_channel1])
        self.ax2.plot(ecg_time, self.df_ecg[current_channel2])

        self.ax1.set_xlim(init_range)

        # Plot Heart Rate from WFDB
        self.ax3.plot(ecg_time, heart_rate, color='red', zorder=2.5)

        # Plot Heart Rate
        self.ax3.scatter(qrs_inds_plot+self.start_time, np.insert(heart_rate_r2, 0, 0), color='blue', s=5,
                         zorder=2.5)

        # Create Duplicate of Ax3 for Secondary y-Axis
        self.ax1.vlines(qrs_inds_plot+self.start_time, ymin=0, ymax=255, linestyle='dashed', color='grey', linewidth=1)

        # Adjust RR Array to match the number of Sampling Points
        rrPlot = np.insert(rr, 0, np.mean(rr))

        # Plot RR
        self.rraxis.scatter(qrs_inds_plot+self.start_time, rrPlot, color='green', s=5)

        # Clear Sliders in Case of Reload
        self.slider_ax.clear()

        self.span_slider = RangeSlider(ax=self.slider_ax, label="Zoom", valmin=self.start_time, valmax=self.stop_time,
                                        valinit=init_range, valstep=0.001)
        self.slider_ax._slider = self.span_slider
        self.span_slider.on_changed(self.updateSpan)

        self.selectslider_ax.clear()

        self.left_selection = Rectangle([self.ax3.get_xlim()[0], self.ax3.get_ylim()[0]],
                                        0, self.ax3.get_ylim()[1] + 10, color='lightgrey')

        self.right_selection = Rectangle([self.ax3.get_xlim()[1], self.ax3.get_ylim()[0]],
                                         0, self.ax3.get_ylim()[1] + 10, color='lightgrey')

        self.select_slider = RangeSlider(ax=self.selectslider_ax, label="Selection", valmin=self.start_time, valmax=self.stop_time,
                                         valinit=init_range, valstep=1/sampling_rate)

        self.selectslider_ax._slider = self.select_slider
        self.select_slider.on_changed(self.updateSelectSlider)

        self.span_slider.set_val(init_range)
        self.select_slider.set_val(init_range)

        self.ax3.add_patch(self.left_selection)
        self.ax3.add_patch(self.right_selection)

        if hasattr(self, 'df_slt_slider'):
            self.ax4.vlines([self.start_time, self.stop_time], ymin=-0.5, ymax=10, linestyle='solid', color='red',
                            linewidth=1)
            print('HEEEEEEEEEEE')
            self.ax1.vlines([self.df_slt_slider['UTC'].values[0], self.df_slt_slider['UTC'].values[-1]], ymin=-0.5, ymax=260, linestyle='solid', color='red',
                            linewidth=1)

        self.figP1.canvas.draw()

        return

    def select_ecg_file(self):
        global currentOpenFolder
        global currentOpenStressFolder

        filetypes = (
            ('Log files', '*.log'),
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )

        filename = fd.askopenfile(
            title='Open a File',
            initialdir=currentOpenFolder,
            filetypes=filetypes)

        reply = askyesno(
            title='Selected File is:',
            message=filename.name
        )

        if reply:
            file = os.path.basename(filename.name)
            if file.endswith(".log"):
                print("YES LOG")
                currentOpenFolder = filename.name
                if currentOpenStressFolder == os.getcwd():
                    currentOpenStressFolder = currentOpenFolder

                self.label2text.set("Current ECG File:" + currentOpenFolder)

                stopp_time = os.path.getmtime(filename.name)
                print(f'Stopp at: {datetime.datetime.fromtimestamp(stopp_time)}')

                ################################################################
                ##   Load Data
                #################################################################

                # "G:\EKG\Crew8\VP18_Scenario2.log"

                print(f'Show ECG from: {currentOpenFolder}')

                global columns
                self.df_ecg = pd.read_csv(currentOpenFolder, usecols=columns, skiprows=1, delimiter="\t")

                self.showECG(0)

            elif file.endswith(".csv"):
                print("YES CSV")

                currentOpenFolder = filename.name
                self.label2text.set("Current File:" + currentOpenFolder)


                ################################################################
                ##   Load Data
                #################################################################

                # "G:\EKG\Crew8\VP18_Scenario2.log"

                print(f'Show ECG from: {currentOpenFolder}')

                columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']

                self.df_ecg = pd.read_csv(currentOpenFolder, usecols=columns, delimiter=",")

                self.showECG(0)

            elif file.endswith(".xdf"):
                print("YES xdf")
            else:
                print("NO Right Ending")

        print(reply, currentOpenFolder)

    def show_stresstrack(self):

        # Check if Button is pressed on Empty Data
        if not hasattr(self, 'df_slt_slider'):
            print('No Data Yet')
            return

        ################################################################
        ##   Plotting
        #################################################################
        self.ax4.clear()
        self.ax4.set_ylabel(f'Stress Level')
        self.ax4.grid()

        # Fixed Values
        slider_time = [datetime.datetime.fromtimestamp(t) for t in self.df_slt_slider['UTC']]
        confirmed_time = [datetime.datetime.fromtimestamp(t) for t in self.df_slt_confirmed['UTC']]

        print(f'Last Confirmed Time: {confirmed_time[-1]}')
        print(f'Last Slider Time: {slider_time[-1]}')

        # Plot
        self.ax4.scatter(self.df_slt_slider['UTC'], self.df_slt_slider['DA'], s=5)
        self.ax4.scatter(self.df_slt_confirmed['UTC'], self.df_slt_confirmed['Conf'], s=10)

        if hasattr(self, 'df_ecg'):
            self.ax4.vlines([self.start_time, self.stop_time], ymin=-0.5, ymax=10, linestyle='solid', color='red',
                            linewidth=1)
            print(self.df_slt_slider['UTC'].values[-1])
            self.ax1.vlines([self.df_slt_slider['UTC'].values[0], self.df_slt_slider['UTC'].values[-1]], ymin=-0.5, ymax=260, linestyle='solid', color='red',
                            linewidth=1)

        self.figP1.canvas.draw()

        return

    def select_stress_file(self):
        global currentOpenStressFolder
        global currentOpenFolder

        filetypes = (
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )

        filename_slider = fd.askopenfile(
            title='Open non Confirmed File',
            initialdir=currentOpenStressFolder,
            filetypes=filetypes)

        currentOpenStressFolder = filename_slider.name
        if currentOpenFolder == os.getcwd():
            currentOpenFolder = currentOpenStressFolder

        filename_confirmed = fd.askopenfile(
            title='Open Confirmed File',
            initialdir=currentOpenStressFolder,
            filetypes=filetypes)

        reply = askyesno(
            title='Selected Files are:',
            message=f'Slider: {filename_slider.name}, Confirmed: {filename_confirmed.name}'
        )

        if reply:
            file = os.path.basename(filename_slider.name)

            if file.endswith(".csv"):
                print("YES CSV")

                self.label3text.set("Current Stress File:" + currentOpenStressFolder)

                columns = ['UTC', 'DA']

                columns_confirmed = ['UTC', 'Conf', 'Count']

                ################################################################
                ##   Load Data
                #################################################################
                self.df_slt_slider = pd.read_csv(filename_slider.name, delimiter=',', names=columns)

                self.df_slt_confirmed = pd.read_csv(filename_confirmed.name, delimiter=',', names=columns_confirmed)

                self.show_stresstrack()


            elif file.endswith(".xdf"):
                print("YES xdf")
            else:
                print("NO Right Ending")

        print(reply, filename_slider.name, filename_confirmed.name, 'Dialog over.')

    def loadFolder(self):
        print("I'm here to load a data Folder")
        global currentOpenFolder
        folderpath = fd.askdirectory(title='Select Data Folder', initialdir=currentOpenFolder)
        print(folderpath)

        # Files required: log ECG, 2x csv DA and Conf
        for root, subdirs, files in os.walk(folderpath, topdown=True):
            for file in files:
                print(file)

        #self.load_ecg_button.configure(bg='green')
        #self.load_stress_button.configure(bg='green')

    def loadMRF(self):
        print("I'm here to load MRFs")
        global currentOpenFolder

        filetypes = (
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )

        filename_mrf = fd.askopenfile(
            title='Select MulitRecord File',
            initialdir=currentOpenStressFolder,
            filetypes=filetypes)

        reply = askyesno(
            title='Selected Files are:',
            message=f'Correct MRF Path? \n {filename_mrf.name}'
        )

        if reply:
            file = os.path.basename(filename_mrf.name)

            if file.endswith(".csv"):
                print("YES CSV")

                columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp', 'UTC', 'DA', 'Conf']

                df_mrf = pd.read_csv(filename_mrf.name, delimiter=',', usecols=columns)

                self.df_ecg = df_mrf[['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp', 'UTC']]

                self.df_slt_slider = df_mrf[['UTC', 'DA']]
                self.df_slt_confirmed = df_mrf[['UTC', 'Conf']]

                self.showECG(0)
                self.show_stresstrack()

            else:
                print("NO Right Ending")

        print("All Done!")
    def shift_Left(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin - 1, xmax - 1]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_Right(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin + 1, xmax + 1]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_LeftK(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin - 10, xmax - 10]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_RightK(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin + 10, xmax + 10]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        ##########
        # Create Center Frames
        ##########
        top_frame = Frame(self) #, bg='yellow') # yellow bg for debugging
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=0)

        left_frame = Frame(self) #, bg='grey')
        left_frame.pack(side=tk.LEFT, fill=BOTH)

        right_frame = Frame(self) #, bg='blue')
        right_frame.pack(side=tk.RIGHT, fill=BOTH)

        ##########
        # Plot
        #

        # Create Figure and Axes
        self.figP1, (self.ax1, self.ax2, self.ax3) = plt.subplots(nrows=3, sharex=True)

        self.rraxis = self.ax3.twinx()

        self.slider_ax = self.figP1.add_axes([0.08, 0.04, 0.80, 0.03])

        self.selectslider_ax = self.figP1.add_axes([0.08, 0.01, 0.80, 0.03])

        # self.aniP1 = animation.FuncAnimation(self.figP1, self.showECG, interval=1000000, blit=False)

        # remove vertical gap between subplots
        self.figP1.subplots_adjust(top=0.995,
                                   bottom=0.1,
                                   left=0.04,
                                   right=0.96,
                                   hspace=0.0,
                                   wspace=0.105)

        # Packing
        canvasP1 = FigureCanvasTkAgg(self.figP1, self)
        canvasP1.draw()
        canvasP1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbarP1 = NavigationToolbar2Tk(canvasP1, self)
        toolbarP1.update()
        canvasP1._tkcanvas.pack(fill=tk.BOTH, expand=True)

        ##########
        # Labels
        #

        # Headline
        label = tk.Label(top_frame, text="SimplECG", font=LARGE_FONT)
        label.pack(pady=5, padx=5)

        # Current File Label
        # Needs to be known by Class as it's modified later
        self.label2text = tk.StringVar()
        self.label2text.set("Current File:" + currentOpenFolder)
        self.label2 = tk.Label(top_frame, textvariable=self.label2text, font=NORMAL_FONT)
        self.label2.pack(pady=5, padx=5)

        ##########
        # Navigation Frames
        #
        navFrame = LabelFrame(left_frame, text='Navigation')
        navFrame.pack(fill=X, expand=True)

        viewFrame = LabelFrame(left_frame, text='View Control')
        viewFrame.pack(fill=X, expand=True)

        selectionFrame = LabelFrame(left_frame, text='Channel Selection')
        selectionFrame.pack(fill=X, expand=True)

        analyzeFrame = LabelFrame(left_frame, text='Analyze')
        analyzeFrame.pack(fill=X, expand=True)

        exportFrame = LabelFrame(left_frame, text='Export')
        exportFrame.pack(fill=X, expand=True)

        ##########
        # navFrame
        #

        # Button to Get Back to Main Page
        button_Home = ttk.Button(navFrame, text="Back Home",
                                 command=lambda: controller.show_frame(StartPage))

        # Button to open Files
        load_ecg_button = ttk.Button(navFrame, text='Load ECG File',
                                 command=lambda: self.select_ecg_file())

        # Button to show current Folder
        current_button = ttk.Button(navFrame, text='Current ECG File',
                                    command=lambda: self.printName())

        ##########
        # selectionFrame
        #
        global columns
        ax1Label = ttk.Label(selectionFrame, text="Plot 1:", font=NORMAL_FONT)
        self.ax1Combo = ttk.Combobox(selectionFrame, values=columns)
        self.ax1Combo.set('II')


        ax2Label = ttk.Label(selectionFrame, text="Plot 2:", font=NORMAL_FONT)
        self.ax2Combo = ttk.Combobox(selectionFrame, values=columns)
        self.ax2Combo.set('Resp')

        self.ax1Combo.bind("<<ComboboxSelected>>", self.showECG)
        self.ax2Combo.bind("<<ComboboxSelected>>", self.showECG)

        ##########
        # analyzeFrame
        #

        # Button to analze current Area
        analyze_button = ttk.Button(analyzeFrame, text='Analyze Selection',
                                    command=lambda: self.analyzeSelection())

        start_label = tk.Label(analyzeFrame, text="Start at:", font=NORMAL_FONT)
        self.start_entry = ttk.Entry(analyzeFrame)

        stop_label = tk.Label(analyzeFrame, text="Stop at:", font=NORMAL_FONT)
        self.stop_entry = ttk.Entry(analyzeFrame)

        # Button to accpet inout Area
        setselection_button = ttk.Button(analyzeFrame, text='Set Selection',
                                         command=lambda: self.setSelection())

        ##########
        # exportFrame
        #

        # Button to Export Selected Data Area
        export_button = ttk.Button(exportFrame, text='Export Selection',
                                   command=lambda: self.exportSelection())

        ##########
        # viewFrame
        #

        min_label = tk.Label(viewFrame, text="min:", font=NORMAL_FONT)
        self.min_entry = ttk.Entry(viewFrame)
        max_label = tk.Label(viewFrame, text="max:", font=NORMAL_FONT)
        self.max_entry = ttk.Entry(viewFrame)

        # Button to set Limit Entries
        setEntry_button = ttk.Button(viewFrame, text='Set new Limits',
                                     command=lambda: self.setEntry())

        # Button to set Limit Entries
        resetZoom_button = ttk.Button(viewFrame, text='Reset Zoom',
                                     command=lambda: self.zoomOut())

        # Button to Shift 100 ticks Left
        shiftL_button = ttk.Button(viewFrame, text='<- 1 Seconds',
                                   command=lambda: self.shift_Left())

        # Button to Shift 100 ticks Rigth
        shiftR_button = ttk.Button(viewFrame, text='1 Second ->',
                                   command=lambda: self.shift_Right())

        # Button to Shift 1000 Left
        shiftLK_button = ttk.Button(viewFrame, text='<- 10 Seconds',
                                    command=lambda: self.shift_LeftK())

        # Button to Shift 1000 Rigth
        shiftRK_button = ttk.Button(viewFrame, text='10 Seconds ->',
                                    command=lambda: self.shift_RightK())
        ##########
        # Grid all up
        #
        button_Home.grid(row=0, column=0, pady=10, columnspan=2)#.pack(side=tk.TOP, expand=True)
        load_ecg_button.grid(row=1, column=0, pady=10)#.pack(side=tk.TOP, expand=True)
        current_button.grid(row=1, column=1, padx=10, pady=10)#.pack(side=tk.TOP, expand=True)

        start_label.grid(row=1, column=0, pady=10)
        self.start_entry.grid(row=1, column=1, pady=10)

        stop_label.grid(row=2, column=0, pady=10)
        self.stop_entry.grid(row=2, column=1, pady=10)

        setselection_button.grid(row=4, column=0, pady=10)#.pack(side=tk.TOP, expand=True)
        analyze_button.grid(row=6, column=0, pady=10)#.pack(side=tk.TOP, expand=True)

        export_button.grid(row=0, column=0, pady=10)#.pack(side=tk.TOP, expand=True)

        min_label.grid(row=0, column=0, pady=10)#.pack(side=tk.TOP, expand=True)
        self.min_entry.grid(row=0, column=1, pady=10)#.pack(side=tk.TOP, expand=True)
        max_label.grid(row=2, column=0, pady=10)#.pack(side=tk.TOP, expand=True)
        self.max_entry.grid(row=2, column=1, pady=10)#.pack(side=tk.TOP, expand=True)
        setEntry_button.grid(row=4, column=0, pady=10)#.pack(side=tk.TOP, expand=True)
        resetZoom_button.grid(row=4, column=1, pady=10)#.pack(side=tk.TOP, expand=True)


        shiftL_button.grid(row=5, column=0, pady=10)#.pack(side=tk.TOP, expand=True)
        shiftR_button.grid(row=5, column=1, pady=10)#.pack(side=tk.TOP, expand=True)
        shiftLK_button.grid(row=6, column=0, pady=10)#.pack(side=tk.TOP, expand=True)
        shiftRK_button.grid(row=6, column=1, pady=10)#.pack(side=tk.TOP, expand=True)

        ax1Label.grid(row=0, column=0, pady=10)
        self.ax1Combo.grid(row=0, column=1, pady=10)

        ax2Label.grid(row=2, column=0, pady=10)
        self.ax2Combo.grid(row=2, column=1, pady=10)

        ##########
        # Analysis Read Out Frame
        #

        #rawDataFrame = LabelFrame(right_frame, text='Original Dataset')
        self.rawDataFrame = ECGReadOutFrame(right_frame, text='Original Dataset')
        self.rawDataFrame.pack(fill=X, expand=True)

        self.selectedDataFrame = ECGReadOutFrame(right_frame, text='Selected Dataset')
        self.selectedDataFrame.pack(fill=X, expand=True)

    # Function to be rendered anytime a slider's value changes
    def setEntry(self):
        minVal = self.min_entry.get()
        maxVal = self.max_entry.get()
        if minVal:
            if maxVal:
                if maxVal > minVal:
                    print(minVal, maxVal)
                    self.span_slider.set_val([float(minVal), float(maxVal)])
                    self.figP1.canvas.draw()
                else:
                    reply = askyesnocancel(title="Wrong Order", message="Min Value higher than max Value. Want switch?")
                    if reply:
                        print(f"Min Value: {maxVal} Max Value: {minVal}")
                        self.min_entry.delete(0, END)
                        self.max_entry.delete(0, END)
                        self.min_entry.insert(0, maxVal)
                        self.max_entry.insert(0, minVal)
                        self.span_slider.set_val([float(maxVal), float(minVal)])
            else:
                showerror(title="No Value", message="Please enter a Max Value")
        else:
            showerror(title="No Value", message="Please enter a Min Value")

    def zoomOut(self):
        self.span_slider.set_val([0, len(self.df_ecg)/300])

    def setSelection(self):
        startVal = self.start_entry.get()
        stoppVal = self.stop_entry.get()

        self.select_slider.set_val([float(startVal), float(stoppVal)])

        self.figP1.canvas.draw()

    def updateSpan(self, val):
        # print(f'Span: {self.span_slider.val}')

        self.ax1.set_xlim(self.span_slider.val[0], self.span_slider.val[1])
        self.min_entry.delete(0, END)
        self.max_entry.delete(0, END)
        self.min_entry.insert(0, self.span_slider.val[0])
        self.max_entry.insert(0, self.span_slider.val[1])
        self.figP1.canvas.draw()

    def updateSelectSlider(self, val):
        print(f'Selected: {val}')

        self.start_entry.delete(0, END)
        self.start_entry.insert(0, val[0])

        self.stop_entry.delete(0, END)
        self.stop_entry.insert(0, val[1])

        self.left_selection.set_width(val[0])

        self.right_selection.set_x(val[1])
        self.right_selection.set_width(len(self.df_ecg)/300 - val[1])

        self.figP1.canvas.draw()



    def printName(self):
        print(currentOpenFolder)

    def analyzeSelection(self):
        area = self.select_slider.val

        print(area)
        print(currentOpenFolder)
        ################################################################
        ##   ECG Analytics
        #################################################################
        sampling_rate = 300

        lowerlimit = int(np.round(area[0] * sampling_rate, 0))
        upperlimit = int(np.round(area[1] * sampling_rate, 0))

        print(f'Limits: {lowerlimit} {upperlimit}')

        selected_data = self.df_ecg.loc[lowerlimit:upperlimit, ['II']]

        xqrs = processing.XQRS(sig=selected_data['II'], fs=sampling_rate)


        rr, heart_rate, heart_rate_r2, qrs_inds_plot, mean_rr, sdnn, mean_hr2, min_hr2, max_hr2, nnxx, pnnx, rmssd = analyze_qrs(selected_data['II'], xqrs, sampling_rate)

        ################################################################
        #  Labels
        #

        self.selectedDataFrame.hr_min_label.set(f'{np.round(min_hr2,2)} /min')
        self.selectedDataFrame.hr_max_label.set(f'{np.round(max_hr2, 2)} /min')
        self.selectedDataFrame.hr_mean_label.set(f'{np.round(mean_hr2, 2)} /min')

        self.selectedDataFrame.rr_mean_label.set(f'{np.round(mean_rr, 2)} ms')

        self.selectedDataFrame.rmssd_label.set(f'{np.round(rmssd, 2)} ms')
        self.selectedDataFrame.nnxx_label.set(f'{np.round(nnxx, 2)}')
        self.selectedDataFrame.pnnxx_label.set(f'{np.round(pnnx, 2)} %')
        self.selectedDataFrame.sdnn_label.set(f'{np.round(sdnn, 2)} ms')


    def exportSelection(self):
        area = self.select_slider.val

        sampling_rate = 300

        lowerlimit = int(np.round(area[0] * sampling_rate, 0))
        upperlimit = int(np.round(area[1] * sampling_rate, 0))

        print(f'Limits: {lowerlimit} {upperlimit}')

        f = asksaveasfile(mode='w', defaultextension=".csv")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return

        # saving the DataFrame as a CSV file
        save_df = self.df_ecg.iloc[lowerlimit:upperlimit]
        gfg_csv_data = save_df.to_csv(f.name, index=False)
        f.close()
        return

    def updateECG(self):
        print("I'm Here to Update the pplot")


    def showECG(self, i):
        print(f'Show ECG iter: {i}')

        # Check if Button is pressed on Empty Data
        if not hasattr(self, 'df_ecg'):
            print('No Data Yet')
            return

        ################################################################
        ##   ECG Analytics
        #################################################################
        sampling_rate = 300
        print(f'Plot1 Selected: {self.ax1Combo.get()}')
        print(f'Plot2 Selected: {self.ax2Combo.get()}')

        current_channel1 = self.ax1Combo.get()
        current_channel2 = self.ax2Combo.get()

        xqrs = processing.XQRS(sig=self.df_ecg[current_channel1], fs=sampling_rate)

        rr, heart_rate, heart_rate_r2, qrs_inds_plot, mean_rr, sdnn, mean_hr2, min_hr2, max_hr2, nnxx, pnnx, rmssd = analyze_qrs(self.df_ecg[current_channel1], xqrs, sampling_rate)

        ################################################################
        #  Labels
        #

        self.rawDataFrame.hr_min_label.set(f'{np.round(min_hr2,2)} /min')
        self.rawDataFrame.hr_max_label.set(f'{np.round(max_hr2, 2)} /min')
        self.rawDataFrame.hr_mean_label.set(f'{np.round(mean_hr2, 2)} /min')

        self.rawDataFrame.rr_mean_label.set(f'{np.round(mean_rr, 2)} ms')

        self.rawDataFrame.rmssd_label.set(f'{np.round(rmssd, 2)} ms')
        self.rawDataFrame.nnxx_label.set(f'{np.round(nnxx, 2)}')
        self.rawDataFrame.pnnxx_label.set(f'{np.round(pnnx, 2)} %')
        self.rawDataFrame.sdnn_label.set(f'{np.round(sdnn, 2)} ms')
        ################################################################
        ##   Plotting
        #################################################################
        # Fixed Values
        ecg_runtime = len(self.df_ecg[current_channel1]) / sampling_rate

        # Create Array for x-Axis according to Ch. II and Sampling Rate
        ecg_time = np.arange(0.0, ecg_runtime, 1 / sampling_rate)
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.rraxis.clear()

        self.ax1.set_ylabel(f'Ch {current_channel1}')
        self.ax2.set_ylabel(f'Ch {current_channel2}')
        self.ax3.set_ylabel('Heart Rate', color='red')

        self.rraxis.set_ylabel('RR-Interval', color='green')

        self.ax1.grid()
        self.ax2.grid()
        self.ax3.grid()

        # Plot Ch. II and Resp Curve
        self.ax1.plot(ecg_time, self.df_ecg[current_channel1])
        self.ax2.plot(ecg_time, self.df_ecg[current_channel2])

        self.ax1.set_xlim([0, ecg_runtime])

        # Plot Heart Reate
        self.ax3.plot(ecg_time, heart_rate, color='red', zorder=2.5)
        # Plot Heart Rate from WFDB
        self.ax3.scatter(xqrs.qrs_inds / sampling_rate, np.insert(heart_rate_r2, 0, 0), color='blue', s=5,
                         zorder=2.5)

        # Create Duplicate of Ax3 for Secondary y-Axis
        self.ax1.vlines(qrs_inds_plot, ymin=0, ymax=255, linestyle='dashed', color='grey', linewidth=1)

        # Adjust RR Array to match the number of Sampling Points
        rrPlot = np.insert(rr, 0, np.mean(rr))

        ## Plot RR
        self.rraxis.scatter(qrs_inds_plot, rrPlot, color='green', s=5)

        # Clear Sliders in Case of Reload
        self.slider_ax.clear()

        self.span_slider = RangeSlider(ax=self.slider_ax, label="Zoom", valmin=0, valmax=ecg_runtime,
                                       valinit=[0, ecg_runtime], valstep=0.01)
        self.slider_ax._slider = self.span_slider
        self.span_slider.on_changed(self.updateSpan)

        self.selectslider_ax.clear()

        self.left_selection = Rectangle([0, self.ax3.get_ylim()[0]+1],
                                        0, self.ax3.get_ylim()[1] + 10, color='lightgrey')

        self.right_selection = Rectangle([self.ax3.get_xlim()[1], self.ax3.get_ylim()[0]],
                                         0, self.ax3.get_ylim()[1] + 10, color='lightgrey')

        self.select_slider = RangeSlider(ax=self.selectslider_ax, label="Selection", valmin=0, valmax=ecg_runtime,
                                         valinit=[0, ecg_runtime], valstep=0.01)
        self.selectslider_ax._slider = self.select_slider
        self.select_slider.on_changed(self.updateSelectSlider)

        self.span_slider.set_val([0, ecg_runtime])
        self.select_slider.set_val([0, ecg_runtime])


        self.ax3.add_patch(self.left_selection)
        self.ax3.add_patch(self.right_selection)

        self.figP1.canvas.draw()


        return

    def select_ecg_file(self):
        global currentOpenFolder
        filetypes = (
            ('Log files', '*.log'),
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )

        filename = fd.askopenfile(
            title='Open a File',
            initialdir=currentOpenFolder,
            filetypes=filetypes)

        reply = askyesno(
            title='Selected File is:',
            message=filename.name
        )

        if reply:
            file = os.path.basename(filename.name)
            if file.endswith(".log"):
                print("YES LOG")
                currentOpenFolder = filename.name
                self.label2text.set("Current File:" + currentOpenFolder)
                ################################################################
                ##   Load Data
                #################################################################

                # "G:\EKG\Crew8\VP18_Scenario2.log"

                print(f'Show ECG from: {currentOpenFolder}')

                global columns
                self.df_ecg = pd.read_csv(currentOpenFolder, usecols=columns, skiprows=1, delimiter="\t")

                self.showECG(0)

            elif file.endswith(".csv"):
                print("YES CSV")

                currentOpenFolder = filename.name
                self.label2text.set("Current File:" + currentOpenFolder)
                ################################################################
                ##   Load Data
                #################################################################

                # "G:\EKG\Crew8\VP18_Scenario2.log"

                print(f'Show ECG from: {currentOpenFolder}')

                columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']

                self.df_ecg = pd.read_csv(currentOpenFolder, usecols=columns, delimiter=",")

                self.showECG(0)

            elif file.endswith(".xdf"):
                print("YES xdf")
            else:
                print("NO Right Ending")

        print(reply, currentOpenFolder)

    def shift_Left(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin - 1, xmax - 1]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_Right(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin + 1, xmax + 1]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_LeftK(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin - 10, xmax - 10]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_RightK(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin + 10, xmax + 10]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        global currentOpenFolder

        ##########
        # Interface
        ##########
        label = tk.Label(self, text="SimplECG", font=LARGE_FONT)
        label.pack(pady=5, padx=5)

        self.label2text = tk.StringVar()
        self.label2text.set("Current File:" + currentOpenFolder)
        self.label2 = tk.Label(self, textvariable=self.label2text, font=NORMAL_FONT)
        self.label2.pack(pady=5, padx=5)

        # Button to Get Back to Main Page
        button_Home = ttk.Button(self, text="Back Home",
                                 command=lambda: controller.show_frame(StartPage))

        # Button to open Files
        open_button = ttk.Button(self, text='Open File',
                                 command=lambda: self.select_file())

        # Button to show current Folder
        current_button = ttk.Button(self, text='Current File',
                                    command=lambda: self.printName())

        # Button to analze current Area
        analyze_button = ttk.Button(self, text='Analyze Selection',
                                    command=lambda: self.analyzeSelection())

        # Button to Export Selected Data Area
        export_button = ttk.Button(self, text='Export Selection',
                                   command=lambda: self.exportSelection())

        min_label = tk.Label(self, text="min:", font=NORMAL_FONT)
        self.min_entry = ttk.Entry(self)
        max_label = tk.Label(self, text="max:", font=NORMAL_FONT)
        self.max_entry = ttk.Entry(self)

        # Button to set Limit Entries
        setEntry_button = ttk.Button(self, text='Set new Limits',
                                     command=lambda: self.setEntry())

        # Button to Shift 100 ticks Left
        shiftL_button = ttk.Button(self, text='<- 1 Seconds',
                                   command=lambda: self.shift_Left())

        # Button to Shift 100 ticks Rigth
        shiftR_button = ttk.Button(self, text='1 Second ->',
                                   command=lambda: self.shift_Right())

        # Button to Shift 1000 Left
        shiftLK_button = ttk.Button(self, text='<- 10 Seconds',
                                    command=lambda: self.shift_LeftK())

        # Button to Shift 1000 Rigth
        shiftRK_button = ttk.Button(self, text='10 Seconds ->',
                                    command=lambda: self.shift_RightK())

        ##########
        # Plots
        ##########

        # Create Figure and Axes
        self.figP1, (self.ax1, self.ax2, self.ax3) = plt.subplots(nrows=3, sharex=True)

        self.ax1.set_ylabel('Ch II')
        self.ax2.set_ylabel('Ch Resp')
        self.ax3.set_ylabel('Heart Rate', color='red')
        self.rraxis = self.ax3.twinx()
        self.rraxis.set_ylabel('RR-Interval', color='green')

        self.slider_ax = self.figP1.add_axes([0.08, 0.05, 0.80, 0.03])

        self.selectslider_ax = self.figP1.add_axes([0.08, 0.01, 0.80, 0.03])

        # self.aniP1 = animation.FuncAnimation(self.figP1, self.showECG, interval=1000000, blit=False)

        # remove vertical gap between subplots
        self.figP1.subplots_adjust(top=0.995,
                                   bottom=0.1,
                                   left=0.04,
                                   right=0.96,
                                   hspace=0.0,
                                   wspace=0.105)

        ##########
        # Packing
        ##########
        canvasP1 = FigureCanvasTkAgg(self.figP1, self)
        canvasP1.draw()
        canvasP1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbarP1 = NavigationToolbar2Tk(canvasP1, self)
        toolbarP1.update()
        canvasP1._tkcanvas.pack(fill=tk.BOTH, expand=True)

        # Navigation Frame

        button_Home.pack(side=tk.LEFT, expand=True)
        open_button.pack(side=tk.LEFT, expand=True)
        current_button.pack(side=tk.LEFT, expand=True)
        analyze_button.pack(side=tk.LEFT, expand=True)
        export_button.pack(side=tk.LEFT, expand=True)

        min_label.pack(side=tk.LEFT, expand=True)
        self.min_entry.pack(side=tk.LEFT, expand=True)
        max_label.pack(side=tk.LEFT, expand=True)
        self.max_entry.pack(side=tk.LEFT, expand=True)
        setEntry_button.pack(side=tk.LEFT, expand=True)

        shiftLK_button.pack(side=tk.LEFT, expand=True)
        shiftL_button.pack(side=tk.LEFT, expand=True)
        shiftR_button.pack(side=tk.LEFT, expand=True)
        shiftRK_button.pack(side=tk.LEFT, expand=True)

    # Function to be rendered anytime a slider's value changes
    def setEntry(self):
        minVal = self.min_entry.get()
        maxVal = self.max_entry.get()
        if minVal:
            if maxVal:
                if maxVal > minVal:
                    print(minVal, maxVal)
                    self.span_slider.set_val([float(minVal), float(maxVal)])
                    self.figP1.canvas.draw()
                else:
                    reply = askyesnocancel(title="Wrong Order", message="Min Value higher than max Value. Want switch?")
                    if reply:
                        print(f"Min Value: {maxVal} Max Value: {minVal}")
                        self.min_entry.delete(0, END)
                        self.max_entry.delete(0, END)
                        self.min_entry.insert(0, maxVal)
                        self.max_entry.insert(0, minVal)
                        self.span_slider.set_val([float(maxVal), float(minVal)])
            else:
                showerror(title="No Value", message="Please enter a Max Value")
        else:
            showerror(title="No Value", message="Please enter a Min Value")

    def updateSpan(self, val):
        # print(f'Span: {self.span_slider.val}')

        self.ax1.set_xlim(self.span_slider.val[0], self.span_slider.val[1])
        self.min_entry.delete(0, END)
        self.max_entry.delete(0, END)
        self.min_entry.insert(0, self.span_slider.val[0])
        self.max_entry.insert(0, self.span_slider.val[1])
        self.figP1.canvas.draw()

    def updateSelectSlider(self, val):
        print(val)
        self.left_selection.set_width(val[0])
        self.right_selection.set_x(val[1])
        self.right_selection.set_width(self.ax3.get_xlim()[1] - val[1])

        self.figP1.canvas.draw()

    def printName(self):
        print(currentOpenFolder)

    def analyzeSelection(self):
        area = self.select_slider.val

        print(area)
        print(currentOpenFolder)
        ################################################################
        ##   ECG Analytics
        #################################################################
        sampling_rate = 300

        lowerlimit = int(np.round(area[0] * sampling_rate, 0))
        upperlimit = int(np.round(area[1] * sampling_rate, 0))

        print(f'Limits: {lowerlimit} {upperlimit}')

        selected_data = self.df_ecg.loc[lowerlimit:upperlimit, ['II']]

        xqrs = processing.XQRS(sig=selected_data['II'], fs=sampling_rate)
        analyze_qrs(selected_data['II'], xqrs, sampling_rate)

    def exportSelection(self):
        area = self.select_slider.val

        sampling_rate = 300

        lowerlimit = int(np.round(area[0] * sampling_rate, 0))
        upperlimit = int(np.round(area[1] * sampling_rate, 0))

        print(f'Limits: {lowerlimit} {upperlimit}')

        f = asksaveasfile(mode='w', defaultextension=".csv")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return

        # saving the DataFrame as a CSV file
        save_df = self.df_ecg.iloc[lowerlimit:upperlimit]
        gfg_csv_data = save_df.to_csv(f.name, index=False)
        f.close()

    def showECG(self, i):
        print(f'Show ECG iter: {i}')

        ################################################################
        ##   ECG Analytics
        #################################################################
        sampling_rate = 300

        xqrs = processing.XQRS(sig=self.df_ecg['II'], fs=sampling_rate)

        rr, heart_rate, heart_rate_r2, qrs_inds_plot = analyze_qrs(self.df_ecg['II'], xqrs, sampling_rate)

        ################################################################
        ##   Plotting
        #################################################################
        # Fixed Values
        ecg_runtime = len(self.df_ecg['II']) / sampling_rate

        # Create Array for x-Axis according to Ch. II and Sampling Rate
        ecg_time = np.arange(0.0, ecg_runtime, 1 / sampling_rate)
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.rraxis.clear()

        self.ax1.grid()
        self.ax2.grid()
        self.ax3.grid()

        # Plot Ch. II and Resp Curve
        self.ax1.plot(ecg_time, self.df_ecg['II'])
        self.ax2.plot(ecg_time, self.df_ecg['Resp'])

        self.ax1.set_xlim([0, ecg_runtime])

        # Plot Heart Reate
        self.ax3.plot(ecg_time, heart_rate, color='red', zorder=2.5)
        # Plot Heart Rate from WFDB
        self.ax3.scatter(xqrs.qrs_inds / sampling_rate, np.insert(heart_rate_r2, 0, 0), color='blue', s=5,
                         zorder=2.5)

        # Create Duplicate of Ax3 for Secondary y-Axis
        self.ax1.vlines(qrs_inds_plot, ymin=0, ymax=255, linestyle='dashed', color='grey', linewidth=1)

        # Adjust RR Array to match the number of Sampling Points
        rrPlot = np.insert(rr, 0, np.mean(rr))

        ## Plot RR
        self.rraxis.scatter(qrs_inds_plot, rrPlot, color='green', s=5)

        # Clear Sliders in Case of Reload
        self.slider_ax.clear()

        self.span_slider = RangeSlider(ax=self.slider_ax, label="Time", valmin=0, valmax=ecg_runtime,
                                       valinit=[0, ecg_runtime], valstep=0.01)
        self.slider_ax._slider = self.span_slider
        self.span_slider.on_changed(self.updateSpan)

        self.selectslider_ax.clear()
        self.select_slider = RangeSlider(ax=self.selectslider_ax, label="Selection", valmin=0, valmax=ecg_runtime,
                                         valinit=[0, ecg_runtime], valstep=0.01)
        self.selectslider_ax._slider = self.select_slider
        self.select_slider.on_changed(self.updateSelectSlider)

        self.left_selection = Rectangle([0, self.ax3.get_ylim()[0]],
                                        0, self.ax3.get_ylim()[1] + 10, color='lightgrey')

        self.right_selection = Rectangle([self.ax3.get_xlim()[1], self.ax3.get_ylim()[0]],
                                         0, self.ax3.get_ylim()[1] + 10, color='lightgrey')

        self.ax3.add_patch(self.left_selection)
        self.ax3.add_patch(self.right_selection)

        self.figP1.canvas.draw()

        return

    def select_file(self):
        global currentOpenFolder
        filetypes = (
            ('Log files', '*.log'),
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )

        filename = fd.askopenfile(
            title='Open a File',
            initialdir=currentOpenFolder,
            filetypes=filetypes)

        reply = askyesno(
            title='Selected File is:',
            message=filename.name
        )

        if reply:
            file = os.path.basename(filename.name)
            if file.endswith(".log"):
                print("YES LOG")
                currentOpenFolder = filename.name
                self.label2text.set("Current File:" + currentOpenFolder)
                ################################################################
                ##   Load Data
                #################################################################

                # "G:\EKG\Crew8\VP18_Scenario2.log"

                print(f'Show ECG from: {currentOpenFolder}')

                columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']

                self.df_ecg = pd.read_csv(currentOpenFolder, usecols=columns, skiprows=1, delimiter="\t")

                self.showECG(0)

            elif file.endswith(".csv"):
                print("YES CSV")

                currentOpenFolder = filename.name
                self.label2text.set("Current File:" + currentOpenFolder)
                ################################################################
                ##   Load Data
                #################################################################

                # "G:\EKG\Crew8\VP18_Scenario2.log"

                print(f'Show ECG from: {currentOpenFolder}')

                columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'C1', 'Resp']

                self.df_ecg = pd.read_csv(currentOpenFolder, usecols=columns, delimiter=",")

                self.showECG(0)

            elif file.endswith(".xdf"):
                print("YES xdf")
            else:
                print("NO Right Ending")

        print(reply, currentOpenFolder)

    def shift_Left(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin - 1, xmax - 1]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_Right(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin + 1, xmax + 1]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_LeftK(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin - 10, xmax - 10]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)

    def shift_RightK(self):
        xmin, xmax = self.ax1.get_xlim()
        # print(f'Xmin Xmax: {xmin} {xmax}')
        val = [xmin + 10, xmax + 10]
        self.span_slider.set_val(val=val)
        self.updateSpan(val=val)


def main():
    app = MainApp()
    #app.geometry("600x400+20+20")
    app.minsize(width=800, height=1000)
    app.mainloop()


if __name__ == '__main__':
    main()
