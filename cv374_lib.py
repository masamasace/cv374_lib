from pathlib import Path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import gc
import scipy.signal as ss
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import math

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "k"

def setup_figure(num_row=1, num_col=1, width=5, height=4, left=0.125, right=0.9, hspace=0.2, wspace=0.2):

    fig, axes = plt.subplots(num_row, num_col, figsize=(width, height), squeeze=False)   
    fig.subplots_adjust(left=left, right=right, hspace=hspace, wspace=wspace)
    return (fig, axes)


## class for initialize directory structure and processed data
class InitializeData:
    
    def __init__(self, dir_path) -> None:
        
        self.dir_path = Path(dir_path).resolve()
        
        # search .t3w file and .log file
        temp_t3w_file_list = list(self.dir_path.glob("**.t3w"))
        
        
               

class CV374Data:
    """
    Class for reading and processing CV374 data.

    Attributes:
        freq (int): Frequency of the data.
        ylim (list): Y-axis limits for time series record.
        start_indexes (list): Start indexes for exporting time series record.
        clip_num_data (int): Number of data points to clip for HV spectrum calculation.
        data_path (str): Directory path of CV374 data.
        dir_result (str): Directory path for saving results.
        time_dif (int): Time difference between UTC and local time.
        is_export_timeseries_butterworth_filter (bool): Flag for applying Butterworth filter.
        is_cosine_taper (bool): Flag for applying cosine taper.
        is_parzen_smoothing (bool): Flag for applying Parzen smoothing.
        parzen_width (float): Width parameter for Parzen smoothing.
        is_only_show_parzen_filter_result (bool): Flag for showing only Parzen filter result.
        is_export_figure_3D_running_spectra (bool): Flag for exporting 3D running spectra.
        surface_thin_out_interval (int): Interval for thinning out the surface in 3D running spectra.
        freq_lim (list): Frequency limits for HV spectrum.
        HVSR_lim (list): HVSR limits for HV spectrum.

    Methods:
        __init__(): Initializes the CV374Data object.
        read_data(data_path, time_dif=-6): Reads the data from the specified data path and stores it in the object.
        export_time_series_record(ylim=[-0.001, 0.001]): Export the time series record as a plot.
        calcurate_HV_spectrum(clip_num_data=16384, cosine_taper=True, is_export_timeseries_butterworth_filter=False,
                              butterworth_order=2, butterworth_lowcut=0.1, butterworth_highcut=20,
                              is_parzen_smoothing=True, parzen_width=0.2, is_konno_ohmachi_smoothing=False,
                              konno_ohmachi_width=0.2, is_export_figure_3D_running_spectra=False,
                              surface_thin_out_interval=10, is_export_figure_2D_running_spectra=False,
                              freq_lim=[0.1, 50], HVSR_lim=[0.05, 50]): Calculates the HV spectrum.

    """

    def __init__(self) -> None:
        """
        Initializes the CV374Data object.

        Default values for CV374Data object attributes are set in this method.

        """
        self.freq = 100  # Frequency of the data
        self.ylim = [-0.001, 0.001]  # Y-axis limits for time series record
        self.start_indexes = [0, 5000, 10000]  # Start indexes for exporting time series record
        
        self.total_num_data = 90000  # Total number of data points
        self.clip_num_data = 16384  # Number of data points to clip for HV spectrum calculation
        self.data_path = ""  # Directory path of CV374 data
        self.dir_result = ""  # Directory path for saving results
        self.time_dif = 0  # Time difference between UTC and local time
        self.max_index_HVSR = 0  # Maximum index for HVSR power spectrum
        
        self.is_apply_butterworth_filter = True  # Flag for applying Butterworth filter
        self.is_export_timeseries_butterworth_filter = False  # Flag for applying Butterworth filter
        self.is_cosine_taper = True  # Flag for applying cosine taper
        self.is_parzen_smoothing = True  # Flag for applying Parzen smoothing
        self.parzen_width = 0.2  # Width parameter for Parzen smoothing
        self.is_only_show_parzen_filter_result = True  # Flag for showing only Parzen filter result
        self.is_export_figure_3D_running_spectra = False  # Flag for exporting 3D running spectra
        self.is_export_figure_2D_running_spectra = False  # Flag for exporting 3D running spectra
        self.is_export_csv_geomean_spectrum = False  # Flag for exporting 3D running spectra
        self.surface_thin_out_interval = 10  # Interval for thinning out the surface in 3D running spectra
        self.freq_lim = [0.1, 50]  # Frequency limits for HV spectrum
        self.HVSR_lim = [0.01, 100]  # HVSR limits for HV spectrum
        
        self.col_names = []  # Column names of the time series data
        self.time_series_data = None  # Time series data
        self.asc_file_stem_list = []  # List of file stem names for exporting figures

    def read_data(self, data_path, time_dif=-6) -> None:
        """
        Reads the data from the specified data path and stores it in the object.

        Args:
            data_path (str): The path to the data file or directory.
            time_dif (int, optional): The time difference in hours. Defaults to -6.

        Returns:
            None

        """
        # Resolve the data path and set the time difference
        self.data_path = Path(data_path).resolve()
        self.time_dif = datetime.timedelta(hours=time_dif)

        # Check if the data path is a directory
        if self.data_path.is_dir() == True:
            # Create a directory for saving results
            self.dir_result = self.data_path / "result"
            self.dir_result.mkdir(exist_ok=True, parents=True)

            print("\nDirectory Path:", self.data_path)

            # Get the list of ASC file stems in the directory
            temp_asc_file_stem_list = []
            for temp_asc_file_path_1 in self.data_path.glob("*.asc"):
                temp_asc_file_stem_list.append(temp_asc_file_path_1.stem[:22])

            self.asc_file_stem_list = list(sorted(set(temp_asc_file_stem_list)))

            temp_time_series_data = np.empty((0, 3))

            # Read the data from each ASC file
            for temp_asc_file_stem in self.asc_file_stem_list:

                temp_time_series_data_each = []

            # Read each acceleration component from the ASC file
            for i in range(3):
                temp_asc_file_path_2 = self.data_path / (temp_asc_file_stem + ".0" + str(i + 1) + ".asc")
                temp_acc_comp_each = pd.read_csv(temp_asc_file_path_2, skiprows=8, header=None)
                temp_time_series_data_each.append(temp_acc_comp_each.values.flatten())

            temp_time_series_data_each = np.array(temp_time_series_data_each).T
            temp_time_series_data = np.append(temp_time_series_data, temp_time_series_data_each, axis=0)

            temp_col_names = ["x", "y", "z"]
            temp_time_series_data = pd.DataFrame(temp_time_series_data, columns=temp_col_names)

            temp_initial_time_format = "%Y%m%d%H%M%S"
            self.initial_time = datetime.datetime.strptime(self.asc_file_stem_list[0][:14], temp_initial_time_format) + self.time_dif
            record_duration = (datetime.datetime.strptime(self.asc_file_stem_list[1][:14], temp_initial_time_format) -\
                        datetime.datetime.strptime(self.asc_file_stem_list[0][:14], temp_initial_time_format)) * len(self.asc_file_stem_list)

            temp_time_stamp = pd.date_range(self.initial_time, 
                            self.initial_time + record_duration - datetime.timedelta(seconds=1/self.freq), 
                            freq=datetime.timedelta(seconds=1/self.freq))
            temp_time_stamp = pd.DataFrame({"time":temp_time_stamp})

            self.time_series_data = pd.concat((temp_time_stamp, temp_time_series_data), axis=1)

        else:
            
            print("\nRead:", self.data_path.stem)
            
            # Create a directory for saving results
            self.dir_result = self.data_path.parent / "result" / self.data_path.stem
            self.dir_result.mkdir(exist_ok=True, parents=True)

            temp_col_names = ["time", "x", "y", "z"]
            self.time_series_data = pd.read_csv(self.data_path, header=None, names=temp_col_names)

            self.time_series_data["time"] = pd.to_timedelta(self.time_series_data["time"], unit="s")

            self.asc_file_stem_list = [self.data_path.stem]
            self.initial_time = datetime.datetime.strptime(self.asc_file_stem_list[0][2:16], "%Y%m%d%H%M%S") + self.time_dif

            self.time_series_data["time"] = self.time_series_data["time"] + self.initial_time

        self.col_names = self.time_series_data.columns.values
        self.total_num_data = len(self.time_series_data)
        
                
    
    def export_time_series_record(self, ylim=[-0.001, 0.001]):
        """
        Export the time series record as a plot.

        Args:
            ylim (list, optional): Y-axis limits for the plot. Defaults to [-0.001, 0.001].

        Returns:
            None

        """
        # Set the y-axis limits
        self.ylim = ylim
        
        # Export the time series record plot
        fig, _ = self._export_time_series_record_base()
        
        fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_timeseries.png")
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported time-series record!")
        
        plt.clf()
        plt.close()
        gc.collect()
    
    
    def calcurate_HV_spectrum(self, clip_num_data=16384, cosine_taper=True, is_apply_butterworth_filter=True,
                              is_export_timeseries_butterworth_filter=False, butterworth_order=2,
                              butterworth_lowcut=0.1, butterworth_highcut=20,
                              is_parzen_smoothing=True, parzen_width=0.2, 
                              is_konno_ohmachi_smoothing=False, konno_ohmachi_width=0.2,
                              is_export_figure_3D_running_spectra=False, surface_thin_out_interval=10,
                              is_export_figure_2D_running_spectra=False, 
                              is_export_csv_geomean_spectrum=False,
                              freq_lim=[0.1, 50], HVSR_lim=[0.05, 50]):
        """
        Calculates the HV spectrum.

        Args:
            clip_num_data (int, optional): Number of data points to clip for HV spectrum calculation. Defaults to 16384.
            cosine_taper (bool, optional): Flag for applying cosine taper. Defaults to True.
            is_apply_butterworth_filter (bool, optional): Flag for applying Butterworth filter. Defaults to True.
            is_export_timeseries_butterworth_filter (bool, optional): Flag for exporting time series record with Butterworth filter applied. Defaults to False.
            butterworth_order (int, optional): Order of the Butterworth filter. Defaults to 2.
            butterworth_lowcut (float, optional): Lowcut frequency for the Butterworth filter. Defaults to 0.1.
            butterworth_highcut (float, optional): Highcut frequency for the Butterworth filter. Defaults to 20.
            is_parzen_smoothing (bool, optional): Flag for applying Parzen smoothing. Defaults to True.
            parzen_width (float, optional): Width parameter for Parzen smoothing. Defaults to 0.2.
            is_konno_ohmachi_smoothing (bool, optional): Flag for applying Konno-Ohmachi smoothing. Defaults to False.
            konno_ohmachi_width (float, optional): Width parameter for Konno-Ohmachi smoothing. Defaults to 0.2.
            is_export_figure_3D_running_spectra (bool, optional): Flag for exporting 3D running spectra. Defaults to False.
            surface_thin_out_interval (int, optional): Interval for thinning out the surface in 3D running spectra. Defaults to 10.
            is_export_figure_2D_running_spectra (bool, optional): Flag for exporting 2D running spectra. Defaults to False.
            is_export_csv_geomean_spectrum (bool, optional): Flag for exporting HV spectrum as CSV. Defaults to False.
            freq_lim (list, optional): Frequency limits for HV spectrum. Defaults to [0.1, 50].
            HVSR_lim (list, optional): HVSR limits for HV spectrum. Defaults to [0.05, 50].

        Returns:
            None

        """
        self.clip_num_data = clip_num_data
        
        self.is_apply_butterworth_filter = is_apply_butterworth_filter
        self.is_export_timeseries_butterworth_filter = is_export_timeseries_butterworth_filter
        self.butterworth_order = butterworth_order
        self.butterworth_lowcut = butterworth_lowcut
        self.butterworth_highcut = butterworth_highcut

        self.is_cosine_taper = cosine_taper
        
        self.is_parzen_smoothing = is_parzen_smoothing
        self.parzen_width = parzen_width
        
        self.is_export_figure_3D_running_spectra = is_export_figure_3D_running_spectra
        self.surface_thin_out_interval = surface_thin_out_interval
        
        self.is_export_figure_2D_running_spectra = is_export_figure_2D_running_spectra
        self.is_export_csv_geomean_spectrum = is_export_csv_geomean_spectrum
        self.freq_lim = freq_lim
        self.HVSR_lim = HVSR_lim        
        
        self.is_konno_ohmachi_smoothing = is_konno_ohmachi_smoothing
        self.konno_ohmachi_width = konno_ohmachi_width
        
        # apply butterworth filter if is_export_timeseries_butterworth_filter is True
        if self.is_apply_butterworth_filter:
            temp_b, temp_a = ss.butter(self.butterworth_order, [self.butterworth_lowcut, self.butterworth_highcut], btype="band", fs=self.freq)
            self.time_series_data.iloc[:, 1:] = ss.filtfilt(temp_b, temp_a, self.time_series_data.iloc[:, 1:], axis=0)
        
        if self.is_apply_butterworth_filter and self.is_export_timeseries_butterworth_filter:
            fig, _ = self._export_time_series_record_base()
            
            fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_timeseries_butterworth.png")
            
            fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
            print("Exported time-series (buttterworth) record!")
            
            plt.clf()
            plt.close()
            gc.collect()
        
        temp_last_index = self.total_num_data - self.clip_num_data
        temp_fft_interval = 1 * self.freq
        
        self.frequecy_domain_index = np.arange(0, temp_last_index, temp_fft_interval)
        
        temp_csv_path = self.dir_result / (self.asc_file_stem_list[0] + "_HV_spectrum.csv")
        
        if temp_csv_path.exists():
            self.frequecy_domain_data = pd.read_csv(temp_csv_path)
            print("Loaded HV spectrum!")
        
        else:
            for i in self.frequecy_domain_index:
                
                temp_col_name = "HVSR_power_smoothed_" + "{:07d}".format(i)
                temp_frequecy_domain_data = self._calcurate_HV_spectrum_base(self.time_series_data.iloc[i:self.clip_num_data+i, 1:]).copy()
                
                if i == 0:
                    self.frequecy_domain_data = temp_frequecy_domain_data[["freq", "HVSR_power_smoothed"]]
                
                else:
                    self.frequecy_domain_data = pd.concat((self.frequecy_domain_data, temp_frequecy_domain_data["HVSR_power_smoothed"]), axis=1)
                
                self.frequecy_domain_data.rename(columns={"HVSR_power_smoothed":temp_col_name}, inplace=True)
                
                print("\r", str(i), "/", temp_last_index, end="")
        
            print("")
            print("Calcurated HV spectrum!")
        
        self.max_index_HVSR = self.frequecy_domain_data.columns.values[-1]
        self.max_index_HVSR = int(self.max_index_HVSR.split("_")[-1])
        
        if self.is_export_figure_3D_running_spectra:
            self._export_running_HV_spectra_3D() 
        
        if self.is_export_figure_2D_running_spectra:
            self._export_running_HV_spectra_2D()
                
        if not temp_csv_path.exists():
            self.frequecy_domain_data.to_csv(temp_csv_path, index=False)
            
            print("Saved HV spectrum!")       
                        
        
    def export_HV_spectrum(self, start_indexes = [0, 1], is_only_show_parzen_filter_result = True, ylim=[-0.001, 0.001]):
        
        self.ylim = ylim
        
        self.start_indexes = start_indexes
        self.is_only_show_parzen_filter_result = is_only_show_parzen_filter_result
        
        if self.is_apply_butterworth_filter:
            temp_b, temp_a = ss.butter(self.butterworth_order, [self.butterworth_lowcut, self.butterworth_highcut], btype="band", fs=self.freq)
            self.time_series_data.iloc[:, 1:] = ss.filtfilt(temp_b, temp_a, self.time_series_data.iloc[:, 1:], axis=0)
        
        fig, axes = self._export_time_series_record_base()
        
        # Add rectangular shape to the plot
        for i in range(6):
            
            for j, start_index in enumerate(self.start_indexes):
                
                temp_bottom = axes[i, 0].get_ylim()[0]
                temp_top = axes[i, 0].get_ylim()[1]
                temp_height = temp_top - temp_bottom
                temp_width = self.time_series_data["time"].iloc[start_index + self.clip_num_data] - self.time_series_data["time"].iloc[start_index]
                axes[i, 0].add_patch(patches.Rectangle(xy=(self.time_series_data["time"].iloc[start_index], temp_bottom),
                                                       width=temp_width, height=temp_height, linewidth=0, facecolor="r", alpha=0.2))
                # Add text at the upper left of the rectangular shape with buffer
                axes[i, 0].text(self.time_series_data["time"].iloc[start_index], temp_top, str(j) + "," + str(start_index), 
                                fontsize=4, color="k", verticalalignment="top", horizontalalignment="left", rotation="vertical")
                    
        fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_timeseries_with_clipped_section.png")
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported time-series record with section windows!")
        
        plt.clf()
        plt.close()
        gc.collect()  
                
        temp_freq_at_peaks = self._export_HV_spectrum_base()
        
        return temp_freq_at_peaks
    
    
    def _calcurate_HV_spectrum_base(self, acc_data=None):
        
        temp_fft_freq = np.fft.fftfreq(self.clip_num_data, d=1/self.freq)
        temp_fft_freq = temp_fft_freq[:self.clip_num_data//2]
        
        acc_data = acc_data.values
               
        temp_acc_data = acc_data - np.mean(acc_data)
        temp_len_acc_data = len(temp_acc_data)
        
        # apply 5% cosine taper if is_cosine_taper is True
        # in reality, 5% cosine taper is called as tukey window
        
        if self.is_cosine_taper:
            
            temp_acc_data = temp_acc_data * ss.tukey(temp_len_acc_data, alpha=0.05).reshape(-1, 1)
            
        # Compute power spectrum density
        temp_acc_data_fft = np.fft.fft(temp_acc_data, axis=0)
        temp_acc_data_fft = np.abs(temp_acc_data_fft) ** 2
        temp_acc_data_fft = temp_acc_data_fft[:self.clip_num_data//2]
        temp_acc_data_fft[1:-1] = 2 * temp_acc_data_fft[1:-1]
        
        # Create pandas dataframe with concatanating frequency and power spectrum density
        temp_acc_data_fft_freq = pd.DataFrame({"freq":temp_fft_freq, 
                                                "x_fft_power":temp_acc_data_fft[:, 0], 
                                                "y_fft_power":temp_acc_data_fft[:, 1], 
                                                "z_fft_power":temp_acc_data_fft[:, 2]})
        
        temp_acc_data_fft_freq["h_fft_power"] = np.sqrt(temp_acc_data_fft_freq["x_fft_power"] ** 2 +\
                                                    temp_acc_data_fft_freq["y_fft_power"] ** 2)
        temp_acc_data_fft_freq["v_fft_power"] = np.sqrt(temp_acc_data_fft_freq["z_fft_power"] ** 2)
        temp_acc_data_fft_freq["HVSR_power"] = (temp_acc_data_fft_freq["h_fft_power"] / temp_acc_data_fft_freq["v_fft_power"]) ** (1/2)
        temp_acc_data_fft_freq["HVSR_power_smoothed"] = temp_acc_data_fft_freq["HVSR_power"]
        
        if self.is_parzen_smoothing and self.parzen_width > 0:
            temp_acc_data_fft_freq["HVSR_power_smoothed"] = 0

            # 要修正：parzen windowの適用が正しくない．freq=0の値が他の値に影響を与えている．単純に平滑化するだけであればfreq=0の値は他の周波数の振幅に影響を与えるべきではない．
            # 簡易修正：freq=0の値を除外して平滑化する．
            
            temp_acc_data_fft_freq_HVSR_power = temp_acc_data_fft_freq["HVSR_power"].copy()
            temp_acc_data_fft_freq_HVSR_power.iloc[0] = 0
            temp_freq_interval = temp_acc_data_fft_freq.iloc[1, 0] - temp_acc_data_fft_freq.iloc[0, 0]
            
            temp_acc_data_fft_freq["HVSR_power_smoothed"]= temp_acc_data_fft_freq_HVSR_power.rolling(window=int(self.parzen_width / temp_freq_interval), center=True, win_type="parzen").mean()

        elif self.is_konno_ohmachi_smoothing:
            
            # 要修正：konno-ohmachi windowの適用ができていない→計算量が多い可能性あり
            pass
                    
           
        temp_acc_data_fft_freq["HVSR_power_smoothed"] = temp_acc_data_fft_freq["HVSR_power_smoothed"].fillna(method="bfill")
        temp_acc_data_fft_freq["HVSR_power_smoothed"] = temp_acc_data_fft_freq["HVSR_power_smoothed"].fillna(method="ffill")
        
        return temp_acc_data_fft_freq
            
    
    def _export_time_series_record_base(self):
        
        fig, axes = setup_figure(num_row=6, height=8, width=10, hspace=.125)
        
        axes[0, 0].plot(self.time_series_data["time"], self.time_series_data["x"], "k", linewidth=0.5)
        axes[1, 0].plot(self.time_series_data["time"], self.time_series_data["y"], "k", linewidth=0.5)
        axes[2, 0].plot(self.time_series_data["time"], self.time_series_data["z"], "k", linewidth=0.5)
        axes[3, 0].plot(self.time_series_data["time"], self.time_series_data["x"], "k", linewidth=0.5)
        axes[4, 0].plot(self.time_series_data["time"], self.time_series_data["y"], "k", linewidth=0.5)
        axes[5, 0].plot(self.time_series_data["time"], self.time_series_data["z"], "k", linewidth=0.5)
        
        for i in range(6):
            axes[i, 0].set_xlim(self.time_series_data["time"].iloc[0], self.time_series_data["time"].iloc[-1])            
            axes[i, 0].spines["top"].set_visible(False)
            axes[i, 0].spines["bottom"].set_linewidth(0.5)
            axes[i, 0].spines["right"].set_visible(False)
            axes[i, 0].spines["left"].set_linewidth(0.5)
            axes[i, 0].xaxis.set_tick_params(width=0.5)
            axes[i, 0].yaxis.set_tick_params(width=0.5)
            axes[i, 0].set_ylabel(self.col_names[i % 3 + 1] + " vel. (cm/s)")
            
            if i < 3:
                axes[i, 0].set_ylim(self.ylim)
            
            if i == 5:
                axes[i, 0].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
                axes[i, 0].xaxis.set_minor_locator(mdates.SecondLocator(bysecond=range(0, 60, 6)))
                axes[i, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                axes[i, 0].tick_params(axis="x", which="major", labelsize=8)
                axes[i, 0].set_xlabel("Time", fontsize=8)
                
                sub_axes_1 = axes[i, 0].twiny()
                sub_axes_1.spines["top"].set_visible(False)
                sub_axes_1.spines["right"].set_visible(False)
                sub_axes_1.spines["left"].set_visible(False)
                sub_axes_1.spines["bottom"].set_position(("outward", 28))
                sub_axes_1.spines["bottom"].set_linewidth(0.5)
                sub_axes_1.xaxis.set_ticks_position("bottom")
                sub_axes_1.xaxis.set_label_position("bottom")
                sub_axes_1.xaxis.set_tick_params(width=0.5)
                sub_axes_1_majorlabel = np.arange(0, self.total_num_data + 1, 5000, dtype=int)
                sub_axes_1_minorlabel = np.arange(0, self.total_num_data + 1, 1000, dtype=int)
                sub_axes_1.set_xticks(sub_axes_1_majorlabel)
                sub_axes_1.set_xticks(sub_axes_1_minorlabel, minor="True")
                sub_axes_1.set_xticklabels(sub_axes_1_majorlabel, fontsize=8, rotation=90)
                sub_axes_1.set_xlim(0, self.total_num_data)             
                sub_axes_1.set_xlabel("Start Index Number", fontsize=8)
                
                sub_axes_2 = axes[i, 0].twiny()
                sub_axes_2.spines["top"].set_visible(False)
                sub_axes_2.spines["right"].set_visible(False)
                sub_axes_2.spines["left"].set_visible(False)
                sub_axes_2.spines["bottom"].set_position(("outward", 84))
                sub_axes_2.spines["bottom"].set_linewidth(0.5)
                sub_axes_2.xaxis.set_ticks_position("bottom")
                sub_axes_2.xaxis.set_label_position("bottom")
                sub_axes_2.xaxis.set_tick_params(width=0.5)
                sub_axes_2_majorlabel = np.arange(0, self.total_num_data + 1, 5000, dtype=int) - self.clip_num_data
                sub_axes_2_majorlabel[sub_axes_2_majorlabel < 0] = 0
                sub_axes_2.set_xticks(sub_axes_1_majorlabel)
                sub_axes_2.set_xticks(sub_axes_1_minorlabel, minor="True")
                sub_axes_2.set_xticklabels(sub_axes_2_majorlabel, fontsize=8, rotation=90)
                sub_axes_2.set_xlim(0, self.total_num_data)             
                sub_axes_2.set_xlabel("End Index Number", fontsize=8)
                  
            else:
                axes[i, 0].xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
                axes[i, 0].xaxis.set_minor_locator(mdates.SecondLocator(bysecond=range(0, 60, 6)))
                axes[i, 0].xaxis.set_ticklabels([])
        
        return fig, axes

    def _export_HV_spectrum_base(self):
    
        if self.is_only_show_parzen_filter_result:
            fig, axes = setup_figure()
            
            temp_section_col_name = ["HVSR_power_smoothed_" + "{:07d}".format(i) for i in self.start_indexes]
            
            for temp_section_col_name_each in temp_section_col_name:
                axes[0, 0].loglog(self.frequecy_domain_data["freq"], self.frequecy_domain_data[temp_section_col_name_each], "k", linewidth=0.5, alpha=0.25)
            
            geomean_HVSR = np.prod(self.frequecy_domain_data[temp_section_col_name], axis=1) ** (1/len(self.start_indexes))
            self.frequecy_domain_data["geomean_HVSR"] = geomean_HVSR
            
            axes[0, 0].loglog(self.frequecy_domain_data["freq"], geomean_HVSR, "r", linewidth=1, alpha=0.875)
                        
            # find the peaks of the HVSR
            geomean_HVSR_peaks, geomean_HVSR_properties = ss.find_peaks(geomean_HVSR[self.frequecy_domain_data["freq"] < 10], prominence=1)
            
            # sort the peaks by prominence
            geomean_HVSR_peaks = geomean_HVSR_peaks[np.argsort(geomean_HVSR_properties["prominences"])[::-1]]
            
            # limit the number of peaks to 4
            if len(geomean_HVSR_peaks) > 4:
                geomean_HVSR_peaks = geomean_HVSR_peaks[:4]
                
            axes[0, 0].vlines(x=self.frequecy_domain_data["freq"].iloc[geomean_HVSR_peaks], ymin=self.HVSR_lim[0], ymax=self.HVSR_lim[1], color="r", linewidth=0.5, linestyle="--")
            
            # annotate the peaks with their frequencies
            for i, temp_peak in enumerate(geomean_HVSR_peaks):
                axes[0, 0].annotate("#"+str(i+1)+", "+str(round(self.frequecy_domain_data["freq"].iloc[temp_peak], 2)), 
                                    (self.frequecy_domain_data["freq"].iloc[temp_peak], geomean_HVSR[temp_peak]), 
                                    textcoords='data', 
                                    xytext=(self.frequecy_domain_data["freq"].iloc[temp_peak], self.HVSR_lim[1]*10**-0.05), ha="right", 
                                    va="top", fontsize=6, rotation=90)
            
            axes[0, 0].set_xlim(self.freq_lim[0], self.freq_lim[1])
            axes[0, 0].set_ylim(self.HVSR_lim[0], self.HVSR_lim[1])
            axes[0, 0].spines["top"].set_linewidth(0.5)
            axes[0, 0].spines["bottom"].set_linewidth(0.5)
            axes[0, 0].spines["right"].set_linewidth(0.5)
            axes[0, 0].spines["left"].set_linewidth(0.5)
            axes[0, 0].xaxis.set_tick_params(width=0.5)
            axes[0, 0].yaxis.set_tick_params(width=0.5)
            axes[0, 0].set_ylabel("HVSR")
            axes[0, 0].set_xlabel("Frequency (Hz)")
            axes[0, 0].grid(which="major", linestyle="-", linewidth=0.25)
            
        else:
            # Under maintenance. The code will not be working properly.
            fig, axes = setup_figure(num_col=2, width=10)
            
            for j in range(2):
                for i in range(len(self.start_indexes)):
                    if j == 0:
                        axes[0, j].loglog(self.frequecy_domain_data[i]["freq"], self.frequecy_domain_data[i]["HVSR_power"], "k", linewidth=0.5)
                    elif j == 1:             
                        axes[0, j].loglog(self.frequecy_domain_data[i]["freq"], self.frequecy_domain_data[i]["HVSR_power_smoothed"], "k", linewidth=0.5)
                        
                axes[0, j].set_xlim(0.1, 50)
                axes[0, j].set_ylim(0.1, 100)
                axes[0, j].spines["top"].set_linewidth(0.5)
                axes[0, j].spines["bottom"].set_linewidth(0.5)
                axes[0, j].spines["right"].set_linewidth(0.5)
                axes[0, j].spines["left"].set_linewidth(0.5)
                axes[0, j].xaxis.set_tick_params(width=0.5)
                axes[0, j].yaxis.set_tick_params(width=0.5)
                axes[0, j].set_ylabel("HVSR")
                axes[0, j].set_xlabel("Frequency (Hz)")
                    
        fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_HV_spectrum.png")
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported HV spectrum!")
        
        plt.clf()
        plt.close()
        gc.collect()
        
        if self.is_export_csv_geomean_spectrum:
            temp_geomean_result = self.frequecy_domain_data[["freq", "geomean_HVSR"]].copy()
            temp_geomean_csv_path = self.dir_result / (self.asc_file_stem_list[0] + "_geomean_HV_spectrum.csv")
            temp_geomean_result.to_csv(temp_geomean_csv_path, index=False)
            
            print("Exported geomean HV spectrum file!")
            
        
        return self.frequecy_domain_data["freq"].iloc[geomean_HVSR_peaks].values
    
    
    def _export_running_HV_spectra_3D(self):
                
        temp_frequecy_domain_data = self.frequecy_domain_data.copy()
        y = temp_frequecy_domain_data["freq"][1:]
        
        temp_frequecy_domain_data = temp_frequecy_domain_data.iloc[1:, 1::self.surface_thin_out_interval]
        temp_frequecy_domain_data_columns = temp_frequecy_domain_data.columns.values
        
        temp_x_ticktext = [int(temp_frequecy_domain_data_columns[i][-7:]) for i in range(len(temp_frequecy_domain_data_columns))]
        temp_zmin_log10 = math.log10(self.HVSR_lim[0])
        temp_zmax_log10 = math.log10(self.HVSR_lim[1])
        temp_z_tickvals = np.linspace(temp_zmin_log10, temp_zmax_log10, 5)
        temp_z_ticktext = np.round(np.logspace(temp_zmin_log10, temp_zmax_log10, 5), 2)
        temp_title_text = "File: " + self.asc_file_stem_list[0] + " Running HV Spectra (3D)" + \
                        "<br>Frequency Range: " + str(self.freq_lim[0]) + " - " + str(self.freq_lim[1]) + " Hz" + \
                        "<br>HVSR Range: " + str(self.HVSR_lim[0]) + " - " + str(self.HVSR_lim[1]) + \
                        "<br>Window Samples: " + str(self.clip_num_data) + \
                        "<br>Total Samples: " + str(self.total_num_data)
                            
        x = np.arange(0, len(temp_frequecy_domain_data_columns))
        z = np.log10(temp_frequecy_domain_data.values)
        
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, cmin=temp_zmin_log10, cmax=temp_zmax_log10)])
        
        fig.update_layout(scene=dict(
            xaxis = dict(title="Index",
                         tickmode="array", 
                         tickvals=np.arange(0, len(temp_frequecy_domain_data_columns), 1), 
                         ticktext=temp_x_ticktext,
                         autorange="reversed"),
            yaxis = dict(title="Frequency (Hz)",
                         type="log"),
            zaxis = dict(title="HVSR",
                         range=[temp_zmin_log10, temp_zmax_log10],
                         tickmode="array",
                         tickvals=temp_z_tickvals,
                         ticktext=temp_z_ticktext),
            aspectratio=dict(x=2, y=1, z=0.5),
            ),
            title = go.layout.Title(text=temp_title_text)
        )
        
        fig.update_layout(coloraxis=dict(
            colorbar=dict(
            title="HVSR",
            tickmode="array",
            tickvals=temp_z_tickvals,
            ticktext=temp_z_ticktext
            )
        ))
                
        # save the figure as an interactive plot
        fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_running_HV_spectra_3D.html")
        fig.write_html(fig_name)
        
        print("Exported running HV spectra (3D)!")
        
    
    def _export_running_HV_spectra_2D(self):
        
        temp_frequecy_domain_data = self.frequecy_domain_data.copy()
        y = temp_frequecy_domain_data["freq"][1:]
        
        temp_frequecy_domain_data = temp_frequecy_domain_data.iloc[1:, 1:]
        temp_x_col_name = temp_frequecy_domain_data.columns.values
        
        x = np.array([int(temp_x_col_name[i][-7:]) for i in range(len(temp_x_col_name))]) + self.clip_num_data // 2
        z = np.log10(temp_frequecy_domain_data.values)
        
        x, y = np.meshgrid(x, y)
        
        fig, axes = setup_figure(num_row=1, num_col=1, width=10)
        
        temp_zmin_log10 = math.log10(self.HVSR_lim[0])
        temp_zmax_log10 = math.log10(self.HVSR_lim[1])
        norm = mcolors.Normalize(vmin=temp_zmin_log10, vmax=temp_zmax_log10)
        temp_axes = axes[0, 0].pcolormesh(x, y, z, cmap="jet", shading="auto", norm=norm)
        temp_axes_colorbar = fig.colorbar(temp_axes, ax=axes[0, 0], norm=norm)
        
        axes[0, 0].set_xlabel("Index")
        axes[0, 0].set_xlim(0, self.time_series_data.shape[0] - 1)
        
        axes[0, 0].xaxis.set_tick_params(rotation=90)
        
        axes[0, 0].set_ylabel("Frequency (Hz)")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_ylim(self.freq_lim[0], self.freq_lim[1])
        
        temp_axes_colorbar.ax.set_ylabel("log10(HVSR)", rotation=90)
                        
        fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_running_HV_spectra_2D.png")
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        
        print("Exported running HV spectra (2D)!")
        
        plt.clf()
        plt.close()
        gc.collect()  
        
        
                