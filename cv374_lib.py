from pathlib import Path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import gc
import scipy.signal as ss
import matplotlib.dates as mdates
import matplotlib.patches as patches



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
               

class CV374Data:
    """
    Class for reading and processing CV374 data.
    
    data_path: str
        Directory path of CV374 data.
    time_dif: int
        Time difference between UTC and local time.
    
    """
    def __init__(self) -> None:
        self.freq = 100
        self.ylim = [-0.01, 0.01]
        self.start_indexes = [0, 5000, 10000]
        self.clip_num_data = 16384
        self.parzen_width = 0.2
        self.data_path = ""
        self.dir_result = ""
        self.time_dif = 0
        
        self.is_cosine_taper = True
        self.is_parzen_smoothing = False
        self.is_only_show_parzen_filter_result = True
    
    def read_data(self, data_path, time_dif = -6) -> None:
        
        self.data_path = Path(data_path).resolve()
        self.time_dif = datetime.timedelta(hours=time_dif)
        
        
        if self.data_path.is_dir() == True:
            self.dir_result = self.data_path / "result"
            self.dir_result.mkdir(exist_ok=True, parents=True)
            
            print("Directory Path:", self.data_path)
            
            temp_asc_file_stem_list = []
            for temp_asc_file_path_1 in self.data_path.glob("*.asc"):
                temp_asc_file_stem_list.append(temp_asc_file_path_1.stem[:22])
            
            self.asc_file_stem_list = list(sorted(set(temp_asc_file_stem_list)))
            
            temp_time_series_data = np.empty((0, 3))
                    
            for temp_asc_file_stem in self.asc_file_stem_list:
                
                temp_time_series_data_each = []
                
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
            self.dir_result = self.data_path.parent / "result"
            self.dir_result.mkdir(exist_ok=True, parents=True)
            
            temp_col_names = ["time", "x", "y", "z"]
            self.time_series_data = pd.read_csv(self.data_path, header=None, names=temp_col_names)
            
            self.time_series_data["time"] = pd.to_timedelta(self.time_series_data["time"], unit="s")

            self.asc_file_stem_list = [self.data_path.stem]
            self.initial_time = datetime.datetime.strptime(self.asc_file_stem_list[0][2:16], "%Y%m%d%H%M%S") + self.time_dif
            
            self.time_series_data["time"] = self.time_series_data["time"] + self.initial_time
            
        self.col_names = self.time_series_data.columns.values
                
    
    def export_time_series_record(self, ylim=[-0.001, 0.001]):
        
        self.ylim = ylim
        fig, _ = self._export_time_series_record_base()
        
        fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_timeseries.png")
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported time-series record!")
        
        plt.clf()
        plt.close()
        gc.collect()
    
    def calcurate_HV_spectrum(self, clip_num_data = 16384, cosine_taper = True, 
                              is_butterworth_filter = False, butterworth_order = 2,
                              butterworth_lowcut = 0.1, butterworth_highcut = 20,
                              is_parzen_smoothing = False, parzen_width = 0.2, 
                              is_konno_ohmachi_smoothing = True, konno_ohmachi_width = 0.2):
        
        self.clip_num_data = clip_num_data
        
        self.is_butterworth_filter = is_butterworth_filter
        self.butterworth_order = butterworth_order
        self.butterworth_lowcut = butterworth_lowcut
        self.butterworth_highcut = butterworth_highcut

        self.is_cosine_taper = cosine_taper
        
        self.is_parzen_smoothing = is_parzen_smoothing
        self.parzen_width = parzen_width
        
        self.is_konno_ohmachi_smoothing = is_konno_ohmachi_smoothing
        self.konno_ohmachi_width = konno_ohmachi_width
        
        # apply butterworth filter if is_butterworth_filter is True
        if self.is_butterworth_filter:
            temp_b, temp_a = ss.butter(self.butterworth_order, [self.butterworth_lowcut, self.butterworth_highcut], btype="band", fs=self.freq)
            self.time_series_data.iloc[:, 1:] = ss.filtfilt(temp_b, temp_a, self.time_series_data.iloc[:, 1:], axis=0)
            
            fig, _ = self._export_time_series_record_base()
            
            fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_timeseries_butterworth.png")
            
            fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
            print("Exported time-series (buttterworth) record!")
            
            plt.clf()
            plt.close()
            gc.collect()
        
        
        self._calcurate_HV_spectrum_base(self.time_series_data.iloc[:self.clip_num_data, 1:])
        
        
    def export_HV_spectrum(self, start_indexes = [0, 5000, 10000], is_only_show_parzen_filter_result = True):
        
        self.start_indexes = start_indexes

        self.is_only_show_parzen_filter_result = is_only_show_parzen_filter_result
        
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
                axes[i, 0].text(self.time_series_data["time"].iloc[start_index], temp_top, str(j) + "," + str(start_index), fontsize=6, color="k", verticalalignment="top", horizontalalignment="left", rotation="vertical")
                    
        fig_name = self.dir_result / (self.asc_file_stem_list[0] + "_timeseries_with_clipped_section.png")
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported time-series record!")
        
        plt.clf()
        plt.close()
        gc.collect()  
        
        self.frequecy_domain_data, self.geomean_frequecy_domain_data = self._calcurate_HV_spectrum()
        
        self._export_HV_spectrum_base()
    
    
    def _calcurate_HV_spectrum_base(self, acc_data=None):
        
        temp_fft_freq = np.fft.fftfreq(self.clip_num_data, d=1/self.freq)
        temp_fft_freq = temp_fft_freq[:self.clip_num_data//2]
        
        acc_data = acc_data.values
        
        print(acc_data)
        
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
        
    def _calcurate_HV_spectrum(self):
        
        HV_spectrum_data = []
        
        for start_index in self.start_indexes:
            temp_time_data = self.time_series_data["time"].iloc[start_index:start_index + self.clip_num_data]
            temp_fft_freq = np.fft.fftfreq(self.clip_num_data, d=1/self.freq)
            temp_fft_freq = temp_fft_freq[:self.clip_num_data//2]
            
            temp_acc_data = self.time_series_data.iloc[start_index:start_index + self.clip_num_data, 1:]
            temp_acc_data = temp_acc_data - np.mean(temp_acc_data)
            
            if self.is_cosine_taper:
                temp_acc_data = temp_acc_data * ss.parzen(self.clip_num_data).reshape(-1, 1)
            
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
            
            if self.parzen_width > 0:
                temp_acc_data_fft_freq["HVSR_power_smoothed"] = 0

                # 要修正：parzen windowの適用が正しくない．freq=0の値が他の値に影響を与えている．単純に平滑化するだけであればfreq=0の値は他の周波数の振幅に影響を与えるべきではない．
                # 簡易修正：freq=0の値を除外して平滑化する．
                
                
                temp_acc_data_fft_freq_HVSR_power = temp_acc_data_fft_freq["HVSR_power"].copy()
                temp_acc_data_fft_freq_HVSR_power.iloc[0] = 0
                temp_freq_interval = temp_acc_data_fft_freq.iloc[1, 0] - temp_acc_data_fft_freq.iloc[0, 0]
                
                temp_acc_data_fft_freq["HVSR_power_smoothed"]= temp_acc_data_fft_freq_HVSR_power.rolling(window=int(self.parzen_width / temp_freq_interval), center=True, win_type="parzen").mean()

            temp_acc_data_fft_freq["HVSR_power_smoothed"] = temp_acc_data_fft_freq["HVSR_power_smoothed"].fillna(method="bfill")
            temp_acc_data_fft_freq["HVSR_power_smoothed"] = temp_acc_data_fft_freq["HVSR_power_smoothed"].fillna(method="ffill")
            
            HV_spectrum_data.append(temp_acc_data_fft_freq)
        
        # calculate the geometric mean of HVSR_power_smoothed
        temp_HVSR_power_smoothed_list = []
        for i, start_index in enumerate(self.start_indexes):
            temp_HVSR_power_smoothed_list.append(HV_spectrum_data[i]["HVSR_power_smoothed"])
        
        temp_HVSR_power_smoothed_list = np.array(temp_HVSR_power_smoothed_list)
        temp_geomean_HVSR_power_smoothed = np.exp(np.mean(np.log(temp_HVSR_power_smoothed_list), axis=0))
        
        HV_spectrum_data_geomean = pd.DataFrame({"freq":temp_acc_data_fft_freq["freq"], 
                                                "geomean_HVSR_power_smoothed":temp_geomean_HVSR_power_smoothed})
        
        return (HV_spectrum_data, HV_spectrum_data_geomean)
    
    
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
                sub_axes_1_majorlabel = np.arange(0, len(self.time_series_data) + 1, 5000, dtype=int)
                sub_axes_1_minorlabel = np.arange(0, len(self.time_series_data) + 1, 1000, dtype=int)
                sub_axes_1.set_xticks(sub_axes_1_majorlabel)
                sub_axes_1.set_xticks(sub_axes_1_minorlabel, minor="True")
                sub_axes_1.set_xticklabels(sub_axes_1_majorlabel, fontsize=8, rotation=90)
                sub_axes_1.set_xlim(0, len(self.time_series_data))             
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
                sub_axes_2_majorlabel = np.arange(0, len(self.time_series_data) + 1, 5000, dtype=int) - self.clip_num_data
                sub_axes_2_majorlabel[sub_axes_2_majorlabel < 0] = 0
                sub_axes_2.set_xticks(sub_axes_1_majorlabel)
                sub_axes_2.set_xticks(sub_axes_1_minorlabel, minor="True")
                sub_axes_2.set_xticklabels(sub_axes_2_majorlabel, fontsize=8, rotation=90)
                sub_axes_2.set_xlim(0, len(self.time_series_data))             
                sub_axes_2.set_xlabel("End Index Number", fontsize=8)
                  
            else:
                axes[i, 0].xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
                axes[i, 0].xaxis.set_minor_locator(mdates.SecondLocator(bysecond=range(0, 60, 6)))
                axes[i, 0].xaxis.set_ticklabels([])
        
        return fig, axes

    def _export_HV_spectrum_base(self):
    
        if self.is_only_show_parzen_filter_result:
            fig, axes = setup_figure()
            
            
            for i in range(len(self.start_indexes)):
                axes[0, 0].loglog(self.frequecy_domain_data[i]["freq"], self.frequecy_domain_data[i]["HVSR_power_smoothed"], "k", linewidth=0.5, alpha=0.25)
            
            axes[0, 0].loglog(self.geomean_frequecy_domain_data["freq"], self.geomean_frequecy_domain_data["geomean_HVSR_power_smoothed"], "r", linewidth=1, alpha=0.875)
            
            axes[0, 0].set_xlim(0.1, 50)
            axes[0, 0].set_ylim(0.1, 100)
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