import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime as dt
import scipy 

class scintdata():

    def __init__(self, h5_file, timezone_hrs=5, timezone_mins=30): 
        self.timezone_offset = timezone_hrs + timezone_mins/60.0 #5.5 for IST
        self.timezone_dt = dt.timezone(dt.timedelta(hours=timezone_hrs, minutes=timezone_mins))
        self.h5_file = h5_file

        hf = h5py.File(self.h5_file, 'r')

        self.Nchan = int(hf["data"].attrs['NFFT'])
        self.Ntimes = int(hf["data"].attrs['Ntimes'])
        self.f_low = float(hf["data"].attrs['f_low'])
        self.f_high = float(hf["data"].attrs['f_high'])
        self.delta_f = float(hf["data"].attrs['delta_f'])
        self.t_aver = float(hf["data"].attrs['t_aver'])
        self.t_scan = float(hf["data"].attrs['t_scan'])
        self.tstart_unix_time = float(hf["data"].attrs['t_start'])
        self.tend_unix_time = float(hf["data"].attrs['t_end'])
        self.freq = np.linspace(self.f_low, self.f_high, self.Nchan)/1e6
        self.timestamps = hf["data/timestamps"][()]
        self.radio_data = hf["data/radio"][()]*1e6

        self.times = np.linspace(self.tstart_unix_time, self.tend_unix_time, self.Ntimes)
        hf.close()        

        self.starttime = int(self.h5_file.split("-")[1][0:2])+float(int(self.h5_file.split("-")[1][2:4])/60.0) + self.timezone_offset 
        self.obs_date = str(dt.datetime.fromtimestamp(self.timestamps[0], self.timezone_dt).strftime('%Y-%b-%d %H:%M:%S').split(" ")[0])
        self.hrs_full = (self.starttime+((self.timestamps-self.timestamps[0])/(60*60)))

        self.ref_chans_1_a    = range(153,155)
        self.satdlink_chans_1 = range(155,166)
        self.ref_chans_1_b    = range(166,169)

        self.ref_chans_2_a    = range(177,181)
        self.satdlink_chans_2 = range(181,193)
        self.ref_chans_2_b    = range(193,200)

        print ("Data loaded ..")

    def plot_wfall(self, save=True, show=False):

        first_hr = int(self.hrs_full[0])+1
        first_hr_index = np.where(np.isclose(self.hrs_full, first_hr))[0][0]
        ytick_indices = np.arange(first_hr_index, len(self.timestamps), 5978)
        ylabels = [(str(int(self.hrs_full[ytick_index]/24))+"d"+str(int(self.hrs_full[ytick_index]%24)).zfill(2)+"h") for ytick_index in ytick_indices]

        xtick_indices = np.append(np.arange(0, self.Nchan, int(self.Nchan/4)), int(self.Nchan-1))
        xlabels = [("{:.2f}".format(self.freq[xtick_index])) for xtick_index in xtick_indices]

        plt.figure(figsize=(10,10))
        plt.title(self.obs_date+" Dynamic Spectra")
        plt.imshow(np.log(self.radio_data), cmap='jet', aspect='auto')
        plt.yticks(ytick_indices, ylabels)
        plt.xticks(xtick_indices, xlabels)
        plt.xlabel("Freq (MHz)")
        plt.ylabel("Local Time (IST)")
        if save:
            plt.savefig(self.obs_date+"_waterfall.pdf", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def RFIsub(self):
        self.freq_1 = np.mean(self.freq[self.satdlink_chans_1])
        self.freq_2 = np.mean(self.freq[self.satdlink_chans_2])

        self.satdlink_raw_pow_c1 = np.mean(self.radio_data[:, self.satdlink_chans_1], axis=1)
        self.satdlink_raw_pow_c2 = np.mean(self.radio_data[:, self.satdlink_chans_2], axis=1)

        self.ref_pow_1 = (np.mean(self.radio_data[:,self.ref_chans_1_a], axis=1) + np.mean(self.radio_data[:,self.ref_chans_1_b], axis=1))/2
        self.ref_pow_2 = (np.mean(self.radio_data[:,self.ref_chans_2_a], axis=1) + np.mean(self.radio_data[:,self.ref_chans_2_b], axis=1))/2

        self.satdlink_proc_pow_c1 = self.satdlink_raw_pow_c1-(self.ref_pow_1)
        self.satdlink_proc_pow_c2 = self.satdlink_raw_pow_c2-(self.ref_pow_2)
        print ("RFI subtracted ..")

    def get_indices(self, hr_low, hr_high):
        indices = np.where((self.hrs_full>hr_low) & (self.hrs_full<hr_high))[0]
        return indices
    
    def calc_s4(self, indices, channel='a', av_tsec=60):

        s4_chunksize = int(av_tsec/((self.hrs_full[1]-self.hrs_full[0])*60*60))
        s4_index = np.array([])
        if channel=='a':
            analysis_data = self.satdlink_proc_pow_c1
            print ("Calculating S4 for freq : {:.2f} MHz".format(self.freq_1))
        else:
            analysis_data = self.satdlink_proc_pow_c2
            print ("Calculating S4 for freq : {:.2f} MHz".format(self.freq_2))

        for i in range(int(len(indices))):
            analysis_chunk = analysis_data[indices][i: i+s4_chunksize]
            s4_val = np.std(analysis_chunk)/np.mean(analysis_chunk)
            s4_index = np.append(s4_index, s4_val)
        return s4_index
    
    def calc_welch_fit(self, indices, NFT, Nperseg, f_break_low, f_break_high=np.inf, detrend=False, channel='a', window='flattop'):

        if channel=='a':
            analysis_data = self.satdlink_proc_pow_c1
            print ("Calculating Periodogram for channel a")
        else:
            analysis_data = self.satdlink_proc_pow_c2
            print ("Calculating Periodogram for channel b")
        
        t = (self.hrs_full[0:Nperseg]-self.hrs_full[0])*60*60
        f, Pxx_den = scipy.signal.welch(analysis_data[indices], nfft=NFT, fs=1/(t[1]-t[0]), nperseg=Nperseg, window=window, detrend=detrend)
        psd_freq_chans = np.where((f>f_break_low) & (f<f_break_high))[0]

        f_psd = f[psd_freq_chans]
        pfit_c = np.polyfit(np.log10(f_psd), np.log10(Pxx_den[psd_freq_chans]), 1) #np.log
        pval = 10**np.polyval(pfit_c, (np.log10(f_psd)))

        return f, Pxx_den, f_psd, pval, pfit_c