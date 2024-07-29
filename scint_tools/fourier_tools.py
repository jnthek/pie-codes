import numpy as np
import scipy

def get_fft_fit(analysis_data, hrs_full, hr_low, hr_high, f_break_low, NFT, f_break_high=np.inf, Nperseg=512, detrend=False):

    data_index = np.where((hrs_full>hr_low) & (hrs_full<hr_high))[0] # for 2023-01-04 data

    Nseg = int(len(data_index)/Nperseg)
    Pxx = np.zeros([Nseg, int(NFT/2)+1])
    if detrend:
        for i in range(Nseg):
            data_chunk = analysis_data[data_index][i*Nperseg: (i+1)*Nperseg]
            data_chunk = data_chunk - np.mean(data_chunk)
            Pxx[i,:] = np.abs(np.fft.rfft(data_chunk, NFT))**2
    else:
        for i in range(Nseg):
            data_chunk = analysis_data[data_index][i*Nperseg: (i+1)*Nperseg]
            Pxx[i,:] = np.abs(np.fft.rfft(data_chunk, NFT))**2
        

    samp_freq = (hrs_full[1]-hrs_full[0])*60*60
    # f = np.fft.rfftfreq(NFT, 1/samp_freq)
    f = np.fft.rfftfreq(NFT, samp_freq)
    psd_freq_chans = np.where((f>f_break_low) & (f<f_break_high))[0]

    f_psd = f[psd_freq_chans]
    pfit_c = np.polyfit(np.log10(f_psd), np.log10(np.mean(Pxx, axis=0)[psd_freq_chans]), 1) #np.log
    pval = 10**np.polyval(pfit_c, (np.log10(f_psd)))

    return data_index, f, Pxx, f_psd, pval, pfit_c

def get_welch_fit(analysis_data, hrs_full, hr_low, hr_high, NFT, Nperseg,f_break_low, f_break_high=np.inf, detrend=False, window='flattop'):

    data_index = np.where((hrs_full>hr_low) & (hrs_full<hr_high))[0] 
    print (len(data_index))
    t = (hrs_full[0:Nperseg]-hrs_full[0])*60*60
    f, Pxx_den = scipy.signal.welch(analysis_data[data_index], nfft=NFT, fs=1/(t[1]-t[0]), nperseg=Nperseg, window=window, detrend=detrend)
    psd_freq_chans = np.where((f>f_break_low) & (f<f_break_high))[0]

    f_psd = f[psd_freq_chans]
    pfit_c = np.polyfit(np.log10(f_psd), np.log10(Pxx_den[psd_freq_chans]), 1) #np.log
    pval = 10**np.polyval(pfit_c, (np.log10(f_psd)))

    return data_index, f, Pxx_den, f_psd, pval, pfit_c

def get_s4(analysis_data, hrs_full, hr_low, hr_high, av_tsec=60):
    zoom_index = np.where((hrs_full>hr_low) & (hrs_full<hr_high))[0]

    s4_chunksize = int(av_tsec/((hrs_full[1]-hrs_full[0])*60*60))
    s4_index = np.array([])
    for i in range(int(len(zoom_index))):
        analysis_chunk = analysis_data[zoom_index][i: i+s4_chunksize]
        s4_val = np.std(analysis_chunk)/np.mean(analysis_chunk)
        s4_index = np.append(s4_index, s4_val)
    return zoom_index, s4_index

def get_spec_chunked(analysis_data, hrs_full, hr_low, hr_high, NFT, Nperseg, xform="F", window='flattop', detrend=False):

    data_index = np.where((hrs_full>hr_low) & (hrs_full<hr_high))[0] 
    Nseg = int(len(data_index)/Nperseg)
    Pxx = np.zeros([Nseg, NFT])

    t = (hrs_full[0:Nperseg]-hrs_full[0])*60*60
    f = np.linspace(0,0.5/(t[1]-t[0]),NFT)
    xform_kernel = np.zeros([NFT, Nperseg], dtype=np.complex64)

    window_func = scipy.signal.get_window(window, Nperseg)

    if xform=="B":
        for i in range(NFT):
            xform_kernel[i,:] = scipy.special.jv(0, 2*np.pi*f[i]*t)
    else:
        for i in range(NFT):
            xform_kernel[i,:] = np.exp(2j*np.pi*t*f[i])

    if detrend:
        for i in range(Nseg):
            data_chunk = analysis_data[data_index][i*Nperseg: (i+1)*Nperseg]
            data_chunk = data_chunk - np.mean(data_chunk)
            Pxx[i,:] = np.abs(np.einsum("ft,t,t->f", xform_kernel, window_func, data_chunk, optimize="optimal"))**2
    else:
        for i in range(Nseg):
            data_chunk = analysis_data[data_index][i*Nperseg: (i+1)*Nperseg]
            Pxx[i,:] = np.abs(np.einsum("ft,t->f", xform_kernel, data_chunk, optimize="optimal"))**2

    return f, Pxx

def get_spec_full(analysis_data, hrs_full, hr_low, hr_high, NFT, xform="F", window='flattop', detrend=False):

    data_index = np.where((hrs_full>hr_low) & (hrs_full<hr_high))[0] 
    mean_Pxx = np.zeros([NFT])

    t = (hrs_full[data_index]-hrs_full[data_index][0])*60*60
    f = np.linspace(0,0.5/(t[1]-t[0]),NFT)
    xform_kernel = np.zeros([NFT, len(data_index)], dtype=np.complex64)

    window_func = scipy.signal.get_window(window, len(data_index))

    if xform=="B":
        for i in range(NFT):
            xform_kernel[i,:] = scipy.special.jv(0, 2*np.pi*f[i]*t)
    else:
        for i in range(NFT):
            xform_kernel[i,:] = np.exp(2j*np.pi*t*f[i])

    if detrend:
        mean_Pxx = np.abs(np.einsum("ft,t,t->f", xform_kernel, window_func, analysis_data[data_index]-np.mean(analysis_data[data_index]), optimize="optimal"))**2
    else:
        mean_Pxx = np.abs(np.einsum("ft,t,t->f", xform_kernel, window_func, analysis_data[data_index], optimize="optimal"))**2

    return f, mean_Pxx
