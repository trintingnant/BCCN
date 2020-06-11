from matplotlib import mlab
import numpy as np

def estimate_decoder(time,s,r,dt,nfft=2**12):
    '''
    Estimates the decoding kernel K and reconstructs the original stimuls
    
    Inputs: time: time vector 
            s: stimulus
            r: response
            dt: sampling period
            nfft: number of datapoints to be used for spectral estimation (window length)
            
    Outputs:   k: the decoding kernel (length of the nfft)
               k_time: the time vector of the decoding kernel (length of the nfft)
               s_est: the estimated stimulus (length of s and r minus a window length)
               s_est_time: time vector for s_est
               NOTE: we discard data points 0:nfft/2 and -nnft2:end due to boundary effects
    '''

    Fs=1/dt     # sampling frequency
        
    # compute the cross spectrum between response and stimulus
    Qf_rs, freqs = mlab.csd(r,s,nfft,Fs=Fs) 
    Qf_rs*=dt
    
    # computes the response power spectrum
    Qf_rr, freqs = mlab.psd(r,nfft,Fs=Fs)
    Qf_rr*=dt
    
    # take the ratio 
    Kf = Qf_rs/Qf_rr
    
    # we need to add negative frequency with (conjugate values) to comply to the input 
    # requirements of the ifft function
    Kf= np.hstack([Kf,Kf[::-1].conj()[1:-1]])
    
    k = np.fft.fftshift(np.fft.ifft(Kf).real)/dt
    k_time=(np.arange(nfft)-int(nfft/2))*dt
    
    # compute estimated stimulus
    s_est=np.convolve(r,k,'same')*dt
    
    # crop s_est and time vector appropriately
    s_est = s_est[int(nfft/2):int(-nfft/2)]
    s_est_time = time[int(nfft/2):int(-nfft/2)]-dt
    
    return k,k_time,s_est,s_est_time