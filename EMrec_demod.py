import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import resample_poly, firwin, bilinear, lfilter
import time

samples = np.fromfile('EMrecording1_20130922_193609Z_145941kHz_IQ.wav', np.complex64)
print(samples)

f = open('EMrecording1_20130922_193609Z_145941kHz_IQ.wav', "rb")
f.seek(int(40))
samples = np.fromfile(f, np.int16)
samples = samples/32768
samples = samples[::2] + 1j*samples[1::2]

plt.plot(np.real(samples), np.imag(samples), '.')
plt.grid(True)
plt.show()

fs = 96e3
f0 = 0
Ts = 1 / fs

signal_f, signal_psd = scipy.signal.welch(samples, fs, window='hamming', scaling='spectrum',
                                          return_onesided=False, axis=-1)
signal_f = np.fft.fftshift(signal_f)
signal_psd = np.fft.fftshift(signal_psd)

plt.figure(0, figsize=(14, 6))
plt.semilogy(signal_f, signal_psd, label='PSD')
plt.title("Welch spectrum")
plt.xlabel("Frequency  [Hz]")
plt.ylabel("Power [V**2/Hz]")
plt.grid(True)
plt.show()

psd = np.fft.fftshift(np.abs(np.fft.fft(samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))

plt.figure(0, figsize=(14, 6))
plt.title("Spectral Density [PSD]")
plt.plot(f, psd)
plt.xlabel("Frequency  [Hz]")
plt.ylabel("Magnitude  [dB]")
plt.grid(True)
plt.show()

sample_rate = 96e3
fft_size = 1024
num_rows = int(np.floor(len(samples)/fft_size))
spectrogram = np.zeros((num_rows, fft_size)) # Create an empty array of the right size
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size])))**2)

spectrogram = spectrogram[:,fft_size//2:] # Remove the negative freqs because we view real signal

plt.figure(0, figsize=(14, 8))
plt.imshow(spectrogram, aspect='auto', extent = [0, sample_rate/2/1e3, 0, len(samples)/sample_rate])
plt.xlabel("Frequency [kHz]")
plt.ylabel("Time [s]")
plt.show()

# Freq shift
N = len(samples)
f_o = -18.5e3 # amount we need to shift by
t = np.arange(N)/sample_rate # time vector
x = samples * np.exp(2j*np.pi*f_o*t) # Shift the frequency

# Replot
num_rows = int(np.floor(len(x)/fft_size))
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
spectrogram = spectrogram[:,fft_size//2:] # Remove negative freqs because we view real signal

plt.figure(0, figsize=(14, 8))
plt.imshow(spectrogram, aspect='auto', extent = [0, sample_rate/2/1e3, 0, len(samples)/sample_rate])
plt.xlabel("Frequency [kHz]")
plt.ylabel("Time [s]")
plt.show()

plt.plot(np.real(x[0:1000]), np.imag(x[0:1000]), '.')
plt.grid(True)
plt.show()

# Low-Pass Filter
taps = firwin(numtaps=101, cutoff=7.5e3, fs=sample_rate)
x = np.convolve(x, taps, 'valid')

# Replot
num_rows = int(np.floor(len(x)/fft_size))
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
spectrogram = spectrogram[:,fft_size//2:] # get rid of negative freqs because we view real signal

plt.figure(0, figsize=(14, 10))
plt.imshow(spectrogram, aspect='auto', extent = [0, sample_rate/2/1e3, 0, len(x)/sample_rate])
plt.xlabel("Frequency [kHz]")
plt.ylabel("Time [s]")
plt.show()

# Plot constellation to make sure it looks right
plt.plot(np.real(x[0:1000]), np.imag(x[0:1000]), '.')
plt.grid(True)
plt.show()

# Bandpass filter to isolate the BPSK signal
taps = firwin(numtaps=501, cutoff=[0.05e3, 10e3], fs=sample_rate, pass_zero=False)
x = np.convolve(x, taps, 'valid')

# Decimate by 10, now that we filtered and there wont be aliasing
x = x[::10]
sample_rate = 96e3/10

# Replot
num_rows = int(np.floor(len(x)/fft_size))
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
spectrogram = spectrogram[:,fft_size//2:] # get rid of negative freqs because we view real signal

plt.figure(0, figsize=(14, 10))
plt.imshow(spectrogram, aspect='auto', extent = [0, sample_rate/2/1e3, 0, len(x)/sample_rate])
plt.xlabel("Frequency [kHz]")
plt.ylabel("Time [s]")
plt.show()

# SYMBOL SYNCHRONIZATION. MUELLER AND MULLER TIME SYNC ALGORITHM
samples = x # For the sake of matching variable names
samples_interpolated = resample_poly(samples, 32, 1) # 32 chosen as interpolation factor

sps = 16
mu = 0.01 # initial estimate of phase of sample
out = np.zeros(len(samples) + 10, dtype=np.complex64)
out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # stores values, each iteration we need the previous 2 values plus current value
i_in = 0 # input samples index
i_out = 2 # output index (let first two outputs be 0)

while i_out < len(samples) and i_in+32 < len(samples):
    out[i_out] = samples_interpolated[i_in*32 + int(mu*32)] # grab what we think is the "best" sample
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
    x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
    y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
    mm_val = np.real(y - x)
    mu += sps + 0.01*mm_val
    i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
    mu = mu - np.floor(mu) # remove the integer part of mu
    i_out += 1 # increment output index
x = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)

# Plot constellation to make sure it looks right
plt.plot(np.real(x[0:1000]), np.imag(x[0:1000]), '.')
plt.grid(True)
plt.show()