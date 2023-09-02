from scipy import signal

import numpy as np
import matplotlib.pyplot as plt

dt = 1/48000                #Sampling interval of 0.001 seconds.
t = np.arange(0,2,dt)     #Time duration of 0 to 2.00 seconds
f0 = 440                    #Initial frequency of 50 hz
t1 = 2                     #Period of time evolving signal = 2 secs

# x should be the sinus function with 440 Hz
x = 170 * np.sin(2*np.pi*t*f0)

x_lim = 2
y_lim = 5000

# Plot x over time
plt.figure(1)
plt.plot(t,x)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Chirp Signal')
plt.show()

fs = 1/dt
nfft = 10000
plt.specgram(x, NFFT=nfft, Fs=1/dt, noverlap=500, cmap='jet_r')
plt.colorbar()
plt.ylim([0, y_lim])
plt.xlim([0, x_lim])
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.show()