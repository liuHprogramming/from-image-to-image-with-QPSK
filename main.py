import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

# von Bild zum Bitstream
im = Image.open("sunflower.png")
im.show()
w=im.width
h=im.height
f=im.format
print(w, h, f)
im2 = im.convert("1")
print(im2.mode)
im2.show()
Bitstream = []
for i in range(w):
    for k in range(h):
        Bitstream.append(im2.getpixel((i, k)))
        k = k+1
    i = i+1
for t in range(len(Bitstream)):
    if Bitstream[t] == 255:
        Bitstream[t] = 1
print(Bitstream)
bitstream = np.array(Bitstream)
print(bitstream)

# Modulation
N = len(bitstream)
T = 1
fc = 1/8
Fs = 1
bitstream = 2*bitstream-1 # fuer IQ Zerlegung
I = np.array([]) # ungerade
Q = np.array([]) # gerade

for i in range(1, N+1):
    if np.mod(i, 2) != 0:
        I = np.insert(I, len(I), bitstream[i-1]) # 1/2 des Bitstreams mit ungeraden Indexen
    else:
        Q = np.insert(Q, len(Q), bitstream[i-1]) # 1/2 des Bitstreams mit geraden Indexen
bit_data = np.array([])
for i in range(1,N+1):
    bit_data = np.insert(bit_data,len(bit_data),bitstream[i-1]*np.ones(T*Fs))
I_data = np.array([])
Q_data = np.array([])
for i in range(1,int(N/2)+1):
    I_data = np.insert(I_data,len(I_data),I[i-1]*np.ones(2*T*Fs))
    Q_data = np.insert(Q_data,len(Q_data),Q[i-1]*np.ones(2*T*Fs))
t = np.array([])
for i in np.arange(40,140,1/Fs):
    t = np.insert(t,len(t),i)
plt.subplot(3,1,1)
plt.plot(t,bit_data[40:140])
plt.legend(["Bitstream"],loc='upper right')
plt.subplot(3,1,2)
plt.plot(t,I_data[40:140])
plt.legend(["I_Bitstream"],loc='upper right')
plt.subplot(3,1,3)
plt.plot(t, Q_data[40:140])
plt.legend(["Q_Bistream"], loc='upper right')
plt.show()

# I,Q Bitstream-Signal zum Modulation-QPSK Signal(sinus/cos-formig signal,durch Summe der I und Q Signal)
bit_t = np.array([])
for i in np.arange(0, 4*T, 1/Fs):
    bit_t = np.insert(bit_t, len(bit_t), i)
I_signal=np.array([])
Q_signal=np.array([])
for i in range(1, int(N/2)+1):
    I_signal = np.insert(I_signal, len(I_signal), I[i-1]*np.cos(2*np.pi*fc*bit_t))
    Q_signal = np.insert(Q_signal,len(Q_signal),Q[i-1]*np.cos(2*np.pi*fc*bit_t+np.pi/2))    # naemlich sin(2*pi*fc*t)
QPSK_signal = I_signal+Q_signal
# Zeichen den I-Signal,Q-Signal und QPSK-Signal
plt.subplot(3,1,1)
plt.plot(t, I_signal[40:140])
plt.legend(["I-signal"], loc='upper right')
plt.subplot(3,1,2)
plt.plot(t, Q_signal[40:140])
plt.legend(["Q-signal"], loc='upper right')
plt.subplot(3,1,3)
plt.plot(t, QPSK_signal[40:140])
plt.legend(["QPSK-signal"], loc='upper right')
plt.show()











