import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

# Schritt 1
# von Bild zum Bitstream
im = Image.open("sunflower.png")
im.show()
w=im.width  # width of image
h=im.height  # height of image
f=im.format   # format of image
print(w, h, f)
im2 = im.convert("P")  # zum P Mode umwandeln
print(im2.mode)
im2.show()
Bitstream = []
for i in range(w):
    for k in range(h):
        Farbe = im2.getpixel((i, k))  # bekommen fuer jeden Punkt einen Wert des Farbes aus colorbar
        Farbe_str = ''
        while Farbe > 0:                        # wandeln fuer jeden Zahlwert 8 Bits aus 0,1
            Farbe_str += str(Farbe % 2)
            Farbe = Farbe // 2
        if len(Farbe_str) < 8:
            for q in range(8-len(Farbe_str)):
                Farbe_str += str(0)
        Farbe_str = Farbe_str[::-1]
        for j in Farbe_str:
            Bitstream.append(int(j))
        k = k+1
    i = i+1

print(Bitstream)  # Bitstream List
bitstream = np.array(Bitstream)
print(bitstream)    # Bitstream nummpy array
print(len(bitstream))

# Bitstream in I, Q zerlegen und mit Rechteck Signal falten
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
    bit_data = np.insert(bit_data,len(bit_data),bitstream[i-1]*np.ones(T*Fs))  # Bitstream mit Rechteck Signal falten
I_data = np.array([])
Q_data = np.array([])
for i in range(1,int(N/2)+1):
    I_data = np.insert(I_data,len(I_data),I[i-1]*np.ones(2*T*Fs))  # I-Bitstream mit Rechteck Signal falten
    Q_data = np.insert(Q_data,len(Q_data),Q[i-1]*np.ones(2*T*Fs))   # Q-Bitstream mit Rechteck Signal falten
t = np.array([])
for i in np.arange(40,140,1/Fs):
    t = np.insert(t,len(t),i)

# I, Q und den gesamten Bitstream zeichnen
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

# Schritt 2
# Modulation
# I,Q Bitstream-Signal zum Modulation-QPSK Signal(sinus/cos-formig signal,durch Summe der I und Q Signal)
bit_t = np.array([])
for i in np.arange(0, 4*T, 1/Fs):
    bit_t = np.insert(bit_t, len(bit_t), i)
I_signal=np.array([])
Q_signal=np.array([])
for i in range(1, int(N/2)+1):
    I_signal = np.insert(I_signal, len(I_signal), I[i-1]*np.cos(2*np.pi*fc*bit_t))
# Rechteck-Signal(I-Bitstream) mit cos(2*pi*fc*t) falten
    Q_signal = np.insert(Q_signal,len(Q_signal),Q[i-1]*np.cos(2*np.pi*fc*bit_t+np.pi/2))
# Rechteck-Signal(Q-Bitstream) mit sin(2*pi*fc*t) falten
QPSK_signal = I_signal+Q_signal # QPSK = I+Q

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

# Schritt 3
# Channel
# Funktion fuer additive white gassian noise im Channel
def awgn(x, snr):
    snr = 10 ** (snr/10)
    xpower = np.sum(x**2)/len(x)
    npower = xpower/snr
    noise = np.random.randn(len(x))  # 1*len(x) array with standord normal distribution
    return noise*np.sqrt(npower) + x

snr=1
QPSK_channel = awgn(QPSK_signal, snr)  # Addieren white gassian Noise im Channel

# Schritt 4
# demodulation
I_demo = np.array([])
Q_demo = np.array([])
for i in range(1, int(N/2)+1):
    I_ausgabe = QPSK_channel[(i-1)*len(bit_t):i*len(bit_t)]*np.cos(2*np.pi*fc*bit_t)  # fuer I_Bitstream
# QPSK mit cos() falten und jede Summe des length(bit_t) Werts rechnen
    if np.sum(I_ausgabe) > 0:
        I_demo = np.insert(I_demo, len(I_demo), 1)  # 1 wenn > 0
    else:
        I_demo = np.insert(I_demo, len(I_demo), -1)   # -1 ansonstens

    Q_ausgabe = QPSK_channel[(i-1)*len(bit_t):i*len(bit_t)]*np.cos(2*np.pi*fc*bit_t+np.pi/2)  # fuer Q_Bitstream
# QPSK mit sin() falten und jede Summe des length(bit_t) Werts rechnen
    if np.sum(Q_ausgabe) > 0:
        Q_demo = np.insert(Q_demo, len(Q_demo), 1)  # 1 wenn > 0
    else:
        Q_demo = np.insert(Q_demo, len(Q_demo), -1)  # -1 ansonstens

# bekommen den gesamten Bitstream aus der Summe von I,Q Bitstream
QPSK_demo = np.array([])
for i in range(1, N+1):
    if np.mod(i, 2) != 0:
        QPSK_demo = np.insert(QPSK_demo, len(QPSK_demo), I_demo[int((i-1)/2)])
    else:
        QPSK_demo = np.insert(QPSK_demo, len(QPSK_demo), Q_demo[int((i/2)-1)])

# Bitstream mit Rechteck Signal falten
QPSK_recover = np.array([])
for i in range(1, N+1):
    QPSK_recover = np.insert(QPSK_recover,len(QPSK_recover), QPSK_demo[i-1]*np.ones(T*Fs))
I_recover = np.array([])
Q_recover = np.array([])
# I, Q Bitstream mit Rechteck Signal falten
for i in range(1, int(N/2)+1):
    I_recover = np.insert(I_recover, len(I_recover), I_demo[i-1]*np.ones(2*T*Fs))
    Q_recover = np.insert(Q_recover, len(Q_recover), Q_demo[i-1]*np.ones(2*T*Fs))

# Zeichen des I, Q und gesamten Bitstream (mit Rechteck Signal falten) nach der Modulation
plt.subplot(3, 1, 1)
plt.plot(t, QPSK_recover[40:140])
plt.legend(["Bitstream"], loc='upper right')
plt.subplot(3, 1, 2)
plt.plot(t, I_recover[40:140])
plt.legend(["I_Bitstream"], loc='upper right')
plt.subplot(3, 1, 3)
plt.plot(t, Q_recover[40:140])
plt.legend(["Q_Bitstream"], loc='upper right')
plt.show()

# Schritt 5
# Bitstream zurueck zum Bild
QPSK_demo = (QPSK_demo+1)/2  # -1 -> 0, 1 -> 1
print(QPSK_demo)
print(len(QPSK_demo))
QPSK_demo = list(QPSK_demo)
for i in range(1, len(QPSK_demo)+1):
    QPSK_demo[i-1] = int(QPSK_demo[i-1])
print(QPSK_demo)

# 8 Bits zurueck zu einen int Farbwert umwandeln , z.B. 10001000 -> 136
QPSK_Farbe = []
for i in range(1, int(N/8 + 1)):
    temp = ''
    for j in QPSK_demo[(i-1)*8:(i*8-1)]:
        temp += str(j)
    Farbe = int(temp, 2)
    QPSK_Farbe.append(Farbe)
for i in range(1, len(QPSK_Farbe)+1):
    QPSK_Farbe[i-1] = QPSK_Farbe[i-1]*2

QPSK_Farbe = np.array(QPSK_Farbe)
print(QPSK_Farbe)
print(len(QPSK_Farbe))

# Bild zeichnen
QPSK_Farbe = QPSK_Farbe.reshape(40, 40)  # in einem 40*40 Matrix umwandeln
print(QPSK_Farbe)
# Bild zeichnen, Farbwert zwischen 2 - 45, colorbar=yellow-orange-red
plt.imshow(QPSK_Farbe, origin='lower', vmax=45, vmin=2, cmap="YlOrRd")
plt.colorbar(label='value of color')  # colorbar und label im Bild zeigen
plt.show()


