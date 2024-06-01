import math

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


sample_rate, audio_data = wav.read('C:/Users/79004/Documents/Downloads/sample-6s.wav')

# Преобразование в моно звук
if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)

plt.figure()
plt.subplot(2, 1, 1)
plt.title('Сигнал до изменений')
plt.plot(audio_data)

# Преобразование Фурье
fft_result = np.fft.fft(audio_data)
magnitude = np.abs(fft_result)
phase = np.angle(fft_result)

# Создание уникальной цифровой подписи путем фазового сдвига
n = len(phase)
unique_pattern = np.random.rand(n) * 2 * np.pi  # Генерация случайной фазовой информации

# Применение подписи
# new_fft_result = magnitude * np.exp(1j * (phase + unique_pattern))
for i in range(n-1000):
    unique_pattern[i] = 0
for i in range(n):
    unique_pattern[i] = math.pi
new_fft_result = magnitude * np.exp(1j * (phase + unique_pattern))
for i in range(n//2, n):
# for i in range(n // 2):
    new_fft_result[i] *= 0
# new_fft_result = magnitude * np.exp(1j * (phase))

# Обратное преобразование Фурье
modified_audio_data = np.fft.ifft(new_fft_result).real

# Сохранение модифицированного аудиофайла
wav.write('modified_audio.wav', sample_rate, modified_audio_data.astype(np.int16))

plt.subplot(2, 1, 2)
plt.title('Сигнал после изменений')
plt.plot(modified_audio_data)
plt.show()