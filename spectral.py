#!/usr/bin/env python3
#Refactored code
import librosa
import numpy as np
import scipy
from typing import Tuple

class AudioAnalyzer:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def calculate_spectral_analysis(self, data_float: np.array) -> dict:
        mfccs = self._calculate_mfccs(data_float)
        spectral_centroid = self._calculate_spectral_centroid(data_float)
        spectral_bandwidth = self._calculate_spectral_bandwidth(data_float)
        fft_out = self._calculate_fourier_transform(data_float)
        freqs = self._calculate_frequencies(data_float)
        freqs_p, psd = self._calculate_periodogram(data_float)
        weighted_sum_psd = np.sum(psd * freqs_p) / np.sum(freqs_p)

        return {
            'mfccs': mfccs,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'fft_out': fft_out,
            'freqs': freqs,
            'freqs_p': freqs_p,
            'psd': psd,
            'weighted_sum_psd': weighted_sum_psd,
        }

    def _calculate_mfccs(self, data_float: np.array) -> np.array:
        return librosa.feature.mfcc(y=data_float, sr=self.sample_rate, n_fft=min(200, len(data_float)))
    def _calculate_spectral_centroid(self, data_float: np.array) -> np.array:
        return librosa.feature.spectral_centroid(y=data_float, sr=self.sample_rate)

    def _calculate_spectral_bandwidth(self, data_float: np.array) -> np.array:
        return librosa.feature.spectral_bandwidth(y=data_float, sr=self.sample_rate)

    def _calculate_fourier_transform(self, data_float: np.array) -> np.array:
        return scipy.fft.fft(data_float)

    def _calculate_frequencies(self, data_float: np.array) -> np.array:
        return scipy.fft.fftfreq(len(data_float), d=1/self.sample_rate)

    def _calculate_periodogram(self, data_float: np.array) -> Tuple:
        return scipy.signal.periodogram(data_float, fs=self.sample_rate)

# Example usage
def main():
    analyzer = AudioAnalyzer()
    data_float = np.random.rand(1024)  # Replace with your audio data
    result = analyzer.calculate_spectral_analysis(data_float)
    print(result)

if __name__ == '__main__':
    main()
