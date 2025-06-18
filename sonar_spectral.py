#!/usr/bin/env python3
#Refactored code
import asyncio
import pyaudio

import librosa
import numpy as np
from scipy import signal, fftpack
import scipy
import logging
import librosa
import numpy as np
import scipy
from typing import Tuple
import pywt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
#todos refactor spectral and sonar to use async
#todos, add try except blocks and logging exceptions

class Spectral:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    async def calculate_spectral_analysis(self, data_float: np.array) -> dict:
        n_fft = min(256, len(data_float))
        mfccs = librosa.feature.mfcc(y=data_float, sr=self.sample_rate, n_fft=n_fft)
        spectral_centroid = librosa.feature.spectral_centroid(y=data_float, sr=self.sample_rate, n_fft=n_fft)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data_float, sr=self.sample_rate, n_fft=n_fft)
        fft_out = np.fft.fft(data_float)
        freqs = np.fft.fftfreq(len(data_float), d=1.0 / self.sample_rate)
        psd = np.abs(fft_out) ** 2
        weighted_sum_psd = np.sum(psd * np.abs(freqs)) / np.sum(np.abs(freqs))
        return {
            'mfccs': mfccs,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'fft_out': fft_out,
            'freqs': freqs,
            'freqs_p': freqs,
            'psd': psd,
            'weighted_sum_psd': weighted_sum_psd,
        }
        
    async def _calculate_mfccs(self, data_float: np.array) -> np.array:
        return librosa.feature.mfcc(y=data_float, sr=self.sample_rate, n_fft=min(200, len(data_float)))
    
    async def _calculate_spectral_centroid(self, data_float: np.array) -> np.array:
        return librosa.feature.spectral_centroid(y=data_float, sr=self.sample_rate)

    async def _calculate_spectral_bandwidth(self, data_float: np.array) -> np.array:
        return librosa.feature.spectral_bandwidth(y=data_float, sr=self.sample_rate)

    async def _calculate_fourier_transform(self, data_float: np.array) -> np.array:
        return scipy.fft.fft(data_float)

    async def _calculate_frequencies(self, data_float: np.array) -> np.array:
        return scipy.fft.fftfreq(len(data_float), d=1/self.sample_rate)

    async def _calculate_periodogram(self, data_float: np.array) -> Tuple:
        return scipy.signal.periodogram(data_float, fs=self.sample_rate)


    async def extract_chroma_features(self, echoes):
        chroma_stft = librosa.feature.chroma_stft(y=echoes, sr=self.sample_rate)
        return chroma_stft


class Sonar:
    def __init__(self, frequency, pulse_length, sample_rate):
        """
        Initialize Sonar parameters.

        Args:
            frequency (float): Frequency of the sonar pulse in Hz.
            pulse_length (float): Length of the sonar pulse in seconds.
            sample_rate (int): Sample rate of the audio stream in Hz.
        """
        self.frequency = frequency
        self.pulse_length = pulse_length
        self.sample_rate = sample_rate
        self.t = np.arange(0, pulse_length, 1/sample_rate)
        self.input_stream = None
        self.spectral = Spectral(sample_rate=44100)
        self.spectral_generator = SpectrogramGenerator(sample_rate)
        

    async def generate_pulse(self):
        """
        Generate a sonar pulse with a gentle rise and fall.

        Returns:
            np.ndarray: Sonar pulse waveform.
        """
        try:
            
            pulse = np.sin(2 * np.pi * self.frequency * self.t)
            pulse *= np.hanning(len(pulse))  # Apply a Hanning window
            return pulse.astype(np.float32)
        except Exception as e:
            logging.exception(msg=e)
            

    async def generate_multiple_pulses(self, *frequencies):
        """
        Generate multiple sonar pulses at different frequencies.
        
        Args:
            *frequencies (float): Variable number of frequencies for the sonar pulses.
        """
        pulses = []
        for frequency in frequencies:
            t = np.arange(0, self.pulse_length, 1/self.sample_rate)
            pulse = np.sin(2 * np.pi * frequency * t)
            pulse *= np.hanning(len(pulse))  # Apply a Hanning window
            pulses.append(pulse.astype(np.float32))
        return np.concatenate(pulses)

    async def calculate_movement_speed(self, doppler_shift_freq):
        """
        Calculate the movement speed based on the Doppler shift frequency.

        Args:
            doppler_shift_freq (float): Doppler shift frequency in Hz.

        Returns:
            float: Movement speed in meters per second.
        """
        # Assuming the speed of sound in air is approximately 343 m/s
        speed_of_sound = 343
        movement_speed = (doppler_shift_freq * speed_of_sound) / (2 * self.frequency)
        return movement_speed

    async def calculate_doppler_shift(self, echoes, peak_idx):
        """
        Calculate the Doppler shift from the echoes.

        Args:
            echoes (np.ndarray): Concatenated echoes.
            peak_idx (int): Index of the peak in the FFT spectrum.

        Returns:
            float: Doppler shift frequency in Hz.
        """
        fft_echoes = np.fft.fft(echoes)
        freq = np.fft.fftfreq(len(echoes), 1/self.sample_rate)
        return freq[peak_idx]
    
    async def calculate_time_of_flight(self, echoes):
        """
        Calculate the time of flight from the echoes.

        Args:
            echoes (np.ndarray): Concatenated echoes.

        Returns:
            float: Time of flight in seconds.
        """
        # Assuming the speed of sound in air is approximately 343 m/s
        speed_of_sound = 343

        # Calculate the time of flight
        time_of_flight = len(echoes) / self.sample_rate / 2  # Divide by 2 for round trip

        return time_of_flight
    
    async def calculate_range(self, time_of_flight, speed_of_sound=1500):
        """
        Calculate the range from the time of flight and speed of sound.

        Args:
            time_of_flight (float): Time of flight in seconds.
            speed_of_sound (float): Speed of sound in m/s. Default is 1500 m/s for water.

        Returns:
            float: Range in meters.
        """
        # Calculate the range
        range_ = time_of_flight * speed_of_sound / 2  # Divide by 2 for round trip

        return range
    
    async def calculate_water_temperature(self, speed_of_sound):
        """
        Calculate the water temperature based on the speed of sound.

        Args:
            speed_of_sound (float): Speed of sound in meters per second.

        Returns:
            float: Estimated water temperature in degrees Celsius.
        """
        temperature = (speed_of_sound - 1400) / 4.59
        return temperature


    async def calculate_movement_velocity(self, movement_speed, doppler_shift_freq):
            """
            Calculate the movement velocity based on the movement speed and Doppler shift frequency.

                Args:
                movement_speed (float): Movement speed in meters per second.
            doppler_shift_freq (float): Doppler shift frequency in Hz.

        Returns:
            float: Movement velocity in meters per second squared.
            """
        # Assuming a simple linear relationship between movement speed and velocity
            movement_velocity = movement_speed * doppler_shift_freq / self.frequency
            return movement_velocity
        
    async def calculate_wavelet_analysis(self, echoes):
        """Perform wavelet analysis on the echoes using wavedec."""
        coeffs = pywt.wavedec(echoes, 'db4', level=3)
        cA3, cD3, cD2, cD1 = coeffs
        return cA3, cD3, cD2, cD1


    async def calculate_peaks(self, echoes):
        """
        Detect peaks in the echoes.

        Args:
            echoes (np.ndarray): Concatenated echoes.

        Returns:
            np.ndarray: Indices of the detected peaks.
        """
        # Compute the FFT of the echoes
        fft_echoes = np.fft.fft(echoes)

        # Apply a windowing function to reduce spectral leakage
        fft_echoes = fft_echoes * np.hanning(len(fft_echoes))

        # Compute the power spectral density (PSD)
        psd = np.abs(fft_echoes) ** 2

        # Detect peaks in the PSD
        peaks, properties = scipy.signal.find_peaks(psd, height=np.max(psd) * 0.5)

        return peaks
    async def read_input_data(self, input_stream):
        data = np.frombuffer(input_stream.read(num_frames=1024, exception_on_overflow=False), dtype=np.int16)
        data_float = data.astype(np.float32) / 32768.0
        data_float = librosa.util.normalize(data_float)
        return data_float
    
    async def denoise_echoes(self, echoes):
        
        """
        denoise the echoes using wavelet analysis
        
        """
        try:
            
            
            coeffs = pywt.wavedec(echoes, 'db4', level=3)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(echoes)))
            coeffs[-1] = pywt.threshold(coeffs[-1], threshold, mode='soft')
            coeffs[-2] = pywt.threshold(coeffs[-2], threshold, mode='soft')
            coeffs[-3] = pywt.threshold(coeffs[-3], threshold, mode='soft')
            denoised_echoes = pywt.waverec(coeffs, 'db4')
            return denoised_echoes
        
        except Exception as e:
            logging.exception(msg=e)
            
    async def process_echoes(self, echoes):
        
        # Process echoes, including calculating peaks, Doppler shift, movement speed, and movement velocity
        echoes *= np.hanning(len(echoes))
        denoised_echoes = await self.denoise_echoes(echoes=echoes)
        #create a spectrogram from the denoised echoes
    
        
        
       
        #add the spectral generator code here for the map

       
        #chroma data on the denoised echoes
        # Extract chroma features
        #chroma_features = self.spectral.extract_chroma_features(denoised_echoes)

        peaks = await self.calculate_peaks(denoised_echoes)
        results = []
        for peak in peaks:
            doppler_shift_freq = await self.calculate_doppler_shift(denoised_echoes, peak)
            movement_speed = await self.calculate_movement_speed(doppler_shift_freq)
            movement_velocity = await self.calculate_movement_velocity(movement_speed, doppler_shift_freq)
            window_start = max(0, peak - 100)
            window_end = min(len(denoised_echoes), peak + 100)
            #spectral_features = self.spectral.calculate_spectral_analysis(denoised_echoes[window_start:window_end])
            cA3, cD3, cD2, cD1 = await self.calculate_wavelet_analysis(denoised_echoes[window_start:window_end])
           

          
        return results
    

    async def print_results(self, results):
        
        # Print the results
        for result in results:
            print(f"Doppler Shift: {result['doppler_shift_freq']} Hz")
            print(f"Movement Speed: {result['movement_speed']} m/s")
            print(f"Movement Velocity: {result['movement_velocity']} m/s^2")
            # You can also print other spectral features if needed
            print(f"MFCCs: {result['mfccs']}")
            print(f"Spectral Centroid: {result['spectral_centroid']}")
            print(f"Spectral Bandwidth: {result['spectral_bandwidth']}")
            print(f"Wavelet Coeffs: {result['wavelet_coeffs']}")
            print()  # Empty line for better readability
class SpectrogramGenerator:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.fig, self.ax = plt.subplots()

    def generate_spectrogram(self, denoised_echoes):
        self.ax.clear()
        self.ax.specgram(denoised_echoes, Fs=self.sample_rate, cmap='inferno')
        self.ax.set_title('Spectrogram of Denoised Echoes')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

    async def animate(self, denoised_echoes_generator):
        async def update(frame):
            #
            denoised_echoes = next(denoised_echoes_generator)
            await self.generate_spectrogram(denoised_echoes)
           
            return self.ax,

        ani = animation.FuncAnimation(self.fig, update, interval=1)
        
        
    async def run(self, denoised_echoes_generator):
        await self.animate(denoised_echoes_generator)
        

async def main():
    try:
        
        frequency = 200e3  # Hz
        pulse_length = 10e-3  # s
        sample_rate = 44100  # Hz

        sonar = Sonar(frequency, pulse_length, sample_rate)

        p = pyaudio.PyAudio()
        output_stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
        input_stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
        high_frequency = 250e3  # Hz
        med_frequency = 200e3  # Hz
        low_frequency = 150e3  # Hz
        echoes_list = []
        while True:
            pulses = await sonar.generate_multiple_pulses(high_frequency, med_frequency, low_frequency)
            output_stream.write(pulses)
            
            #read input data
            
            data_float = await sonar.read_input_data(input_stream=input_stream)
            
            echoes_list.append(data_float)
            if len(echoes_list) >= 10:  # Accumulate at least 10 echoes
                #process the echoes
            
                echoes = np.concatenate(echoes_list)
                echoes *= np.hanning(len(echoes))
                results = await sonar.process_echoes(echoes)
                
                
    
                #reset the echoes list
                echoes_list = []  # Reset the echoes list
    except Exception as e:
        print(f"the following error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())

