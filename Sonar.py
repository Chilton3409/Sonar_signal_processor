#!/usr/bin/env python3
#Refactored code
import asyncio
import pyaudio
import librosa
import numpy as np
from scipy import signal, fftpack
import scipy

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
    

    def generate_pulse(self):
        """
        Generate a sonar pulse with a gentle rise and fall.

        Returns:
            np.ndarray: Sonar pulse waveform.
        """
        pulse = np.sin(2 * np.pi * self.frequency * self.t)
        pulse *= np.hanning(len(pulse))  # Apply a Hanning window
        return pulse.astype(np.float32)

    def generate_multiple_pulses(self, *frequencies):
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

    def calculate_movement_speed(self, doppler_shift_freq):
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

    def calculate_doppler_shift(self, echoes, peak_idx):
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

    def calculate_movement_velocity(self, movement_speed, doppler_shift_freq):
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

    def calculate_peaks(self, echoes):
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
    def read_input_data(self, input_stream):
        data = np.frombuffer(input_stream.read(num_frames=1024, exception_on_overflow=False), dtype=np.int16)
        data_float = data.astype(np.float32) / 32768.0
        return data_float
    
    def process_echoes(self, echoes):
          # Process echoes, including calculating peaks, Doppler shift, movement speed, and movement velocity
        echoes *= np.hanning(len(echoes))
        peaks = self.calculate_peaks(echoes)
        results = []
        for peak in peaks:
            doppler_shift_freq = self.calculate_doppler_shift(echoes, peak)
            movement_speed = self.calculate_movement_speed(doppler_shift_freq)
            movement_velocity = self.calculate_movement_velocity(movement_speed, doppler_shift_freq)
            results.append((doppler_shift_freq, movement_speed, movement_velocity))
        return results

    def print_results(self, results):
         # Print the results
        for result in results:
            print(f"Doppler Shift: {result[0]} Hz")
            print(f"Movement Speed: {result[1]} m/s")
            print(f"Movement Velocity: {result[2]} m/s^2")
        return



def main():
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
            pulses = sonar.generate_multiple_pulses(high_frequency, med_frequency, low_frequency)
            output_stream.write(pulses)
            
            #read input data
            
            data_float = sonar.read_input_data(input_stream=input_stream)
            
            echoes_list.append(data_float)

            if len(echoes_list) >= 10:  # Accumulate at least 10 echoes
                #process the echoes
                
                
                
                echoes = np.concatenate(echoes_list)
                echoes *= np.hanning(len(echoes))
                results = sonar.process_echoes(echoes)
                
                
                #print the results
                sonar.print_results(results)
                

                #reset the echoes list
                echoes_list = []  # Reset the echoes list
    except Exception as e:
        print(f"the following error occurred: {e}")


if __name__ == "__main__":
    main()

