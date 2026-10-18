##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 06-01-2023

import numpy as np


# This class performs a Fourier transformation on the data to find frequencies that occur
# often and filter noise.
class FourierTransformation:

    # Find the magnitudes of the positive-frequency components using a fast Fourier transform.
    def find_fft_transformation(self, data, sampling_rate):
        transformation = np.fft.rfft(data, n=len(data))
        return np.abs(transformation)

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, cols, window_size, sampling_rate):
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")

        freqs = np.round(np.fft.rfftfreq(window_size) * sampling_rate, 3)

        for col in cols:
            data_table[col + "_max_freq"] = np.nan
            data_table[col + "_freq_weighted"] = np.nan
            data_table[col + "_pse"] = np.nan
            for freq in freqs:
                data_table[
                    col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
                ] = np.nan

        # Start once a complete window is available; use exactly window_size samples.
        for i in range(window_size, len(data_table.index) + 1):
            for col in cols:
                window = data_table[col].iloc[i - window_size:i].to_numpy(dtype=float)
                if not np.all(np.isfinite(window)):
                    continue

                amplitudes = self.find_fft_transformation(window, sampling_rate)
                for j, freq in enumerate(freqs):
                    data_table.loc[
                        data_table.index[i - 1],
                        col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size),
                    ] = amplitudes[j]

                # A zero-energy window has no meaningful dominant or weighted frequency.
                total_amplitude = np.sum(amplitudes)
                if total_amplitude <= 0 or not np.isfinite(total_amplitude):
                    continue

                data_table.loc[data_table.index[i - 1], col + "_max_freq"] = freqs[
                    int(np.argmax(amplitudes))
                ]
                data_table.loc[
                    data_table.index[i - 1], col + "_freq_weighted"
                ] = float(np.sum(freqs * amplitudes) / total_amplitude)

                psd = np.square(amplitudes) / len(amplitudes)
                total_psd = np.sum(psd)
                if total_psd <= 0 or not np.isfinite(total_psd):
                    continue

                psd_pdf = psd / total_psd
                nonzero_probabilities = psd_pdf[psd_pdf > 0]
                data_table.loc[data_table.index[i - 1], col + "_pse"] = -np.sum(
                    nonzero_probabilities * np.log(nonzero_probabilities)
                )

        return data_table
