##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 22-12-2022

import numpy as np
import scipy.stats as stats

# Class to abstract a history of numerical values we can use as an attribute.
class NumericalAbstraction:

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std').
    def aggregate_value(self, aggregation_function):
        aggregations = {
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "median": np.median,
            "std": np.std,
        }
        try:
            return aggregations[aggregation_function]
        except KeyError as error:
            supported = ", ".join(aggregations)
            raise ValueError(
                f"Unsupported aggregation function {aggregation_function!r}. "
                f"Choose one of: {supported}."
            ) from error

    # Abstract numerical columns specified given a window size (i.e. the number of time points from
    # the past considered) and an aggregation function.
    def abstract_numerical(self, data_table, cols, window_size, aggregation_function):
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        aggregate = self.aggregate_value(aggregation_function)

        # Create new columns for the temporal data, pass over the dataset and compute values.
        for col in cols:
            data_table[
                col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            ] = data_table[col].rolling(window_size).apply(aggregate)

        return data_table
