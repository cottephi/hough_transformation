# Data Generator

Side repository to generate data containing lines for the Hough Transformation to find

The generated data are in the HDF5 format, with the follwing fields:

 * data: contains the data to give to the Hough Transformation
 * lines: the lines that are expected to be found in the data. It is a list of
   tuples (theta, r, s_theta, s_r) where theta and r and the Hough coordinates
   and $s_{\theta}$ and $s_r$ are the associated uncertainties.
