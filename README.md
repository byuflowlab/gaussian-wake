GaussianWake is a wake and wind farm model for horizontal-axis wind turbines based on a Gaussian distribution.
The model is primarily an implementation of the model presented by Bastankhah and Porte Agel (2016, 2014). The
Farm model includes elements of work by Niayifar and Porte Agel (2016, 2015) and Crespo and Hernandez (1999),
along with options to use other wake combination methods and local turbulence intensity calculations.

The default options include analytic gradients obtained through algorithmic differentiation via Tapenade.

Dependencies: OpenMDAO >= 1.7.3
