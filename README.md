GaussianWake is a wake and wind farm model for horizontal-axis wind turbines based on a Gaussian distribution.
The model is primarily an implementation of the model presented by Bastankhah and Porte Agel (2016, 2014). The
Farm model includes elements of work by Niayifar and Porte Agel (2016, 2015) and Crespo and Hernandez (1999),
along with options to use other wake combination methods and local turbulence intensity calculations.

This wake model is compatible with Wake Expansion Continuation (WEC). If you use this functionality, please cite Thomas, J. J., Annoni, J., Fleming, P., and Ning, A., “Comparison of Wind Farm Layout Optimization Results Using a Simple Wake Model and Gradient-Based Optimization to Large-Eddy Simulations,” AIAA Scitech 2019 Forum, San Diego, CA, Jan. 2019. doi:10.2514/6.2019-0538

The default options include analytic gradients obtained through algorithmic differentiation via Tapenade.

Dependencies: OpenMDAO  v1.7.4

Install in bash using $python setup.py install
