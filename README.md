# Superseded by FLOWFarm.jl

GaussianWake is a wake and wind farm model for horizontal-axis wind turbines based on a Gaussian distribution. The model is primarily an implementation of the model presented by Bastankhah and Porte Agel (2016, 2014). The Farm model includes elements of work by Niayifar and Porte Agel (2016, 2015) and Crespo and Hernandez (1999), along with options to use other wake combination methods and local turbulence intensity calculations.

This wake model is compatible with Wake Expansion Continuation (WEC). If you use this functionality, please cite Thomas, J. J., and Ning, A., “A Method for Reducing Multi-Modality in the Wind Farm Layout Optimization Problem,” Journal of Physics: Conference Series, Vol. 1037, No. 042012, Milano, Italy, The Science of Making Torque from Wind, Jun. 2018. doi:10.1088/1742-6596/1037/4/042012

The default options include analytic gradients obtained through algorithmic differentiation via Tapenade.

Dependencies: OpenMDAO v1.7.4

Install in bash using $python setup.py install
