### This is the code for dynamic dp

To run the code, you should install opacus version 0.14.0 by

pip install opacus==0.14.0

Then, to implement dynamic dp, you should replace the privacy_engine.py file in the opacus package by the one in the utils folder of this repo. Or, you can just copy and paste the last two functions in the 
utils.privacy_engine.py: set_clip and set_unit_sigma into the opacus package.

Then, you can just run the code simply by

python dp_mnist.py

