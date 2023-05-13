This repository has all of the jupyter notebooks from the eyetracking model URSI project in the summer of 2022 as well as the raw python files that accompany them. The python files still have the cell function of jupyter notebooks but can also be run irrespective of this in a terminal window. There is some room to improve the scripts organizationally. This could be done by putting classes at the top followed by functions, followed by the execution of those functions. All import commands could be put at the top and code could be more detailed. Additionally, there is not a standard indentation style across files. This could be fixed using Python’s Black formatter. These are just suggestions to make understanding the code easier for people that are coming back to a script after awhile or are looking at it for the first time. Functionally, the models are written to be run on google drive and need to be changed to run on a personal machine or hopper (basically vassar’s supercomputer). 

The repo is made up of models, experiments, and data processing scripts.
 
Experiments: tests how effective the models are at getting the right calibration points
Gaze Prediction Models: uses VAE encoded face points and eye position to guess gaze of the subject
Pipeline: processes raw video files as json files, then configures VAE to the json
Preprocess Data: processes, cleans, and sorts json data
Webgazer Models: non-ML regression models for gaze prediction
VAE: configures VAE to recognize head/body position

jupyter_converter.py makes copies of all jupyter scripts and converts them to python scripts in a separate directory in case you need to do convert any other jupyter scripts.

If any of this information doesn’t match the scripts, feel free to change it.

