# Introduction
Welcome our course "From Single-Agent to Multi-Agent", abbreviated FSTMA, where you will learn the theoretical aspects of various single-agent and multi-agent AI frameworks, as well as practical work with different systems using Python based implementations. This tutorial contains guides to all the tools required to successfully run and implement a tutorial on your local machine. The main tools we will be using are Anaconda, Jupyter Lab, and OpenAI Gym based environments, each of which has a separate demo written as jupyter notebooks.

# Installing python w/ anaconda
In this course, we will use miniconda to define isolated python environments. To install miniconda, follow the installation instructions in the `AnacondaCrashCourse` file (provided as a PDF file for your convenience). This will enable us to run `conda` commands from the terminal (Windows users must use the "Anaconda Prompt" app).

# Installing the conda environment
Once we have a terminal with an active anaconda environment, we can install the tutorial 1 environment. To do this, navigate in your terminal to the tut01 directory and run the command:
```
conda env update -f environment.yml
```
This will install all the requirements and create an environment called `fstma-tut01`. To activate this environment, run
```
conda activate fstma-tut01
```

# Running jupyter lab
The `fstma-tut01` environment was installed with the `jupyterlab` package, meaning that we can run the `jupyter` command from our terminal when this environment is active. To open Jupyter Lab and view the tutorial notebooks, navigate in your terminal to the tut01 directory, make sure `fstma-tut01` has been activated, and run the command:
```
jupyter lab
```
This will run an instance of the jupyter server on the first available port starting from 8888. Once the server has been initialized, a new tab should be opened on your default web browser displaying jupyter lab. If not, then the server output to your terminal should contain severl links that you can try in your browser. If all else fails, you can try navigating to http://localhost:8888/lab/, http://localhost:8889/lab/, http://localhost:8890/lab/, etc.

# What next?
If anaconda has been installed successfully, the environment has been created with all dependencies, and jupyter lab is running, we are ready to run the tutorial. in the file browser, accessible via the folder-shaped button in the top left corner of the jupyter lab interface, open the notebooks (`.ipynb`) files with a double-click and start running. The recommended order for the notebooks is:
1. `JupyterLabDemo.ipynb` - a short demo of jupyter lab and jupyter notebooks.
2. `MultiTaxiEnvDemo.ipynb` - a demo of the multi taxi environment.
3. `PettingZooDemo.ipynb` - a demo of the speaker-listener environment implemented in PettingZoo.

Every tutorial must have a single `environment.yml` file that installs an environment capable of running the tutorial code without error. Thus, before writing a tutorial, it is important to go over the `AnacondaCrashCourse` (available as PDF and markdown files) in detail and understand how to do this in a clean and corss-platform-friendly way.
