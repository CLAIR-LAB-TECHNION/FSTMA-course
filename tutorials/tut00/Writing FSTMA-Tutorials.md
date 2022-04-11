<img src="images/clippy.png" align="center"/>

# Introduction

In FSTMA, the students write the tutorials! Every week, two topics will be presented by two students. The goal is to learn about different sequential decision making algorithms by documenting them in detail with live code demonstrations and implementations, combining parctice and theory into a single document. Below are a few guidelines on how to write these tutorials and present them in class.


# Submission Format

## Tutorial files
IPython notebooks allow us to demonstrate live code with formatted text, including $\LaTeX$ style math, making it the ideal tool for presenting both the theoretical and the practical aspects of these algorithms. Each topic must be presented in a separate notebook (unless otherwise approved). All content must appear in the IPython notebook. Powerpoint presentations, PDF slides, or any other file formats **WILL NOT BE ACCEPTED**.


## Files structure
All content should be saved in a directory named `tutXX` where `XX` is the two digit tutorial number (e.g. `tut05`, `tut11`, etc.). This directory should contain:
* Topic1.ipynb - tutorial notebook for topic 1 (choose a fitting filename)
* Topic2.ipynb - tutorial notebook for topic 2 (choose a fitting filename)
* environment.yml - the anaconda environment file.
* (OPTIONAL) - `.py` code files
* (OPTIONAL) Required resources - e.g., reasonably sized images, data and model weights (100mb is the GitHub file size limit).
* (OPTIONAL) .gitignore - if you wish to not upload result outputs, builds, or other unwanted files.

The directory should be tidy, and files and directories should have appropriate names. `.py` files and resources should be packaged in separate directories. You can (and should if necessary) crete multiple code packages and files to help with code understandability.


# Notebook content
The `tut02/ImitationLearning.ipynb` notebook is an example of what a tutorial notebook should look like, both in the level of theoretical and implementation detail. Most importantly, your notebooks should be detailed and clearly written such that a person familiar with the subjects presented in the lectures will be able to understand the presented topic without having attended the tutorial in class. Below we break down the notebook content requirements.


## Presented Environemnts
For each topic, the student must implement the presented method for both a discrete space and a continuous space domain. The spaces we are using are:
* Single-agent:
    * `pettingzoo.mpe.simple_v2` (continuous)
    * MultiTaxi with a single taxi. (discrete)
    * (APPROVED TOPICS ONLY) gym's Taxi_v3 evnironment (discrete)
* Multi-agent:
    * `pettingzoo.mpe.simple_speaker_listener_v3` (continuous)
    * MultiTaxi (discrete)

When writing the notebook, choose one environment that will receive main focus and will guide the reader throughout the tutorial. We will call this the **main environment**. The other environment is used to showcase the transition (or lack there of) from one kind of space to another (continuous to discrete or vice versa). We call this the **seconddary environment**. Ideally, the week's topics will be presented with different main environments, though this is not a requirement.

## Main Content
The notebook should contain (but is not limitted to) the following:
* an introduction of the topic
* the world model and problem formulation
* a detailed description of the solution you are presenting
* key code blocks used in the solution implementation
* method weaknesses, issues, and failures
* A working example on a chosen environment to present (main environment)
    * should guide the tutorial
    * display results and visualizations (plots, environment renderings, etc.)
* An additional working example on the other environment (secondary environment)
    * with less explanations than the main environment
    * should highlight the differences in the method when moving from a continuous environment to a discrete environment (and vise versa).
    * display results and visualizations.

To reduce clutter within the notebook, create a code cell containing all imports, constants, and globals at the top of the notebook.

## Implementations
You are asked to make the code **AS GENERIC AS POSSIBLE**. The algorithms you are presenting are domain-independent, meaning they support any environment under the assumption of some unified API (e.g., pettingzoo). Therfore, generic code will be **reusable** for other domains that follow this API. A good test of code generality is to be able to run the same code on both presented environments. This might not always be possible using the raw environments, but one can use wrpapers to overcome this.

Implementations of core algorithms in your presented topic should be written directly in the notebook as they are the main source of practical information in the notebook. However, if a code block you wish to present has already been written in a previous tutorial notebook, you need not present it yourself. Instead, cite the previous tutorial and encapsulate this code in `.py` files.

Your code **will be scrutinized**, and so, to help us understand your code more easily, it must be rogorously documented. The level of documentation should match that of Tutorial 2. Note that external `.py` file code should be documented in even more detail since it is not surrounded by formatted text explanations as in the notebook.


## Using other people's code
Many (if not all) of the algorithms covered in the tutorials are already implemented and are freely accessible on the web. We encourage students to save precious time by integrating code from the web into the tutorial. However, do not use code you do not understand and do not forget to cite / link everything! Make sure this does not make your code less generic. A good starting point is the [AI Agents](https://github.com/sarah-keren/AI_agents) repository.


# Tutorial presentation
Each topic is alloted 25 minutes (exactly half) of the tutorial. This is not enough time to discuss all the contents of the notebook in detail. Therefore, you must choose which sections of the notebook you will focus on when presenting your topic.

To enable students to follow along with the presentation, presenting students are asked to submit their tutorial files on the day before the day of the tutorial. This will give the course staff time to run it and upload it to the course repository for everyone to download.


# Anaconda environment file
In this course, we recomend running code locally. Though it is tempting to use a ready-to-go tool like google colab, we must remember that running code remotely can have its limitations. Simply put, not everything works when using these systems, e.g., `pettingzoo.mpe` environments require a connection to a screen in order to render themselves.

The biggest issue with running code locally is that python environments can act differently from each other, especially across platforms. We resolve this by using anaocnda for cross-platform python environment management. To support this in your tutorials, each `tutXX` directory must contain an `environment.yml` file that lists the required packages to run the tutorial. For multi-platform "friendliness" we must make the environment file as minimal as possible. Please read "Anaconda Crash Course" to see how to do this correctly.


# Recap
The following are some of the main bullet points from above that we wish to emphasize:
* Every week, 2 topics are presented by students (25 mins each)
* must be written in python using IPython notebooks
* the gym's single-taxi environment is not to be used (unless instructed to do so by the staff).
    * use only approved environments from the list in a previous section.
* must present both environemnts in the notebook.
    * one is the main environment and should be discussed in full
    * the other is the secondary environment, which should present a working example and highlight the differences between the two environments.
* use Tutorial 2 as an example.
* implementations must be generic and well documented
* code must be well documenteed.
* cite / link other people's code if used.
* work locally and provide environment.yml
* **We want you to succeed**! Contact the staff for assistance.
