# AOI_FHG
## Work on state-of-the-art Deep Learning neural networks to help to improve the performance of an object detection model based on a Convolutional Neural Network (CNN).
## The model is used to automatically inspect electronic components on circuit boards. 

### This work will include:

- Learning about the theory of CNNs for object detection
- Data collection & extension of an existing data set with a semi-automatic labelling tool
- Training of a CNN model with the extended data
- Evaluation of model performance (Accuracy, Precision, Recall etc. ) and comparing it to previous models
- Development of an additional classification module to analyse the connectivity of the electric components.

## Setup
- [Google Chrome](https://www.google.com/chrome/?brand=BNSD&gclid=CjwKCAjwmqKJBhAWEiwAMvGt6KJUMk5ORuRLC-lSw6ou9Whsrg739TgL-19-Dm_2QCMPr1c6snRnvBoCLw8QAvD_BwE&gclsrc=aw.ds)
- Editor (VS Code)
- Git: `sudo apt install git-all`
- Python
    - `sudo apt update`
    - `sudo apt install python-is-python3`
    - install the dependencies from [pyenv - Suggested build environment](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)
    - install pyenv using [pyenv installer](https://github.com/pyenv/pyenv-installer)
    - don't forget to setup '.profile' and '.bashrc' as mentioned in the terminal after install
    - check correct install: `pyenv virtualenv --version`
    - For Windows how to activate the env `.\(nameofenv)\Scripts\activate.bat`
    - Installation of virtualenv ^^ https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html
    - look at available python versions: `pyenv install --list`
    - install latest python 3.8: e.g. `pyenv install 3.8.11`
- CUDA
    - follow the steps in [Installing Multiple CUDA & cuDNN Versions in Ubuntu](https://towardsdatascience.com/installing-multiple-cuda-cudnn-versions-in-ubuntu-fcb6aa5194e2)
    - do Step 3, 4 for the CUDA version shown in `nvidia-smi` and CUDA 11.1
    - in Step 3 don't forget tor replace last command with the one for the target version
    - for Step 4, you will need to create an Nvidia developer account, or use the file in 'resources' 
- Python Environment
    - `pyenv virtualenv 3.8.11 aoi_demo`
    - create a file '.python-version' in the project folder with the content 'aoi_demo'
    - activate the environment: `pyenv activate aoi_demo` (or open a terminal in project folder after creating '.python-version')
    - run `pip install -r requirements.txt` inside the 'aoi-demo-model-dev' folder
- Iriun (or similar app)
    - install the ubuntu client from [Iriun](https://iriun.com/)
    - install the app on phone
- Docker
    - [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
    - [Install Docker Compose](https://docs.docker.com/compose/install/)
    - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- CVAT
    - [CVAT installation guide](https://openvinotoolkit.github.io/cvat/docs/administration/basics/installation/#quick-installation-guide)
    - skip docker, docker-compose, git as it is already installed

## Learning Material

### Python

- [An Effective Python Environment: Making Yourself at Home](https://realpython.com/effective-python-environment/)
- [How to Use Jupyter Notebook in 2020: A Beginner’s Tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

### Electronics

- [How to Use a Breadboard](https://www.sciencebuddies.org/science-fair-projects/references/how-to-use-a-breadboard)
- [Tutorial 1: Building a Circuit on Breadboard](https://startingelectronics.org/beginners/start-electronics-now/tut1-breadboard-circuits/)
- [Resistor Color Code](https://eepower.com/resistor-guide/resistor-standards-and-codes/resistor-color-code/#)

### Tools

- [Docker](https://docs.docker.com/get-started/overview/)
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/)

### Deep Learning

- [Coursera, Andrew Ng, Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
    - in particular relevant: [Coursera, Andrew Ng, Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)
- [Detectron2 Beginner's Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
- [Digging into Detectron 2](https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd)

## Procedure

### Data Collection

1. come up with a breadboard circuit
2. take several images of the board using the Setup with Iriun, store them in one folder
3. follow the instructions in the Readme inside 'aoi-demo-labeling' to create annotations for the board. In cvat folder:
   - start: docker-compose up -d
   - stop: docker-compose down
   - interface: http://localhost:8080/
   - check running container: docker container ls
4. repeat step 1-3 several times

### Model Development

the code for model development is in 'aoi-demo-model-dev'

- use the script 'plain_train_net.py' or the notebook 'Defect Detection.ipynb' to train the model
- use the notebook 'Defect Detection Inference.ipynb' to test the model
- feel free to change anything inside the script or notebooks to add further analyses or training features

