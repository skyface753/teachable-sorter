# Teachable Sorter

This project is a sorter for objects using a camera and a Coral USB Accelerator. The sorter can be trained with the Teachable Machine and then used to sort objects.

## Activate the Environment

To activate the previously created environment, use the following command on the `sorter` machine:

> Note: To create a new environment, refer to the [install on Raspi section](#install-on-raspi).

```bash
pyenv activate venv_myprojecta
```

## Train

Refer to the [official documentation](https://coral.ai/projects/teachable-sorter#step-4-connect-to-teachable-machine) for a detailed explanation.

### Prepare the SSH Connection

SSH to the Raspi with a port forward:

```bash
ssh -L 8889:localhost:8889 pi@sorter.it-lab.cc -i ~/.ssh/sorter
```

> Note: The key is not included in the repository.

### Start the Sorter in Train Mode

Activate the environment:

```bash
pyenv activate venv_myprojecta
```

Start the training:

```bash
cd teachable-sorter/Sorter
python sorter.py --train --flir --biquad
```

### Open the Teachable Machine with the Port Forward

Open teachable machine in the browser:

```bash
https://teachablemachine.withgoogle.com
```

Create a new project and add the following to the url

```bash
?network=true
```

Now you can train the model with the images from the camera.

#### Pretrained models by Skyface753

Coins:
https://teachablemachine.withgoogle.com/train/image/1FsMQauZdUvkO1DCY1AiGVlOUeKQYNtEN

Balls:
https://teachablemachine.withgoogle.com/train/image/1onSVriAxYLNihy_weE4JKnwiaLM7AR-0

Colab Train:
https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_classification_ptq_tf2.ipynb

### Save the Model

Save as `model_edgetpu.tflite` and `labels.txt` in root directory.

## Run the Sorter

### Activate the Environment

```bash
pyenv activate venv_myprojecta
```

### Run the Sorter

```bash
python sorter.py --sort --flir --biquad
```

You can also use `--debug` to show the debug display.

## Arguments

| Argument          | Description                               |
| ----------------- | ----------------------------------------- |
| --sort            | Sort the objects (counterpart to --train) |
| --train           | Train the model (counterpart to --sort)   |
| --flir            | Use the Flir camera                       |
| --opencv          | Use the OpenCV camera                     |
| --biquad          | Use the Biquad filter for sorting         |
| --biquad2         | Use the Biquad2 filter for sorting        |
| --center_of_mass  | Use the center of mass for sorting        |
| --zone-activation | Use the zone activation for sorting       |
| --debug           | Show debug display                        |

## Eval

```bash
python eval.py --model model_edgetpu.tflite --dataset dataset/test/ --labels labels.txt
```

## Install on Raspi

Install pyenv and python 3.9

```bash
curl -fsSL https://pyenv.run | bash

# füge die sachen ein die in der console ausgegeben werden
nano ~/.bashrc

# install dependencies
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Restart shell

```bash
pyenv install 3.8.20
pyenv virtualenv 3.8.20 sorter
pyenv activate sorter
```

Install Spinnaker

```bash
# dependencies
sudo apt-get install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools;

# for python
wget https://flir.netx.net/file/asset/68780/original/attachment -O spinnaker_python-3.8.tar.gz
mkdir spinnaker_python-3.8
tar -xzvf spinnaker_python-3.8.tar.gz -C spinnaker_python-3.8
cd spinnaker_python-3.8
pip install spinnaker_python-4.2.0.46-cp38-cp38-linux_aarch64.whl

# ubuntu arm
wget https://flir.netx.net/file/asset/68778/original/attachment -O spinnaker_sdk_ubuntu-20.04.tar.gz
mkdir spinnaker_sdk_ubuntu-20.04
tar -xzvf spinnaker_sdk_ubuntu-20.04.tar.gz -C spinnaker_sdk_ubuntu-20.04
cd spinnaker_sdk_ubuntu-20.04/spinnaker-4.2.0.46-arm64

# comment out the following lines in install_spinnaker_arm.sh
nano install_spinnaker_arm.sh
# sudo dpkg -i spinview-qt_*.deb
# sudo dpkg -i spinview-qt-dev_*.deb

sudo bash install_spinnaker_arm.sh
```

Install pycoral

```bash
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

Sorter

```bash
git clone https://github.com/skyface753/teachable-sorter.git
cd teachable-sorter
pip install -r a.txt
```
