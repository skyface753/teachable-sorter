## Train

Coins:
https://teachablemachine.withgoogle.com/train/image/1FsMQauZdUvkO1DCY1AiGVlOUeKQYNtEN

Balls:
https://teachablemachine.withgoogle.com/train/image/1onSVriAxYLNihy_weE4JKnwiaLM7AR-0

Colab Train:
https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_classification_ptq_tf2.ipynb

Save as `model_edgetpu.tflite` and `labels.txt` in root directory.

## Eval

```bash
python eval.py --model model_edgetpu.tflite --dataset dataset/test/ --labels labels.txt
```

## Run Sorter

```bash
cd Sorter
python sorter.py --opencv --biquad
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

## Run on Raspi

```bash
cd teachable-sorter
cd Sorter
pyenv activate sorter
python sorter.py --flir
```
