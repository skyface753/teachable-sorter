#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if grep -s -q "Raspberry Pi" /sys/firmware/devicetree/base/model; then
    echo "Installing Pi's Sorter Dependencies"
    pip3 install -r requirements.txt
    pip3 install opencv-python opencv-contrib-python
    sudo apt-get install libusb-1.0-0
	echo "For FLIR support, complete the installation instructions listed in the README" 
else
    echo "Platform not supported"
fi 
