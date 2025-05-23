# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import PySpin
import cv2
import logging

# initialize logging for this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class FlirBFS(object):
    # takes in a mode, a path to a model, and a callback function that gets called each new frame
    def __init__(self, on_new_frame=None, frame_rate=120, display=False):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.frame_rate = frame_rate
        self.display = display
        self.on_new_frame = on_new_frame
        self.processor = PySpin.ImageProcessor()
        # TODO: SEE THIS Set default image processor color processing method
        self.processor.SetColorProcessing(
            PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        if len(self.cam_list) == 0:
            raise Exception('No FLIR camera connected!')

        self.cam = self.cam_list[0]
        self.cam.Init()

    def get_supported_pixel_formats(self):
        supported_formats = []
        try:
            node_pixel_format = PySpin.CEnumerationPtr(
                self.cam.GetNodeMap().GetNode("PixelFormat"))
            if node_pixel_format is not None and PySpin.IsAvailable(node_pixel_format) and PySpin.IsReadable(node_pixel_format):
                entries = node_pixel_format.GetEntries()
                for entry in entries:
                    entry_symbolic = PySpin.CEnumEntryPtr(entry)
                    if entry_symbolic is not None and PySpin.IsAvailable(entry_symbolic) and PySpin.IsReadable(entry_symbolic):
                        supported_formats.append(entry_symbolic.GetSymbolic())
        except Exception as e:
            print(f"Fehler beim Abrufen der unterstützten Pixelformate: {e}")

        return supported_formats

    # This function pre configures camera settings on the flir.
    def run_cam(self):
        try:

            # self.nodemap_tldevice = cam.GetTLdeviceNodeMap()
            nodemap = self.cam.GetNodeMap()
            stream_nodemap = self.cam.GetTLStreamNodeMap()

            # Configure Camera Settings
            self.cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.AcquisitionFrameRate.SetValue(self.frame_rate)

            handling_mode = PySpin.CEnumerationPtr(
                stream_nodemap.GetNode('StreamBufferHandlingMode'))
            handling_mode_entry = handling_mode.GetEntryByName(
                'NewestOnly')
            handling_mode.SetIntValue(handling_mode_entry.GetValue())

            # From Julian: https://github.com/ITLab-CC/Sorter/blob/main/capture_process.py
            # pixel_format_enum = PySpin.CEnumerationPtr(
            #     self.cam.GetNodeMap().GetNode("PixelFormat"))
            # pixel_format_bayerrg8 = pixel_format_enum.GetEntryByName(
            #     "BayerRG8")
            # pixel_format_enum.SetIntValue(pixel_format_bayerrg8.GetValue())

            ### Set Pixel Format to BGR8 ###
            node_pixel_format = PySpin.CEnumerationPtr(
                nodemap.GetNode('PixelFormat'))
            error = False
            if not PySpin.IsAvailable(node_pixel_format):
                print(
                    'Unavailable Pixel Format. Aborting...')
                error = True
            if not PySpin.IsWritable(node_pixel_format):
                print('Unable to set Pixel Format to BGR8 (enum retrieval). Aborting...')
                error = True
            node_pixel_format_BGR8 = node_pixel_format.GetEntryByName('BGR8')
            if not PySpin.IsAvailable(node_pixel_format_BGR8):
                print(
                    'Pixel Format BGR8 not available. Aborting...')
                error = True
            if not PySpin.IsReadable(node_pixel_format_BGR8):
                print(
                    'Unable to set Pixel Format to BGR8 (entry retrieval). Aborting...')
                error = True
            if not error:
                pixel_format_BGR8 = node_pixel_format_BGR8.GetValue()
                node_pixel_format.SetIntValue(pixel_format_BGR8)

            self.acquire_images(nodemap, stream_nodemap)
        except PySpin.SpinnakerException as ex:
            print('Error {}'.format(ex))

    def acquire_images(self, nodemap, stream_nodemap):

        self.cam.BeginAcquisition()
        # Überprüfe das aktuelle Bildformat
        current_pixel_format = PySpin.CEnumerationPtr(
            self.cam.GetNodeMap().GetNode("PixelFormat")).GetCurrentEntry()
        print("Aktuelles Bildformat:", current_pixel_format.GetSymbolic())
        while True:
            try:
                image_result = self.cam.GetNextImage()
                if image_result.IsIncomplete():
                    print('Image incomplete with image status {} ...'.format(
                        image_result.GetImageStatus()))
                else:
                    # TODO: SEE THIS https://softwareservices.flir.com/Spinnaker/latest/group___camera_defs__h.html#gabd5af55aaa20bcb0644c46241c2cbad1
                    # TODO: SEE THIS https://courses.ideate.cmu.edu/16-375/f2023/Python/theater/hallway-monitor.py
                    open_cv_mat = image_result.GetNDArray()
                    # color_image = self.processor.Convert(image_result,
                    #  PySpin.PixelFormat_BGR8)
                    #open_cv_mat = cv2.cvtColor(
                    #    open_cv_mat, cv2.COLOR_BGR2RGB)
                    image_result.Release()
                    if (self.on_new_frame != None):
                        self.on_new_frame(cv_mat=open_cv_mat)

                    if (self.display == True):
                        cv2.imshow('sorter_camera', open_cv_mat)
                        cv2.waitKey(1)
            except PySpin.SpinnakerException as ex:
                print('Error {}'.format(ex))
                del self.cam

                # Clear camera list before releasing system
                self.cam_list.Clear()

                # Release system instance
                self.system.ReleaseInstance()


if __name__ == "__main__":
    flircam = FlirBFS(display=True)
    flircam.run_cam()
