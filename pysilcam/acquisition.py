# -*- coding: utf-8 -*-
import warnings
import time
import numpy as np
import logging

#Try import pymba, if not available, revert to in-package mockup
try:
    import pymba
except:
    warnings.warn('Pymba not available, using mocked version', ImportWarning)
    print('Pymba not available, using mocked version')
    import pysilcam.fakepymba as pymba


logger = logging.getLogger(__name__)


def _init_camera(vimba):
    '''Initialize the camera system from vimba object'''

    # get system object
    system = vimba.getSystem()

    # list available cameras (after enabling discovery for GigE cameras)
    if system.GeVTLIsPresent:
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        time.sleep(0.2)
    cameraIds = vimba.getCameraIds()
    for cameraId in cameraIds:
        logger.debug('Camera ID:', cameraId)

    #Check that we found a camera, if not, raise an error
    print('')
    print(cameraIds)
    if len(cameraIds) == 0:
        raise RuntimeError('No cameras detected!')
        camera = None
    else:
        # get and open a camera
        camera = vimba.getCamera(cameraIds[0])
        camera.openCamera()

    return camera


def _configure_camera(camera, config=dict()):
    '''Configure the camera.
    
    Config is an optioinal dictionary of parameter-value pairs.
    '''

    #Default settings
    camera.AcquisitionFrameRateAbs = 15
    camera.TriggerSource = 'FixedRate'
    camera.AcquisitionMode = 'SingleFrame'
    camera.ExposureTimeAbs = 150
    camera.PixelFormat = 'BayerRG8'
    camera.StrobeDuration = 150
    camera.StrobeDelay = 0
    camera.StrobeDurationMode = 'Controlled'
    camera.StrobeSource = 'FrameTriggerReady'
    camera.SyncOutPolarity = 'Normal'
    camera.SyncOutSelector = 'SyncOut1'
    camera.SyncOutSource = 'Strobe1'
    
    #camera.GVSPPacketSize = 9194
    camera.GVSPPacketSize = 1500


    #If a config is specified, override those values
    for k, v in config.items():
        setattr(camera, k, v)

    return camera


def _acquire_frame(camera, frame0):
    '''Aquire a single frame in Bayer format'''

    #Aquire single fram from camera
    camera.startCapture()
    frame0.queueFrameCapture()
    camera.runFeatureCommand('AcquisitionStart')
    camera.runFeatureCommand('AcquisitionStop')
    frame0.waitFrameCapture()
    
    #Copy frame data to numpy array (Bayer format)
    #bayer_img = np.ndarray(buffer = frame0.getBufferByteData(),
    #                       dtype = np.uint8,
    #                       shape = (frame0.height, frame0.width, 3))
    img = np.ndarray(buffer = frame0.getBufferByteData(),
                    dtype = np.uint8,
                    shape = (frame0.height, frame0.width, 1))
 
    camera.endCapture()

    return img


def print_camera_config(camera):
    '''Print the camera configuration'''
    config_info_map = {
        'AquisitionFrameRateAbs': 'Frame rate',
        'ExposureTimeAbs': 'Exposure time',
        'PixelFormat': 'PixelFormat', 
        'StrobeDuration': 'StrobeDuration',
        'StrobeDelay': 'StrobeDelay',
        'StrobeDurationMode': 'StrobeDurationMode',
        'StrobeSource': 'StrobeSource',
        'SyncOutPolarity': 'SyncOutPolarity',
        'SyncOutSelector': 'SyncOutSelector',
        'SyncOutSource': 'SyncOutSource',
    }
    
    config_info = '\n'.join(['{0}: {1}'.format(a, camera.getattr(a))
                             for a, b in config_info_map])

    print(config_info)


def acquire():
    '''Aquire images from SilCam'''

    with pymba.Vimba() as vimba:
        #Initialize the camera interface, retry every five seconds if camera not found
        camera = None
        while not camera:
            try:
                camera = _init_camera(vimba)
            except RuntimeError:
                print('Could not connect to camera, sleeping five seconds and then retrying')
                time.sleep(5)

        #Configure camera
        camera = _configure_camera(camera)

        #Prepare for image acquisition and create a frame
        frame0 = camera.getFrame()
        frame0.announceFrame()

        #Aquire raw images and yield to calling context
        try:
            while True:
                img = _acquire_frame(camera, frame0)
                yield img
        finally:
            #Clean up after capture
            camera.revokeAllFrames()
    
            #Close camera
            #@todo


def acquire_rgb():
    '''Aquire images and convert to RGB color space'''
    for img_bayer in acquire():
        #@todo Implement a working Bayer->RGB conversion
        yield img_bayer


def acquire_gray64():
    '''Aquire images and convert to float64 grayscale'''
    for img_bayer in acquire():
        #@todo Implement a working Bayer->grayscale conversion
        imgray = img_bayer[:, :, 0]

        #Yield float64 image
        yield np.float64(imgray)


def acquire_disk():
    '''Aquire images from SilCam and write them to disk.'''
    for count, img in enumerate(acquire()):
        filename = 'data/foo{0}.bmp'.format(count)
        imageio.imwrite(filename, img)
        logger.debug("Stored image image {0} to file {1}.".format(count, filename))
