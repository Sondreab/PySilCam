# -*- coding: utf-8 -*-
import sys
import time
import logging
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cProfile
import pstats
from io import StringIO
from pysilcam import __version__
from pysilcam.acquisition import Acquire
from pysilcam.background import backgrounder
import pysilcam.process
from pysilcam.process import statextract
import pysilcam.postprocess as sc_pp
import pysilcam.plotting as scplt
import pysilcam.datalogger as datalogger
import pysilcam.oilgas as scog
from pysilcam.config import PySilcamSettings
from skimage import color
import imageio
import os
import pysilcam.silcam_classify as sccl
import multiprocessing
from multiprocessing.managers import BaseManager
from queue import LifoQueue
import psutil

title = '''
 ____        ____  _ _  ____
|  _ \ _   _/ ___|(_) |/ ___|__ _ _ __ ___
| |_) | | | \___ \| | | |   / _` | '_ ` _ \
|  __/| |_| |___) | | | |__| (_| | | | | | |
|_|    \__, |____/|_|_|\____\__,_|_| |_| |_|
       |___/
'''

def check_path(filename):
   file = os.path.normpath(filename)
   path = os.path.dirname(file)
   if path:
      if not os.path.isdir(path):
         try:
            os.makedirs(path)
         except:
            print('Could not create catalog:',path)

def configure_logger(settings):
    '''
    Description of the function.

    Args:
        param1 (type): description
        param2 (type): description
    Returns:
        type: description
    '''
    if settings.logfile:
        check_path(settings.logfile)
        logging.basicConfig(filename=settings.logfile,
                            level=getattr(logging, settings.loglevel))
    else:
        logging.basicConfig(level=getattr(logging, settings.loglevel))


def silcam():
    '''Aquire/process images from the SilCam

    Usage:
      silcam acquire <datapath>
      silcam process <configfile> <datapath> [--nbimages=<number of images>] [--nomultiproc]
      silcam realtime <configfile> <datapath> [--discwrite] [--nomultiproc]
      silcam -h | --help
      silcam --version

    Arguments:
        acquire     Acquire images
        process     Process images
        realtime    Acquire images from the camera and process them in real time

    Options:
      --nbimages=<number of images>     Number of images to process.
      --discwrite                       Write images to disc.
      --nomultiproc                     Deactivate multiprocessing.
      -h --help                         Show this screen.
      --version                         Show version.

    '''
    print(title)
    print('')
    args = docopt(silcam.__doc__, version='PySilCam {0}'.format(__version__))


    if args['<datapath>']:
        # The following is solving problems in transfering arguments from shell on windows
        # Remove ' characters
        datapath = os.path.normpath(args['<datapath>'].replace("'",""))
        # Remove " characters at the end (occurs when user give \" at the end)
        while datapath[-1] == '"':
            datapath = datapath[:-1]

    # this is the standard processing method under development now
    if args['process']:
        multiProcess = True
        if args['--nomultiproc']:
            multiProcess = False
        nbImages = args['--nbimages']
        if (nbImages != None):
            try:
                nbImages = int(nbImages)
            except ValueError:
                print('Expected type int for --nbimages.')
                sys.exit(0)
        silcam_process(args['<configfile>'] ,datapath, multiProcess=multiProcess, realtime=False, nbImages=nbImages)

    elif args['acquire']: # this is the standard acquisition method under development now
        silcam_acquire(args['<datapath>'])

    elif args['realtime']:
        discWrite = False
        if args['--discwrite']:
            discWrite = True
        multiProcess = True
        if args['--nomultiproc']:
            multiProcess = False
        silcam_process(args['<configfile>'], datapath, multiProcess=multiProcess, realtime=True, discWrite=discWrite)

def silcam_acquire(datapath):
    acq = Acquire(USE_PYMBA=True) # ini class
    t1 = time.time()
    aqgen = acq.get_generator()
    for i, (timestamp, imraw) in enumerate(aqgen):
        filename = os.path.join(datapath, timestamp.strftime('D%Y%m%dT%H%M%S.%f.silc'))
        with open(filename, 'wb') as fh:
            np.save(fh, imraw, allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
        print('Written', filename)

        t2 = time.time()
        aq_freq = np.round(1.0/(t2 - t1), 1)
        requested_freq = 16.0
        rest_time = (1 / requested_freq) - (1 / aq_freq)
        rest_time = np.max([rest_time, 0.])
        time.sleep(rest_time)
        actual_aq_freq = 1/(1/aq_freq + rest_time)
        print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, actual_aq_freq))
        t1 = time.time()

# the standard processing method under active development
def silcam_process(config_filename, datapath, multiProcess=True, realtime=False, discWrite=False, nbImages=None, gui=None):

    '''Run processing of SilCam images

    The goal is to make this as fast as possible so it can be used in real-time

    Function requires the filename (including path) of the config.ini file
    which contains the processing settings

    '''
    print(config_filename)

    print('PROCESS MODE')
    print('')
    #---- SETUP ----

    #Load the configuration, create settings object
    settings = PySilcamSettings(config_filename)

    #Print configuration to screen
    print('---- CONFIGURATION ----\n')
    settings.config.write(sys.stdout)
    print('-----------------------\n')

    #Configure logging
    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.silcam_process')

    logger.info('Processing path: ' + datapath)

    #Initialize the image acquisition generator
    aq = Acquire(USE_PYMBA=realtime)
    aqgen = aq.get_generator(datapath, writeToDisk=discWrite)

    #Get number of images to use for background correction from config
    print('* Initializing background image handler')
    bggen = backgrounder(settings.Background.num_images, aqgen, bad_lighting_limit = settings.Process.bad_lighting_limit)

    # make datafilename autogenerated for easier batch processing
    if (not os.path.isdir(settings.General.datafile)):
       logger.info('Folder ' + settings.General.datafile + ' was not found and is created')
       os.mkdir(settings.General.datafile)

    procfoldername = os.path.split(datapath)[-1]
    datafilename = os.path.join(settings.General.datafile,procfoldername)
    logger.info('output stats to: ' + datafilename)

    if os.path.isfile(datafilename + '-STATS.csv'):
        logger.info('removing: ' + datafilename + '-STATS.csv')
        os.remove(datafilename + '-STATS.csv')

    # Create export directory if needed
    if settings.ExportParticles.export_images:
       if (not os.path.isdir(settings.ExportParticles.outputpath)):
          logger.info('Export folder ' + settings.ExportParticles.outputpath + ' was not found and is created')
          os.mkdir(settings.ExportParticles.outputpath)

    #---- END SETUP ----

    #---- RUN PROCESSING ----


    # If only one core is available, no multiprocessing will be done
    multiProcess = multiProcess and (multiprocessing.cpu_count() > 1)

    print('* Commencing image acquisition and processing')

    rts = scog.rt_stats(settings)

    if (multiProcess):
        proc_list = []
        mem = psutil.virtual_memory()
        memAvailableMb = mem.available >> 20
                
        inputQueue, outputQueue = defineQueues(realtime, int(memAvailableMb / 2 * 1/15))
      
        distributor(inputQueue, outputQueue, config_filename, proc_list, gui)

        # iterate on the bggen generator to obtain images
        for i, (timestamp, imc) in enumerate(bggen):
            # handle errors if the loop function fails for any reason
            if (nbImages != None):
                if (nbImages <= i):
                    break
            try:
                inputQueue.put_nowait((i, timestamp, imc)) # the tuple (i, timestamp, imc) is added to the inputQueue
            except:
                continue
            # write the images that are available for the moment into the csv file
            collector(inputQueue, outputQueue, datafilename, proc_list, False,
                      settings, rts=rts)

            if not gui==None:
                while (gui.qsize() > 0):
                    try:
                        gui.get_nowait()
                        time.sleep(0.001)
                    except:
                        continue
                #try:
                rtdict = dict()
                rtdict = {'dias': rts.dias,
                        'vd_oil': rts.vd_oil,
                        'vd_gas': rts.vd_gas,
                        'oil_d50': rts.oil_d50,
                        'gas_d50': rts.gas_d50}
                gui.put_nowait((timestamp, imc, rtdict))
                #except:
                #    continue

        if (not realtime):
            for p in proc_list:
                inputQueue.put(None)

        # some images might still be waiting to be written to the csv file
        collector(inputQueue, outputQueue, datafilename, proc_list, True,
                  settings, rts=rts)

        for p in proc_list:
            p.join()
            print ('%s.exitcode = %s' % (p.name, p.exitcode) )

    else:
        # load the model for particle classification and keep it for later
        nnmodel = []
        nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)

        # iterate on the bggen generator to obtain images
        for i, (timestamp, imc) in enumerate(bggen):
            # handle errors if the loop function fails for any reason
            if (nbImages != None):
                if (nbImages <= i):
                    break

            image = (i, timestamp, imc)
            # one single image is processed at a time
            stats_all = processImage(nnmodel, class_labels, image, settings, logger, gui)

            if (not stats_all is None): # if frame processed 
                # write the image into the csv file
                writeCSV( datafilename, stats_all)

    #---- END ----

def addToQueue(realtime, inputQueue, i, timestamp, imc):
    '''
    Put a new image into the Queue.

    Args:
        realtime: boolean indicating wether the processing is done in realtime
        inputQueue: queue where the images are added for processing
        i: index of the image acquired
        timestamp: timestqmp of the acquired image
        imc: corrected image
    '''
    if (realtime):
        try:
            inputQueue.put_nowait((i, timestamp, imc))
        except:
            pass
    else:
        inputQueue.put((i, timestamp, imc))

def defineQueues(realtime, size):
    '''
    Define the input and output queues depending on wether we are in realtime mode

    Args:
        realtime: boolean indicating wether the processing is done in realtime
        size: max size of the queue

    Returns:
        inputQueue
        outputQueue
    '''
    createQueues = createLIFOQueues if realtime else createFIFOQueues
    return createQueues(size)

def createLIFOQueues(size):
    '''
    Create a LIFOQueue (Last In First Out)

    Args:
        size: max size of the queue

    Returns:
        inputQueue
        outputQueue
    '''
    manager = MyManager()
    manager.start()
    inputQueue = manager.LifoQueue(size)
    outputQueue = manager.LifoQueue(size)
    return inputQueue, outputQueue

def createFIFOQueues(size):
    '''
    Create a FIFOQueue (First In First Out)

    Args:
        size: max size of the queue

    Returns:
        inputQueue
        outputQueue
    '''
    inputQueue = multiprocessing.Queue(size)
    outputQueue = multiprocessing.Queue(size)
    return inputQueue, outputQueue

class MyManager(BaseManager):
    ''' 
    Customized manager class used to register LifoQueues
    '''
    pass

MyManager.register('LifoQueue', LifoQueue)


def processImage(nnmodel, class_labels, image, settings, logger, gui):
    '''
    Proceses an image
    '''
    try:
        i = image[0]
        timestamp = image[1]
        imc = image[2]

        #time the full acquisition and processing loop
        start_time = time.clock()

        logger.info('Processing time stamp {0}'.format(timestamp))

        #Calculate particle statistics
        stats_all, imbw, saturation = statextract(imc, settings, timestamp,
                                                  nnmodel, class_labels)

        # if there are not particles identified, assume zero concentration.
        # This means that the data should indicate that a 'good' image was
        # obtained, without any particles. Therefore fill all values with nans
        # and add the image timestamp
        if len(stats_all) == 0:
            print('ZERO particles idenfitied')
            z = np.zeros(len(stats_all.columns)) * np.nan
            stats_all.loc[0] = z
            # 'export name' should not be nan because then this column of the
            # DataFrame will contain multiple types, so label with string instead
            if settings.ExportParticles.export_images:
                stats_all['export name'] = 'not_exported'

        # add timestamp to each row of particle statistics
        stats_all['timestamp'] = timestamp

        #Time the particle statistics processing step
        proc_time = time.clock() - start_time

        #Print timing information for this iteration
        infostr = '  Image {0} processed in {1:.2f} sec ({2:.1f} Hz). '
        infostr = infostr.format(i, proc_time, 1.0/proc_time)
        print(infostr)

        #---- END MAIN PROCESSING LOOP ----
        #---- DO SOME ADMIN ----

    except:
        infostr = 'Failed to process frame {0}, skipping.'.format(i)
        logger.warning(infostr, exc_info=True)
        print(infostr)
        return None

    return stats_all


def loop(config_filename, inputQueue, outputQueue, gui=None):
    '''
    Main processing loop, run for each image
    '''
    settings = PySilcamSettings(config_filename)
    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.silcam_process')

    # load the model for particle classification and keep it for later
    nnmodel = []
    nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)

    while True:
        task = inputQueue.get()
        if task is None:
            outputQueue.put(None)
            break
        stats_all = processImage(nnmodel, class_labels, task, settings, logger, gui)

        if not stats_all is None:
            outputQueue.put(stats_all)


def distributor(inputQueue, outputQueue, config_filename, proc_list, gui=None):
    '''
    distributes the images in the input queue to the different loop processes
    '''
    
    numCores = max(1, multiprocessing.cpu_count() - 2)

    for nbCore in range(numCores):
        proc = multiprocessing.Process(target=loop, args=(config_filename, inputQueue, outputQueue, gui))
        proc_list.append(proc)
        proc.start()

def collector(inputQueue, outputQueue, datafilename, proc_list, testInputQueue,
        settings, rts=None):
    '''
    collects all the results and write them into the stats.csv file
    '''

    countProcessFinished = 0

    while ((outputQueue.qsize()>0) or (testInputQueue and inputQueue.qsize()>0)):

        task = outputQueue.get()

        if (task is None):
            countProcessFinished = countProcessFinished + 1
            if (len(proc_list) == 0): # no multiprocessing
                break
            # The collector can be stopped only after all loop processes are finished
            elif (countProcessFinished == len(proc_list)):
                break
            continue

        writeCSV(datafilename, task)
        collect_rts(settings, rts, task)


def collect_rts(settings, rts, stats_all):
    if settings.Process.real_time_stats:
        try:
            rts.stats = rts.stats().append(stats_all)
        except:
            rts.stats = rts.stats.append(stats_all)
        rts.update()
        filename = os.path.join(settings.General.datafile,
                'OilGasd50.csv')
        rts.to_csv(filename)


def writeCSV(datafilename, stats_all):
    '''
    writes into the csv ouput file
    '''

    # create or append particle statistics to output file
    # if the output file does not already exist, create it
    # otherwise data will be appended
    # @todo accidentally appending to an existing file could be dangerous
    # because data will be duplicated (and concentrations would therefore
    # double)
    if not os.path.isfile(datafilename + '-STATS.csv'):
        stats_all.to_csv(datafilename +
                '-STATS.csv', index_label='particle index')
    else:
        stats_all.to_csv(datafilename + '-STATS.csv',
                mode='a', header=False)


def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
