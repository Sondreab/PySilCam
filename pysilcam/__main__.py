# -*- coding: utf-8 -*-
import sys
import time
import datetime
import logging
from docopt import docopt
import numpy as np
from pysilcam import __version__
from pysilcam.acquisition import Acquire
from pysilcam.background import backgrounder
from pysilcam.process import statextract
import pysilcam.oilgas as scog
from pysilcam.config import PySilcamSettings
import os
import pysilcam.silcam_classify as sccl
import multiprocessing
from multiprocessing.managers import BaseManager
from queue import LifoQueue
import psutil
from shutil import copyfile
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

title = '''
 ____        ____  _ _  ____
|  _ \ _   _/ ___|(_) |/ ___|__ _ _ __ ___
| |_) | | | \___ \| | | |   / _` | '_ ` _ \
|  __/| |_| |___) | | | |__| (_| | | | | | |
|_|    \__, |____/|_|_|\____\__,_|_| |_| |_|
       |___/
'''

def silcam():
    '''Aquire/process images from the SilCam

    Usage:
      silcam acquire <configfile> <datapath>
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
        silcam_acquire(datapath, args['<configfile>'], writeToDisk=True)

    elif args['realtime']:
        discWrite = False
        if args['--discwrite']:
            discWrite = True
        multiProcess = True
        if args['--nomultiproc']:
            multiProcess = False
        silcam_process(args['<configfile>'], datapath, multiProcess=multiProcess, realtime=True, discWrite=discWrite)


def silcam_acquire(datapath, config_filename, writeToDisk=True, gui=None):
    '''Aquire images from the SilCam

    Args:
       datapath              (str)          :  Path to the image storage
       config_filename=None  (str)          :  Camera config file
       writeToDisk=True      (Bool)         :  True will enable writing of raw data to disc
                                               False will disable writing of raw data to disc
       gui=None          (Class object)     :  Queue used to pass information between process thread and GUI
                                               initialised in ProcThread within guicals.py
    '''

    #Load the configuration, create settings object
    settings = PySilcamSettings(config_filename)

    #Print configuration to screen
    print('---- CONFIGURATION ----\n')
    settings.config.write(sys.stdout)
    print('-----------------------\n')

    if (writeToDisk):
        # Copy config file
        configFile2Copy = datetime.datetime.now().strftime('D%Y%m%dT%H%M%S.%f') + os.path.basename(config_filename)
        copyfile(config_filename, os.path.join(datapath, configFile2Copy))

    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.silcam_acquire')

    # update path_length
    updatePathLength(settings, logger)

    acq = Acquire(USE_PYMBA=True) # ini class
    t1 = time.time()

    aqgen = acq.get_generator(datapath, camera_config_file=config_filename, writeToDisk=writeToDisk)

    for i, (timestamp, imraw) in enumerate(aqgen):
        t2 = time.time()
        aq_freq = np.round(1.0/(t2 - t1), 1)
        requested_freq = 16.0
        rest_time = (1 / requested_freq) - (1 / aq_freq)
        rest_time = np.max([rest_time, 0.])
        time.sleep(rest_time)
        actual_aq_freq = 1/(1/aq_freq + rest_time)
        print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, actual_aq_freq))
        t1 = time.time()

        if not gui==None:
            while (gui.qsize() > 0):
                try:
                    gui.get_nowait()
                    time.sleep(0.001)
                except:
                    continue
            #try:
            rtdict = dict()
            rtdict = {'dias': 0,
                    'vd_oil': 0,
                    'vd_gas': 0,
                    'oil_d50': 0,
                    'gas_d50': 0,
                    'saturation': 0}
            gui.put_nowait((timestamp, imraw, imraw, rtdict))

# the standard processing method under active development
def silcam_process(config_filename, datapath, multiProcess=True, realtime=False, discWrite=False, nbImages=None, gui=None,
                   overwriteSTATS = True):

    '''Run processing of SilCam images

    Args:
      config_filename   (str)               :  The filename (including path) of the config.ini file
      datapath          (str)               :  Path to the data directory
      multiProcess=True (bool)              :  If True, multiprocessing is used
      realtime=False    (bool)              :  If True, a faster but less accurate methods is used for segmentation and rts stats become active
      discWrite=False   (bool)              :  True will enable writing of raw data to disc
                                               False will disable writing of raw data to disc
      nbImages=None     (int)               :  Number of images to skip
      gui=None          (Class object)      :  Queue used to pass information between process thread and GUI
                                               initialised in ProcThread within guicals.py
    '''
    print(config_filename)

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

    if realtime:
        if discWrite:
            # copy config file into data path
            configFile2Copy = datetime.datetime.now().strftime('D%Y%m%dT%H%M%S.%f') + os.path.basename(config_filename)
            copyfile(config_filename, os.path.join(datapath, configFile2Copy))

        # update path_length
        updatePathLength(settings, logger)

    #Initialize the image acquisition generator
    aq = Acquire(USE_PYMBA=realtime)
    aqgen = aq.get_generator(datapath, writeToDisk=discWrite,
            camera_config_file=config_filename)

    #Get number of images to use for background correction from config
    print('* Initializing background image handler')
    bggen = backgrounder(settings.Background.num_images, aqgen,
            bad_lighting_limit = settings.Process.bad_lighting_limit,
            real_time_stats=settings.Process.real_time_stats)

    # make datafilename autogenerated for easier batch processing
    if (not os.path.isdir(settings.General.datafile)):
       logger.info('Folder ' + settings.General.datafile + ' was not found and is created')
       os.mkdir(settings.General.datafile)

    procfoldername = os.path.split(datapath)[-1]
    datafilename = os.path.join(settings.General.datafile,procfoldername)
    logger.info('output stats to: ' + datafilename)

    if os.path.isfile(datafilename + '-STATS.csv') and overwriteSTATS:
        logger.info('removing: ' + datafilename + '-STATS.csv')
        print('Overwriting ' + datafilename + '-STATS.csv')
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

    # initialise realtime stats class regardless of whether it is used later
    rts = scog.rt_stats(settings)

    if (multiProcess):
        proc_list = []
        mem = psutil.virtual_memory()
        memAvailableMb = mem.available >> 20
        distributor_q_size = np.min([int(memAvailableMb / 2 * 1/15), np.copy(multiprocessing.cpu_count() * 4)])

        logger.debug('setting up processing queues')
        inputQueue, outputQueue = defineQueues(realtime, distributor_q_size)

        logger.debug('setting up processing distributor')
        distributor(inputQueue, outputQueue, config_filename, proc_list, gui)

        # iterate on the bggen generator to obtain images
        logger.debug('Starting acquisition loop')
        t2 = time.time()
        for i, (timestamp, imc, imraw) in enumerate(bggen):
            t1 = np.copy(t2)
            t2 = time.time()
            print(t2-t1, 'Acquisition loop time')
            logger.debug('Corrected image ' + str(timestamp) +
                        ' acquired from backgrounder')

            # handle errors if the loop function fails for any reason
            if (nbImages != None):
                if (nbImages <= i):
                    break

            logger.debug('Adding image to processing queue: ' + str(timestamp))
            addToQueue(realtime, inputQueue, i, timestamp, imc) # the tuple (i, timestamp, imc) is added to the inputQueue
            logger.debug('Processing queue updated')

            # write the images that are available for the moment into the csv file
            logger.debug('Running collector')
            collector(inputQueue, outputQueue, datafilename, proc_list, False,
                      settings, rts=rts)
            logger.debug('Data collected')

            if not gui==None:
                logger.debug('Putting data on GUI Queue')
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
                        'gas_d50': rts.gas_d50,
                        'saturation': rts.saturation}
                gui.put_nowait((timestamp, imc, imraw, rtdict))
                logger.debug('GUI queue updated')

        logger.debug('Acquisition loop completed')
        if (not realtime):
            logger.debug('Halting processes')
            for p in proc_list:
                inputQueue.put(None)

        # some images might still be waiting to be written to the csv file
        logger.debug('Running collector on left over data')
        collector(inputQueue, outputQueue, datafilename, proc_list, True,
                  settings, rts=rts)
        logger.debug('All data collected')

        for p in proc_list:
            p.join()
            logger.info('%s.exitcode = %s' % (p.name, p.exitcode) )

    else:
        # load the model for particle classification and keep it for later
        nnmodel = []
        nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)

        # iterate on the bggen generator to obtain images
        for i, (timestamp, imc, imraw) in enumerate(bggen):
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

    print('PROCESSING COMPLETE.')

    #---- END ----

def addToQueue(realtime, inputQueue, i, timestamp, imc):
    '''
    Put a new image into the Queue.

    Args:
        realtime     (bool)     : boolean indicating wether the processing is done in realtime
        inputQueue   ()         : queue where the images are added for processing
                                  initilised using defineQueues()
        i            (int)      : index of the image acquired
        timestamp    (timestamp): timestamp of the acquired image
        imc          (uint8)    : corrected image
    '''
    if (realtime):
        try:
            inputQueue.put_nowait((i, timestamp, imc))
        except:
            pass
    else:
        while True:
            try:
                inputQueue.put((i, timestamp, imc), True, 0.5)
                break
            except:
                pass

def defineQueues(realtime, size):
    '''
    Define the input and output queues depending on wether we are in realtime mode

    Args:
        realtime: boolean indicating whether the processing is done in realtime
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
    
    Args:
        nnmodel (tensorflow model object)   :  loaded using sccl.load_model()
        class_labels (str)                  :  loaded using sccl.load_model()
        image  (tuple)                      :  tuple contianing (i, timestamp, imc)
                                               where i is an int referring to the image number
                                               timestamp is the image timestamp obtained from passing the filename
                                               imc is the background-corrected image obtained using the backgrounder generator
        settings (PySilcamSettings)         :  Settings read from a .ini file
        logger (logger object)              :  logger object created using
                                               configure_logger()
        gui=None (Class object)             :  Queue used to pass information between process thread and GUI
                                               initialised in ProcThread within guicals.py
                                               
    Returns:
        stats_all (DataFrame)               :  stats dataframe containing particle statistics
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
            print('ZERO particles identified')
            z = np.zeros(len(stats_all.columns)) * np.nan
            stats_all.loc[0] = z
            # 'export name' should not be nan because then this column of the
            # DataFrame will contain multiple types, so label with string instead
            if settings.ExportParticles.export_images:
                stats_all['export name'] = 'not_exported'

        # add timestamp to each row of particle statistics
        stats_all['timestamp'] = timestamp

        # add saturation to each row of particle statistics
        stats_all['saturation'] = saturation

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
    
    Args:
        config_filename (str)   : path of the config ini file
        inputQueue  ()          : queue where the images are added for processing
                                  initilised using defineQueues()
        outputQueue ()          : queue where information is retrieved from processing
                                  initilised using defineQueues()
        gui=None (Class object) : Queue used to pass information between process thread and GUI
                                  initialised in ProcThread within guicals.py
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

        if (not stats_all is None):
            outputQueue.put(stats_all)
        else:
            logger.debug('No stats found. skipping image.')


def distributor(inputQueue, outputQueue, config_filename, proc_list, gui=None):
    '''
    distributes the images in the input queue to the different loop processes
    Args:
        inputQueue  ()              : queue where the images are added for processing
                                      initilised using defineQueues()
        outputQueue ()              : queue where information is retrieved from processing
                                      initilised using defineQueues()
        proc_list   (list)          : list of multiprocessing objects
        gui=None (Class object)     : Queue used to pass information between process thread and GUI
                                      initialised in ProcThread within guicals.py
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

    Args:
        inputQueue  ()              : queue where the images are added for processing
                                      initilised using defineQueues()
        outputQueue ()              : queue where information is retrieved from processing
                                      initilised using defineQueues()
        datafilename (str)          : filename where processed data are written to csv
        proc_list   (list)          : list of multiprocessing objects
        testInputQueue (Bool)       : if True function will keep collecting until inputQueue is empty
        settings (PySilcamSettings) : Settings read from a .ini file
        rts (Class):                : Class for realtime stats
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
    '''
    Updater for realtime statistics

    Args:
        settings (PySilcamSettings) : Settings read from a .ini file
                                      settings.logfile is optional
                                       settings.loglevel mest exist
        rts (Class)                 :  Class for realtime stats
                                       initialised using scog.rt_stats()
        stats_all (DataFrame)       :  stats dataframe returned from processImage()
    '''
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
    Writes particle stats into the csv ouput file

    Args:
        datafilename (str):     filame prefix for -STATS.csv file that may or may not include a path
        stats_all (DataFrame):  stats dataframe returned from processImage()
    '''

    # create or append particle statistics to output file
    # if the output file does not already exist, create it
    # otherwise data will be appended
    # @todo accidentally appending to an existing file could be dangerous
    # because data will be duplicated (and concentrations would therefore
    # double) GUI promts user regarding this - directly-run functions are more dangerous.
    if not os.path.isfile(datafilename + '-STATS.csv'):
        stats_all.to_csv(datafilename +
                '-STATS.csv', index_label='particle index')
    else:
        stats_all.to_csv(datafilename + '-STATS.csv',
                mode='a', header=False)


def check_path(filename):
   '''Check if a path exists, and create it if not

   Args:
       filename (str): filame that may or may not include a path
   '''

   file = os.path.normpath(filename)
   path = os.path.dirname(file)
   if path:
      if not os.path.isdir(path):
         try:
            os.makedirs(path)
         except:
            print('Could not create catalog:',path)

def configure_logger(settings):
    '''Configure a logger according to the settings.

    Args:
        settings (PySilcamSettings): Settings read from a .ini file
                                     settings.logfile is optional
                                     settings.loglevel mest exist
    '''
    if settings.logfile:
        check_path(settings.logfile)
        logging.basicConfig(filename=settings.logfile,
                            level=getattr(logging, settings.loglevel))
    else:
        logging.basicConfig(level=getattr(logging, settings.loglevel))

def updatePathLength(settings, logger):
    '''Adjusts the path length of systems with the actuator installed and RS232
    connected.

    Args:
        settings (PySilcamSettings): Settings read from a .ini file
                                     settings.logfile is optional
                                     settings.loglevel mest exist
        logger (logger object)     : logger object created using
                                     configure_logger()
    '''
    try:
        logger.info('Updating path length')
        pl = scog.PathLength(settings.PostProcess.com_port)
        pl.gap_to_mm(settings.PostProcess.path_length)
        pl.finish()
    except:
        logger.warning('Could not open port. Path length will not be adjusted.')
