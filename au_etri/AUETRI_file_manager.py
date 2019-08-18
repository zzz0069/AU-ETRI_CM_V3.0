import os
import enum
import errno

from AUETRI_constants import Model, DatasetName, DatasetType, PROJECT_ROOT_DIR

DATASET_PATH = os.path.join(PROJECT_ROOT_DIR, 'datasets')
LIBRARY_PATH = os.path.join(PROJECT_ROOT_DIR, 'libraries')

class AUETRIFileManager(object):

    def get_dataset_file_path(model, dataset_name=None, dataset_type=None, filename=""):

        # Type checking
        if not isinstance(model, Model):
            raise TypeError('function argument must be an instance of class Enum.')

        if dataset_type is not None and not isinstance(dataset_type, DatasetType):
            raise TypeError('function argument must be an instance of class Enum.')

        if dataset_name is not None and not isinstance(dataset_name, DatasetName):
            raise TypeError('function argument must be an instance of class Enum.')

        modelName = model.value + '_model'

        if dataset_name is None and dataset_type is None:
            dataset = ""
        elif dataset_name is None:
            dataset = dataset_type.value + '_dataset'
        else:
            dataset = dataset_name.value + '_' + dataset_type.value + '_dataset'

        # Build path
        buildPath = os.path.join(DATASET_PATH, modelName, dataset, filename)

        # Check directory exists
        if not os.path.exists(buildPath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), buildPath)

        return buildPath

    def write_to_path(model, file=""):

        # Type checking
        if not isinstance(model, Model):
            raise TypeError('function argument must be an instance of class Enum.')

        modelName = model.value + '_model'
        
        # Build path
        buildPath = os.path.join(PROJECT_ROOT_DIR, 'au_etri', modelName)

        # Check directory exists
        if not os.path.exists(buildPath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), buildPath)

        # Create directory to write file
        resultDirectory =  modelName + '_results'
        buildResultDirPath = os.path.join(buildPath, resultDirectory)

        if not os.path.exists(buildResultDirPath):
            os.makedirs(buildResultDirPath)

        buildFilePath = os.path.join(buildResultDirPath, file)

        return buildFilePath

    def get_baseline_file_path(model, baseline):

        # Type checking
        if not isinstance(model, Model):
            raise TypeError('function argument must be an instance of class Enum.')

        modelName = model.value + '_model'
        baselineType = baseline + '_baseline'

        # Build path
        filename = baselineType + '.py'
        buildFilePath = os.path.join(modelName, filename)

        return buildFilePath

    def get_tree_view_of_directory(directory_path):
        dict = {}
        files = folders = 0
        for rootname, dirnames, filenames in os.walk(directory_path):
            files += len(filenames)
            folders += len(dirnames)
            for d in dirnames:
                p = os.path.join(rootname, d)
                dict[d] = len([name for name in os.listdir(p)])
        dict['total_files'] = files
        dict['total_folders'] = folders
        print("Total file : {:,} files \nTotal folders : {:,} folders".format(files, folders))

        return dict