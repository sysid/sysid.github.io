#! /usr/bin/python
# -*- coding: utf-8 -*-

"""module docstring"""
# https://blog.dominodatalab.com/simple-parallelization/

# imports
import sys, os, argparse, logging  # NOQA
from pprint import pprint
from twBase import *  # NOQA
from utils import *  # NOQA
#from twCfg import *  # NOQA
from ShowMeTheFish import *  # NOQA

from joblib import Parallel, delayed
import multiprocessing

#Allow relative imports to directories above cwd/
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# constants
CHECKING_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "checking")
scriptPath = print(os.path.abspath(__file__))
# exception classes
# interface functions
# classes
# internal functions & classes


def foo(args):
    pprint(args)



def processInput(n, f):
    #progress(".") if n < len(original_image_name_list) else progress(".", True)
    progress(".")
    tPath = os.path.join(TRAIN_FOLDER_PATH, f)
    lPath = os.path.join(LOCALIZATION_FOLDER_PATH, f)
    figure = plots([image.load_img(tPath), image.load_img(lPath)])

    path = os.path.join(CHECKING_FOLDER_PATH, f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path)
    #plt.show()
    figure.clear()
    plt.close()


def checkImgs(args):
    """
        creates images where the original and the localization are displayed side by side.
        runs in parallel
    """
    shutil.rmtree(CHECKING_FOLDER_PATH, ignore_errors=True)

    num_cores = multiprocessing.cpu_count()

    original_image_path_list = sorted(glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*")))
    localization_image_path_list = sorted(glob.glob(os.path.join(LOCALIZATION_FOLDER_PATH, "*/*")))

    # Sanity check
    original_image_name_list = [image_path[len(TRAIN_FOLDER_PATH)+1:] for image_path in original_image_path_list]
    localization_image_name_list = [image_path[len(LOCALIZATION_FOLDER_PATH)+1:] for image_path in localization_image_path_list]
    assert np.array_equal(original_image_name_list, localization_image_name_list)

    log.info("Creating", directory=CHECKING_FOLDER_PATH)

    # fn must be pickled, so must be defined at top-level
    results = Parallel(n_jobs=num_cores)(delayed(processInput)(n, f) for n, f in enumerate(original_image_name_list))
    log.info("Created.", n=n)


def process_command_line(argv):
    '''
    sets the global varialbes m.___
    '''
    # create the top-level parser
    parser = argparse.ArgumentParser(description="programpurpose")
    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands', help='additional help', dest='subcommand')
    subparsers.required = True  # makes sure, that a subcommand is given

    # create the parser for the "checkImgs" command
    parser_checkImgs = subparsers.add_parser('checkImgs', aliases=['ci'])
    #parser_checkImgs.add_argument('img', type=str)
    parser_checkImgs.set_defaults(func=checkImgs)  # defines the function to call

    args = parser.parse_args(argv)

    return args


def main(argv=None):
    args = process_command_line(argv)
    logging.basicConfig(format="", stream=sys.stderr, level=logging.DEBUG)
    logcfg(sys.stderr, logging.DEBUG, RenderEnum.console)

    twStart()

    ### run the subcommand
    args.func(args)

    twEnd()
    return 0  # success

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
