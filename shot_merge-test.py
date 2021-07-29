import os
import sys
import cv2
import pandas as pd
import numpy as np
import time
import argparse
import json
import glob
import signal
import copy
import csv
import poster_processor as pp

class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass

def service_shutdown(signum, frame):
    print('Caught signal %d' % signum)
    raise ServiceExit


def read_shotinfo(content, csvfiledirectory):
    if os.path.isfile(csvfiledirectory):
        data = pd.read_csv(csvfiledirectory)
        selectData = pd.DataFrame(data, columns=[
                                  'FILEPATH', 'TITLE', 'FILENAME', 'SHOTID', 'STARTFRAMEINDEX', 'ENDFRAMEINDEX'])
        selectData = selectData[(selectData['FILENAME'] == content)]
        #print(selectData)
        return selectData
    else:
        print("shot csv 파일이 없습니다.")

def main(args):

    # Register the signal handlers
    signal.signal(signal.SIGTERM, service_shutdown)
    signal.signal(signal.SIGINT, service_shutdown)

    print('Starting main program')

    #--------------------------------------------------------------------------
    # Start the job threads
    try:
        # init
        start = time.time()


        #------------------------------------------------------------------
        # 0. Configure processing information
        #------------------------------------------------------------------
        content = args.content
        shot_csv_path = args.shotcsvpath
        frame_dir = args.framedir + content
        shot_merge_threshold = float(args.threshold)


        #------------------------------------------------------------------
        # 1. Categorize Frames by ShotID
        #------------------------------------------------------------------


        selectShotData = read_shotinfo(content, shot_csv_path)
        startframeindex_list = selectShotData.STARTFRAMEINDEX.tolist()

        jpglist = glob.glob(frame_dir + '**/*.jpg', recursive=True)
        pnglist = glob.glob(frame_dir + '**/*.png', recursive=True)
        filelist = jpglist + pnglist

        shot_dict = {}

        if(len(startframeindex_list) == 0):
            print("csv 파일 내에 shot 정보가 없습니다")
        else:
            for framefile in filelist:
                shotid = pp.find_nearestshotid(startframeindex_list, int(framefile[framefile.rfind('/')+1:-4]))
                str_shotid = pp.zeropad_shotid(shotid)
                if(shot_dict.get(str_shotid, 0)==0):
                    shot_dict[str_shotid] = [framefile[framefile.rfind('/')+1:-4]]
                else:
                    shot_dict[str_shotid].append(framefile[framefile.rfind('/')+1:-4])
                    shot_dict[str_shotid].sort()
            print("### Sorted framelist by shot : ")
            print(sorted(shot_dict.items()))
        
        shot_dict = pp.cut_max_shot(shot_dict, args.maxshotnumber)

        #------------------------------------------------
        # 2. Compare Similarities 
        #------------------------------------------------

        rep_imagelist_dict_with_path = {}
        rep_imagelist_dict_by_frame = {}
        
        for shotid in shot_dict:
            rep_imagelist_dict_with_path[shotid] = frame_dir + '/'+shot_dict[shotid][0]+'.jpg'
            rep_imagelist_dict_by_frame[shotid] = shot_dict[shotid][0]
        print("### Representative framelsit of each shot : ")
        print(sorted(rep_imagelist_dict_with_path.items()))

        rep_imagelist = list(rep_imagelist_dict_with_path.values())


        #------------------------------------------------------------------
        # 3. Merge images based on similarities list 
        #------------------------------------------------------------------
        results_dict = []
        results_dict = pp.factorial_similarity(rep_imagelist, results_dict, shot_merge_threshold, args.modelpath)
        sorted_shotlist = pp.group_shot(
            results_dict, rep_imagelist_dict_by_frame)

        print("#### Final sorted_shotlist: ")
        print(sorted_shotlist)

        end = time.time()
        print('processing time : %03f s' % (end - start))

    except ServiceExit:
        print('stop process')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--content', required=True,
                        help="content name")
    parser.add_argument('-t', '--threshold', required=True,
                        help="threshold for merging shots")
    parser.add_argument('-i', '--framedir', required=True,
                        help="input thumbnail directory")
    parser.add_argument('-v', '--shotcsvpath', required=True,
                        help="path for shotinfo csv file")
    parser.add_argument('-s', '--maxshotnumber', default=30, type=int,
                        help="define max shot number before merging")
    parser.add_argument('-f', '--filemode', default='both', type=str,
                        help='choose jpg, png, or both')
    parser.add_argument('-m', '--modelpath', default='./model/140_224', help='model directory')
    args = parser.parse_args()
    main(args)


