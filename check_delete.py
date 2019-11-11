from data_preparation import reconstruct_path


import os
import sys
from ..credential import s3
from pathlib import Path
from tqdm import tqdm
from librosa.core import load
import argparse
import json
import librosa
import shutil
import torch
import numpy as np
import random


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', action='store_true') # download flag if you change params.json you need to set this flag
  args = parser.parse_args()
  return args

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def find_type(data_list,filepath):
  speaker_id = filepath.split('/')[0].split('-')[-1]
  new_path = filepath
  for data in data_list:
    if data.split('-')[-1]==speaker_id:
      new_path = data.split('-')[0]+'/'+new_path
      break
  return new_path

def span_path(full_filepath):
  parse_speaker_id = full_filepath.split('/')[-4].split('-')[-1]
  parse_file_id = full_filepath.split('/')[-3].split('-')[-1]
  parse_filename = full_filepath.split('/')[-1]
  new_path = "{}-{}-{}".format(parse_speaker_id,parse_file_id,parse_filename)
  return new_path

def cvt_path(s3path):
  splits = s3path.split('/')
  new_path = '/'.join(splits[:-2])+'/wav/'+'/'.join(splits[-1:])
  return new_path


def main():
  wr_file = '/app/missingno.txt'
  wr_dir = '/app/missingno'
  args = parse_args()
  
  with open('/app/params.json') as f:
    json_data = json.load(f)
  data_list = json_data['data']
  
  if os.path.isdir(wr_dir):
    shutil.rmtree(wr_dir)
  os.mkdir(wr_dir)
    
  if not args.d:
    if os.path.isfile(wr_file):
      os.remove(wr_file)
    writer = open(wr_file,'a')
    bucket = s3.Bucket('mindlogic-tts')
    tot_list = list()
    
    print("init list of file",flush=True)      
    for data in data_list:
      parse_id = int(data.split('-')[-1])
      parse_type = data.split('-')[0]
      if parse_type=='nltk':
        spkr_dataset = bucket.objects.filter(Prefix='nltk/speaker-{}/'.format(parse_id)).all()
      else:
        spkr_dataset = bucket.objects.filter(Prefix='celeb/speaker-{}/'.format(parse_id)).all()
      
      for el in spkr_dataset:
        if el.key.endswith('.npz-2'):
          tot_list.append("/".join(el.key.split('/')[1:]))
    
    print("write start",flush=True)      
    for rootpath,dirs,files in os.walk('/data'):
      for file in files:
        filepath = os.path.join(rootpath,file)
        search_key = reconstruct_path(file)
        if search_key in tot_list:
          tot_list.remove(search_key)
          continue
    
    for el in tot_list:
      writer.write(el+'\n')
      print("detected: ",el,flush=True)
    writer.close()
  
  print("download start",flush=True)      
  filelist = files_to_list(wr_file)
  for el in tqdm(filelist,desc='downloading missing #'):
    s3_path1 = find_type(data_list,el)
    #print(s3_path1,flush=True)
    new_path1 = os.path.join(wr_dir,span_path(s3_path1))
    s3.Object('mindlogic-tts',s3_path1).download_file(new_path1)
    s3_path2 = cvt_path(s3_path1).split('.npz-2')[0]+'.wav'
    #print(s3_path2,flush=True)
    new_path2 = os.path.join(wr_dir,span_path(s3_path2))
    s3.Object('mindlogic-tts',s3_path2).download_file(new_path2)
    #print(new_path2,flush=True)
  return


if __name__=='__main__':
  main()
    