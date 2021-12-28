from pydub import AudioSegment
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import editdistance as ed
import pdb
from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.io.wavfile as wav
from python_speech_features import logfbank
import argparse
import re
from tqdm import notebook
import json

##GLOBAL VARIABLES
IGNORE_ID = -1

def make_txt_file(root,data_type,label_source):
    folder_path = root+"\\" +data_type+ "\\" + label_source

    folder_list = os.listdir(folder_path)
    # folder_list = folder_list[:2]
    pre_fix = "json"
    
    for folders in notebook.tqdm(folder_list):
        rower_folder_path = folder_path + "\\" + folders
        end_folders =os.listdir(rower_folder_path)

        # writing 해줄 파일 열 어야함
        write_path = rower_folder_path.replace('label','source')
        txt_name = write_path.split('\\')[-1]
        #print(write_path)
        txt_dir = "\\".join(write_path.split('\\')[:-1])
        txt = txt_dir + "\\" +txt_name+"\\" +txt_name +".trans.txt"
        #print(txt)
        with open(txt, "w", encoding='utf8') as tf:
            for file in end_folders:
                if pre_fix in file:
                    path = rower_folder_path + "\\" + file
                    write_path = path
                    
                    with open(path, 'r',encoding='utf8') as f:
                        json_data = json.load(f)
                    label = json_data['발화정보']['stt']
                    fileNm = json_data['발화정보']['fileNm'].split('.')[0]
                    writed = fileNm +" "+ label+"\n"
                    tf.write(writed)
    return 

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad


def load_vocab(vocab_file):
    unit2idx = {}
    with open(os.path.join(vocab_file), "r", encoding="utf-8") as v:
        for line in v:
            idx, char = line.strip().split(",")
            unit2idx[str(idx)] = char
    return unit2idx


# CreateOnehotVariable function
# *** DEV NOTE : This is a workaround to achieve one, I'm not sure how this function affects the training speed ***
# This is a function to generate an one-hot encoded tensor with given batch size and index
# Input : input_x which is a Tensor or Variable with shape [batch size, timesteps]
#         encoding_dim, the number of classes of input
# Output: onehot_x, a Variable containing onehot vector with shape [batch size, timesteps, encoding_dim]
def CreateOnehotVariable(input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1, input_x, 1)).type(input_type)

    return onehot_x


# TimeDistributed function
# This is a pytorch version of TimeDistributed layer in Keras I wrote
# The goal is to apply same module on each timestep of every instance
# Input : module to be applied timestep-wise (e.g. nn.Linear)
#         3D input (sequencial) with shape [batch size, timestep, feature]
# output: Processed output      with shape [batch size, timestep, output feature dim of input module]
def TimeDistributed(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size, time_steps, -1)


def traverse(root, path, search_fix=".flac", return_label=False):
    f_list = []
    
    # this is for linux // now, fixed for windows
    for p in path:
        p = root + p
        #print(p)
        for sub_p in sorted(os.listdir(p)):
            for sub2_p in sorted(os.listdir(p + sub_p + "\\")):
                if return_label:
                    # Read trans txt
                    with open(p + sub_p + "\\" + sub2_p + "\\" + sub_p + "-" + sub2_p + ".trans.txt", "r") as txt_file:
                        for line in txt_file:
                            f_list.append(" ".join(line[:-1].split(" ")[1:]))
                else:
                    # Read acoustic feature
                    for file in sorted(os.listdir(p + sub_p + "\\" + sub2_p+"\\")):
                        if search_fix in file:
                            #print("p :",p)
                            #print("sub_p :", sub_p)
                            #print("sub2_p :", sub2_p)
                            #print("file :", file)
                            file_path = p + sub_p + "\\" + sub2_p + "\\" + file
                            #file_path = p+sub_p+sub2_p+file
                            f_list.append(file_path)
                    


    #print("f_list :",new_flist[-3:])
    return f_list

def file_list(root, data_type, label_source,pre_fix):
    #root = 'C:\\Users\\USER\\Desktop\\jiwon\\las-pytorch\\data\\kospeech'
    #data_type = "Test"
    #label_source = "label"

    f_list = []

    folder_path = root+"\\" +data_type+ "\\" + label_source

    folder_list = os.listdir(folder_path)
    for folders in folder_list:
        rower_folder_path = folder_path + "\\" + folders
        end_folders =os.listdir(rower_folder_path)
        
        for file in end_folders:
            if pre_fix in file:
                f_list.append(file)
    
    return f_list


def file_list_path(root, data_type, label_source,pre_fix, return_label = False):
    #root = 'C:\\Users\\USER\\Desktop\\jiwon\\las-pytorch\\data\\kospeech'
    #data_type = "Test"
    #label_source = "label"
    if data_type =="train":
        encode = 'utf-8'
    else:
        encode = 'cp949'
    f_list = []

    folder_path = root+"\\" +data_type+ "\\" + label_source

    folder_list = os.listdir(folder_path)
    for folders in folder_list:
        rower_folder_path = folder_path + "\\" + folders
        end_folders =os.listdir(rower_folder_path)
        
        
        if return_label:
            for end in end_folders:
                if pre_fix in end:
                    txt_dir = rower_folder_path + "\\" + end 
                    with open(txt_dir, "r",encoding=encode) as txt_file:
                        for line in txt_file:
                            f_list.append(" ".join(line[:-1].split(" ")[1:]))
                    
            # Read trans txt
            #print(end_folders)
            #with open(p + sub_p + "\\" + sub2_p + "\\" + sub_p + "-" + sub2_p + ".trans.txt", "r") as txt_file:
            #    for line in txt_file:
            #        f_list.append(" ".join(line[:-1].split(" ")[1:]))
        else:
            for file in end_folders:
                #print(file)
                if pre_fix in file:

                    full_path = rower_folder_path + "\\" + file
                    f_list.append(str(full_path))
    
    return f_list


def flac2wav(f_path):
    #print("f_path :",f_path)
    flac_audio = AudioSegment.from_file(f_path, "flac")
    flac_audio.export(f_path[:-5] + ".wav", format="wav")


def mp32wav(f_path):
    mp3_audio = AudioSegment.from_mp3(f_path)
    mp3_audio.export(f_path[:-4] + ".wav", format="wav")


def wav2logfbank(f_path, win_size, n_filters, nfft=512):
    (rate, sig) = wav.read(f_path)
    fbank_feat = logfbank(sig, rate, winlen=win_size, nfilt=n_filters, nfft=nfft)
    #os.remove(f_path)
    np.save(f_path[:-3] + "fb" + str(n_filters), fbank_feat)


def norm(f_path, mean, std):
    np.save(f_path, (np.load(f_path) - mean) / std)


def char_mapping(tr_text, target_path,seq):
    char_map = {}
    char_map["<sos>"] = 0
    char_map["<eos>"] = 1
    char_idx = 2
    

    # map char to index
    for text in tr_text:
        for char in text:
            if char not in char_map:
                char_map[char] = char_idx
                char_idx += 1

    # Reverse mapping
    rev_char_map = {v: k for k, v in char_map.items()}

    # Save mapping
    if seq=="first":
        write_mode = "w"
        with open(target_path + "idx2chap.csv", write_mode,encoding='utf8') as f:
            f.write("idx,char\n")
            for i in range(len(rev_char_map)):
                f.write(str(i) + "," + rev_char_map[i] + "\n")
    else:
        write_mode = "a"
        with open(target_path + "idx2chap.csv", "r",encoding='utf8') as f:
            last = f.readlines()
            idx = last[-1].split(",")[0]
        
        with open(target_path + "idx2chap.csv", write_mode,encoding='utf8') as f:
            # f.write("idx,char\n")
            for i in range(idx,len(rev_char_map)):
                f.write(str(i) + "," + rev_char_map[i] + "\n")
        
    return char_map
