import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pickle
import argparse
import string
from my_utils import Tokenizer


import os
import html
import pickle
import numpy as np
import xml.etree.cElementTree as ElementTree
import matplotlib.pyplot as plt
import functools 

from tqdm import tqdm
import math


'''
Creates the online and offline dataset for training

Before running this script, download the following things from 
https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database

data/lineStrokes-all.tar.gz   -   the stroke xml for the online dataset
data/lineImages-all.tar.gz    -   the images for the offline dataset
ascii-all.tar.gz              -   the text labels for the dataset

extract these contents and put them in the ./data directory (unless otherwise specified)
they should have the same names, e.g. "lineStrokes-all" (unless otherwise specified)
'''

def check_digit(str):
	num = ''
	for x in str:
		if x.isdigit():
			num += x
	return num 

def compute_strokes(l):
	new_l = []
	for ars in l:
		for ar in ars:
			new_l.append(ar)
	# print(new_l) 
	return len(new_l) ,new_l #mew_l is platten strokes
	
def decrease_density_stroke(l):
	new_l = []
	for ars in l:
		new_ars = []
		for i,ar in enumerate(ars):
			if i % 3 == 0 and i != 0:
				continue
			new_ars.append(ar)

		new_l.append(np.array(new_ars))
	# print(new_l) 
	return new_l


def separate(pts):
    seps = []
    for i in range(0, len(pts) - 1):
        if distance(pts[i], pts[i+1]) > 600: #600
            seps += [i + 1]
    return [pts[b:e] for b, e in zip([0] + seps, seps + [len(pts)])]

def distance(p1, p2):
  return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1]-p2[1])**2)

def read_xml(file, root = 'data/vn_handwriting_strokes' ):
  file_name, extension = os.path.splitext(file)
  # print('file_name:', file)
  if extension == '.inkml':
      # print('[{:5d}] File  -- '.format(os.path.join(root, file)), end='')
      xml = ElementTree.parse(os.path.join(root, file)).getroot() #ink
      transcription = xml.findall('traceGroup') #traceGroupe
      # if not transcription:
      #     print('skipped')
      #     return [], []
      
      all_texts = [html.unescape(tracegr.find('annotationXML').find('Tg_Truth').text) for tracegr in transcription ]
      # print(texts, len(texts))
      points = []
      for trg in transcription:
        traces = trg.findall('trace')
        points.append(traces)
        
      # print('points:', len(points), points)
      lines = []
      file_names, texts = [], []
      for idx, tracegr in enumerate(points):
        if len(all_texts[idx]) < 10:   ## remove sample with text has less than 10 character
          continue
        line = []
        previous = None
        for trace in tracegr:
          # print('trace:', trace.text)
          xys = trace.text.split(', ')
          # print('xys:', xys)
          for i, xy in enumerate(xys):
            x,y = xy.split(' ', 1)
            end = 1.0 if  i == len(xys) -1  else 0.0
            x, y = int(x), int(check_digit(y))

            # if previous is not None and [x, -y] == previous:
              # continue
            # point = np.array([int(x), int(check_digit(y)), end])

            point = [x, y, end]
            if len(line) == 0  or end == 1 or line[-1][2] == 1 or distance(point, line[-1]) > 50:
              line.append(point)
            
        strokes, previous = [], [line[0][0], -line[0][1]]
        for i, li in enumerate(line[1: ]):
          previous = [line[i-1][0], -line[i-1][1]]
          strokes.append([li[0]-previous[0], -li[1]-previous[1], li[2]])
        
        lines.append(strokes) 
        file_names.append(file_name+'_'+str(idx)+'.png')
        texts.append(all_texts[idx])
  return lines, file_names, texts


def remove_whitespace(img, thresh, remove_middle=False):
    #removes any column or row without a pixel less than specified threshold
    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)
    
    rows = np.where(row_mins < thresh)
    cols = np.where(col_mins < thresh)

    if remove_middle: return img[rows[0]][:, cols[0]]
    else: 
        rows, cols = rows[0], cols[0]
        return img[rows[0]:rows[-1], cols[0]:cols[-1]]
        
def norms(x): 
    return np.linalg.norm(x, axis=-1)

def combine_strokes(x, n):
    #consecutive stroke vectors who point in similar directions are summed
    #if the pen was picked up in either of the strokes,
    #we pick up the pen for the combined stroke
    s, s_neighbors = x[::2, :2], x[1::2, :2]
    if len(x)%2 != 0: s = s[:-1]
    values = norms(s) + norms(s_neighbors) - norms(s + s_neighbors)
    ind = np.argsort(values)[:n]
    x[ind*2] += x[ind*2+1]
    x[ind*2, 2] = np.greater(x[ind*2, 2], 0)
    x = np.delete(x, ind*2+1, axis=0)
    x[:, :2] /= np.std(x[:, :2])
    return x
 
## return dict: {name_file_img: text}
def parse_page_text(dir_path, id):
    dict = {}
    f = open(dir_path + '/' + id)
    has_started = False
    line_num = -1
    for l in f.readlines():
        if 'CSR' in l: has_started = True
        #the text under 'CSR' is correct, the one labeled under 'OCR' is not
        if has_started:
            if line_num>0: # theres one space after 'CSR'
                dict[id[:-4]+ '-%02d' % line_num]  = l[:-1]
                # add the id of the line -0n as a key to dictionary, 
                # with value of the line number (excluding the last \n)
            line_num += 1
    return dict

def create_dict(path):
    #creates a dictionary of all the line IDs and their respective texts
    dict = {}
    for dir in os.listdir(path):
        dirpath = path + '/' + dir
        for subdir in os.listdir(dirpath):
            subdirpath = dirpath + '/' + subdir
            forms = os.listdir(subdirpath)
            [dict.update(parse_page_text(subdirpath, f)) for f in forms]
    return dict
 
def parse_stroke_xml(strokes):
    strokes = np.array(strokes)
    strokes[:, 2] = np.roll(strokes[:, 2], 1) 
    #currently, a stroke has a 1 if the next stroke is not drawn
    #the pen pickups are shifted by one, so a stroke that is not drawn has a 1
    strokes[:, :2] /= np.std(strokes[:, :2])
    for i in range(3): strokes = combine_strokes(strokes, int(len(strokes)*0.2))
    return strokes

def read_img(path, height):
    img = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale')
    img_arr = tf.keras.preprocessing.image.img_to_array(img).astype('uint8')
    img_arr = remove_whitespace(img_arr, thresh=127)
    h, w, _ = img_arr.shape
    img_arr = tf.image.resize(img_arr, (height, height * w // h))
    # img_arr = tf.image.resize(img_arr, (height, height))
    return img_arr.numpy().astype('uint8')

def create_dataset(strokes_path, images_path, tokenizer, height):
    dataset = []
    # offline_dataset = []
    # same_writer_examples = []
    # forms = open(formlist).readlines()

    # for f in forms:
    #     path = strokes_path + '/' + f[1:4] + '/' + f[1:8]
    #     offline_path = images_path + '/' + f[1:4] + '/' + f[1:8]

    #     samples = [s for s in os.listdir(path) if f[1:-1] in s]
    #     offline_samples = [s for s in os.listdir(offline_path) if f[1:-1] in s]
    #     shuffled_offline_samples = offline_samples.copy()
    #     random.shuffle(shuffled_offline_samples)
        
    #     for i in range(len(samples)):
    #       try:
    #         dataset.append((
    #             parse_stroke_xml(path + '/' + samples[i]),
    #             tokenizer.encode(text_dict[samples[i][:-4]]),
    #             read_img(offline_path + '/' + shuffled_offline_samples[i], height)
    #         ))
    #       except:
    #         print('error', samples[i])        
    for i in tqdm(os.listdir(strokes_path)):
      lines, file_names, texts = read_xml(i, strokes_path)
      for ii in range(len(lines)):
        dataset.append((
          parse_stroke_xml(lines[ii]),
          tokenizer.encode(texts[ii]),
          read_img(images_path+'/'+file_names[ii], height)
        ))

    return dataset

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-t', '--text_path', help='path to text labels, \
                        default ./data/ascii-all', default='./data/ascii-all' )

    parser.add_argument('-s', '--strokes_path', help='path to stroke xml, \
                        default ./data/lineStrokes-all', default='./data/lineStrokes-all')

    parser.add_argument('-i', '--images_path', help='path to line images, \
                        default ./data/lineImages-all', default='./data/lineImages-all')
                        
    parser.add_argument('-H', '--height', help='the height of offline images, \
                        default 96', type=int, default= 96)
    
    parser.add_argument('-p', '--path', help='path to save file dataset, \
                        default 96', type=str, default= 'data/vn_train_strokes.p')
    
    args = parser.parse_args()
    t_path = args.text_path
    s_path = args.strokes_path
    i_path = args.images_path
    H = args.height

    train_info = './data/trainset.txt'
    val1_info = './data/testset_f.txt'  #labeled as test, we validation set 1 as test instead
    val2_info = './data/testset_t.txt'  
    test_info = './data/testset_v.txt'  #labeled as validation, but we use as test

    tok = Tokenizer()
    # labels = create_dict(t_path)  ## dict {img_name: text} all of img
    # train_strokes = create_dataset(train_info, s_path, i_path, tok, labels, H)
    # val1_strokes = create_dataset(train_info, s_path, i_path, tok, labels, H)
    # val2_strokes = create_dataset(train_info, s_path, i_path, tok, labels, H)
    # test_strokes = create_dataset(train_info, s_path, i_path, tok, labels, H)
    
    # train_strokes += val1_strokes
    # train_strokes += val2_strokes
    # random.shuffle(train_strokes)
    # random.shuffle(test_strokes)

    train_strokes = create_dataset(s_path, i_path, tok, H)

    with open(args.path, 'wb') as f:
        pickle.dump(train_strokes, f)
    # with open('./data/test_strokes.p', 'wb') as f:
        # pickle.dump(test_strokes, f)

if __name__ == '__main__':
    main()
