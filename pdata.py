#!/usr/bin/env python
"""
pdata.py

Data preprocessing, from raw examples to pickled gzipped Python objects.

The raw examples are .png images representing levels.

The data format for the neural network is just a list of lists of floats. The
way that this module processes inputs, it first scans all of the input files to
get a palette size, and then creates W x W x P vectors of floats, where W is
the window size and P is the palette size. Each of these vectors has a 1 for
each pixel of the color 1..P and a 0 elsewhere in each W x W slice,
concatenated together to get W x W x P floats.
"""

import os
import pickle
import gzip
import numpy
import random

import theano

from PIL import Image

def process_data(
  directory="data",
  result_file="examples.pkl.gz",
  window_size=8,
  step=4,
  drop_empty=None,
  palette={},
  r_palette={}
):
  all_images = []
  dataset = []
  border_index = -1

  # Collect all *.lvl.png images:
  for dpath, dnames, fnames in os.walk(directory):
    for f in fnames:
      if f.endswith(".lvl.png"):
        all_images.append(Image.open(os.path.join(dpath, f)).convert("RGB"))

  # For each image, iterate over pixels to build our combined palette:
  index = 0
  for img in all_images:
    colors = img.getcolors(img.size[0]*img.size[1])
    for (count, color) in colors:
      if color not in palette:
        palette[color] = index
        r_palette[index] = color
        index += 1
  # Add a "border" out-of-band color (-1):
  palette[-1] = index
  r_palette[index] = (0xff, 0xff, 0x0e) # a quirky orange
  border_index = index

  # Iterate through all possible subregions of each image, turning each region
  # into a training example:
  possible = 0
  for img in all_images:
    for x in range(-window_size + 1, img.size[0], step):
      for y in range(-window_size + 1, img.size[1], step):
        possible += 1
        example = numpy.zeros(
          shape=(window_size, window_size),
          dtype=theano.config.floatX
        )
        pixels = img.crop((x, y, x+window_size, y+window_size)).load()
        non_empty = False
        for ix in range(window_size):
          lx = x + ix
          for iy in range(window_size):
            ly = y + iy
            px = pixels[ix, iy]
            if lx < 0 or ly < 0 or lx > img.size[0]-1 or ly > img.size[1]-1:
              px = -1
            if drop_empty and px not in drop_empty:
              non_empty = True
            example[ix, iy] = palette[px]
        if not drop_empty or non_empty:
          dataset.append(example)

  print("... generated {}/{} examples ...".format(len(dataset), possible))

  # DEBUG:
  #dataset = dataset[:20]
  dataset = numpy.array(dataset)

  # Pickle and gzip the dataset:
  with gzip.open(os.path.join(directory, result_file), 'wb') as fout:
    pickle.dump(
      {
        "examples": dataset,
        "window_size": window_size,
        "palette": palette,
        "r_palette": r_palette,
        "border": border_index,
      },
      fout
    )

if __name__ == "__main__":
  #process_data(window_size=8, step=1, drop_empty=[-1, (0xff, 0xff, 0xff)])
  process_data(window_size=8, step=1, drop_empty=None)
