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

  # Iterate through subregions of the image using two grids offset by 1/2
  # window_size, turning each region into a training example:
  for img in all_images:
    for x in range(0, img.size[0] - window_size + 1, step):
      for y in range(0, img.size[1] - window_size + 1, step):
        example = []
        data = img.crop((x, y, x+window_size, y+window_size)).getdata()
        non_empty = False
        for px in data:
          if px != drop_empty:
            non_empty = True
          example.append(palette[px])
        if drop_empty is None or non_empty:
          dataset.append(example)

  x = len(list(range(0, img.size[0] - window_size + 1, step)))
  y = len(list(range(0, img.size[1] - window_size + 1, step)))
  print("... generated {}/{} examples ...".format(len(dataset), x*y))

  dataset = numpy.array(dataset, dtype=float)

  # Pickle and gzip the dataset:
  with gzip.open(os.path.join(directory, result_file), 'wb') as fout:
    pickle.dump(
      {
        "examples":dataset,
        "window_size":window_size,
        "palette":palette,
        "r_palette":r_palette
      },
      fout,
      protocol=pickle.DEFAULT_PROTOCOL
    )

if __name__ == "__main__":
  process_data(window_size=8, step=1, drop_empty=(0xff, 0xff, 0xff))
  #process_data(window_size=8, drop_empty = None)
