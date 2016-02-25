#!/usr/bin/env python
"""
Generates images that resemble input images.

Stacked denoising autoencoder-style neural network code using theano.

Written with reference to tutorial code from:
  https://github.com/lisa-lab/DeepLearningTutorials.git
Tutorial page:
  http://deeplearning.net/tutorial/SdA.html
"""

import os
import sys
import timeit
import datetime
import functools
import copy

import gzip
import pickle

import numpy

import theano
import theano.tensor as T

from PIL import Image

def debug(*args, **kwargs):
  print(*args, **kwargs)

def load_data(filename="data/examples.pkl.gz"):
  '''
  Loads pickled gzipped data (see pdata.py).
  '''
  debug("... loading data ...")

  # Load the dataset
  with gzip.open(filename, 'rb') as fin:
    dataset = pickle.load(fin)

  # format: a dictionary with three keys:
  #  "examples": a numpy.ndarray with 2 dimensions where each row is an
  #              example
  #  "palette": a dictionary mapping colors to integers
  #  "r_palette": the reverse of the palette, mapping from integers to colors

  dataset["examples"] = numpy.array([
    explode_example(ex, len(dataset["palette"]))
    for ex in dataset["examples"]
  ])

  dataset["examples"] = theano.shared(
    numpy.asarray(dataset["examples"], dtype=theano.config.floatX),
    borrow = False
  )

  return dataset

class AutoEncoder:
  """
  A layer of neurons whose weights can be used for both interpretation and
  reconstruction. It also has functions for training to denoise on a given
  input.
  """

  def __init__(self, numpy_rng, input_size, output_size):
    # RNG for adding noise to input:
    self.theano_rng = theano.tensor.shared_randomstreams.RandomStreams(
      numpy_rng.randint(2 ** 30)
    )

    self.input_size = input_size
    self.output_size = output_size

    # Weights and offsets for deconstruction (input -> output):
    wsx = int(numpy_rng.uniform(low=0, high=input_size))
    wsy = int(numpy_rng.uniform(low=0, high=output_size))
    self.de_weights = theano.shared(
      value=numpy.asarray(
        numpy_rng.uniform(
          low=-4 * numpy.sqrt(6. / (input_size + output_size)),
          high=4 * numpy.sqrt(6. / (input_size + output_size)),
          size=(input_size*2, output_size*2)
        )[wsx:wsx+input_size,wsy:wsy+output_size],
        dtype=theano.config.floatX
      ),
      name='de_weights',
      borrow=True
    )

    self.de_offsets = theano.shared(
      value=numpy.zeros(
        output_size,
        dtype=theano.config.floatX
      ),
      name='de_offsets',
      borrow=True
    )

    # Weights and offsets for reconstruction (output -> input):
    wsx = int(numpy_rng.uniform(low=0, high=output_size))
    wsy = int(numpy_rng.uniform(low=0, high=input_size))
    self.re_weights = theano.shared(
      value=numpy.asarray(
        numpy_rng.uniform(
          low=-4 * numpy.sqrt(6. / (input_size + output_size)),
          high=4 * numpy.sqrt(6. / (input_size + output_size)),
          size=(output_size*2, input_size*2)
        )[wsx:wsx+output_size,wsy:wsy+input_size],
        dtype=theano.config.floatX
      ),
      name='re_weights',
      borrow=True
    )

    self.re_offsets = theano.shared(
      value=numpy.zeros(
        input_size,
        dtype=theano.config.floatX
      ),
      name='re_offsets',
      borrow=True
    )

    self.params = [
      self.de_weights,
      self.de_offsets,
      self.re_weights,
      self.re_offsets
    ]

  # Evaluation functions:
  def get_deconstruct(self, input):
    return T.nnet.sigmoid(
      T.dot(input, self.de_weights) + self.de_offsets
    )

  def get_reconstruct(self, output):
    return T.nnet.sigmoid(
      T.dot(output, self.re_weights) + self.re_offsets
    )

  # Training functions:
  def get_cost_and_updates(self, input, corruption, learning_rate):
      """
      Returns a theano expression for the cost function of the network on the
      given input, along with an update list for updating the weights based on
      the cost gradient.
      """

      corrupt = self.theano_rng.binomial(
        size=input.shape,
        n=1, # number of trials
        p=1 - corruption, # probability of success per trial
        dtype=theano.config.floatX
      )
      static = self.theano_rng.random_integers(
        size=input.shape,
        low=0,
        high=1
      )
      corrupted = (
        corrupt * input
      + (1 - corrupt) * static
      )

      rep = self.get_deconstruct(corrupted) # internal representation
      rec = self.get_reconstruct(rep) # reconstructed input
      #cost = T.sum(input * T.log(rec) + (1 - input) * T.log(1 - rec))
      cost = T.sum((input - rec) ** 2)

      # the gradients of the cost w/r/t/ each parameter:
      gradients = T.grad(cost, self.params)
      # generate the list of updates
      updates = [
          (param, param - learning_rate * gr)
          for param, gr in zip(self.params, gradients)
      ] + self.theano_rng.updates()

      return (cost, updates)

class AEStack:
  """
  A stack of auto-encoders.
  """
  def __init__(self, numpy_rng, input_size, layer_sizes):
    self.rng = numpy_rng
    self.input_size = input_size
    self.layers = []
    i_size = input_size
    for i in range(len(layer_sizes)):
      o_size = layer_sizes[i]
      self.layers.append(AutoEncoder(numpy_rng, i_size, o_size))
      i_size = o_size

  def get_deconstruct(self, input, limit=-1):
    result = input
    if limit < 0:
      limit = len(self.layers)
    for i in range(limit):
      result = self.layers[i].get_deconstruct(result)
    return result

  def get_reconstruct(self, output, limit=-1):
    result = output
    if limit < 0:
      limit = len(self.layers)
    for i in range(limit-1, -1, -1):
      result = self.layers[i].get_reconstruct(result)
    return result

  def get_training_functions(self, corruption_rates, learning_rates):
    """
    Returns a theano shared variable for use as input and a list of functions
    for training each layer of the stack.
    """
    functions = []
    training_input = T.vector(name="training_input", dtype=theano.config.floatX)
    for i in range(len(self.layers)):
      inp = self.get_deconstruct(training_input, limit=i)
      cost_function, updates = self.layers[i].get_cost_and_updates(
        inp,
        corruption_rates[i],
        learning_rates[i]
      )
      functions.append(
        theano.function(
          inputs = [training_input],
          outputs = cost_function,
          updates = updates,
          name = "training_function_layer_{}".format(i)
        )
      )
    return functions

  def train(self, examples, epoch_counts, corruption_rates, learning_rates):
    """
    Trains the stack on the given examples, given lists of epoch counts,
    corruption rates, and learning rates each equal in length to the number of
    layers in the stack.
    """
    tfs = self.get_training_functions(corruption_rates, learning_rates)
    indices = list(range(examples.get_value(borrow=True).shape[0]))
    start_time = timeit.default_timer()
    for i in range(len(self.layers)):
      # TODO: batches?
      for epoch in range(epoch_counts[i]):
        self.rng.shuffle(indices)
        mincost = None
        for j in indices:
          cost = tfs[i](examples.get_value(borrow=True)[j].reshape(-1))
          if mincost is None or cost < mincost:
            mincost = cost
        debug(
          "... [{}] epoch {} at layer {} done: min cost {:0.3f} ...".format(
            str(datetime.timedelta(seconds=timeit.default_timer()-start_time)),
            epoch + 1,
            i,
            float(mincost)
          )
        )

def explode_example(data, n_layers):
  """
  Returns an array with an extra dimension that encodes that data in the given
  array as a one-hot encoding. The values in the array should all be between 0
  and n_layers (exclusive).
  """
  result = numpy.zeros(
    list(data.shape) + [n_layers],
    dtype=theano.config.floatX
  )
  rs = data.reshape(-1)
  for i, x in enumerate(rs):
    coords = []
    irem = i
    for j in range(len(data.shape)):
      if data.shape[j+1:]:
        b = functools.reduce(lambda x, y: x*y, data.shape[j+1:], 1)
        coords.append(irem // b)
        irem = irem % b
      else:
        coords.append(irem)
    result[tuple(coords + [x])] = 1
  return result

def implode_result(data):
  """
  Returns an array with one fewer dimension than the input, where the input's
  final dimension is taken to represent a one-hot encoding of the desired data.
  """
  dshape = data.shape[:-1]
  n_layers = data.shape[-1]

  result = numpy.zeros(dshape, dtype=theano.config.floatX)
  rs = data.reshape(-1, n_layers)

  for i, enc in enumerate(rs):
    coords = []
    irem = i
    for j in range(len(dshape)):
      if data.shape[j+1:]:
        b = functools.reduce(lambda x, y: x*y, dshape[j+1:], 1)
        coords.append(irem // b)
        irem = irem % b
      else:
        coords.append(irem)
    result[tuple(coords)] = numpy.argmax(enc)
  return result

def fake_palette(size):
  result = {}
  fp = [
    (0xdd, 0x00, 0x00),
    (0xee, 0x99, 0x00),
    (0xff, 0xee, 0x00),
    (0x00, 0x99, 0x00),
    (0x11, 0x22, 0xee),
    (0x00, 0x00, 0x55),
    (0x55, 0x00, 0x99),
  ]
  for i in range(size):
    inc = 0x33 * (i // len(fp))
    e = fp[i%len(fp)]
    result[i] = (
      min(e[0] + inc, 0xff),
      min(e[1] + inc, 0xff),
      min(e[2] + inc, 0xff)
    )
  return result

def build_network(
  examples,
  window_size=8,
  palette_size=16,
  batch_size = 1, # TODO: Implement this
  #layer_sizes = (0.7,),
  #training_epochs = (10,),# (30,),
  #corruption_rates = (0.3,),
  #learning_rates = (0.05,), # (0.005,)
  #layer_sizes = (0.8,0.5),
  #training_epochs = (10,10),
  #corruption_rates = (0.6,0.6),
  #learning_rates = (0.05,0.05),
  layer_sizes = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
  training_epochs = (14, 14, 14, 14, 14, 14, 14, 14, 14),
  corruption_rates = (0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
  learning_rates = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
):
  """
  Builds and trains a network for recognizing image fragments.
  """
  # Calculate input and layer sizes:
  input_size = window_size * window_size * palette_size
  hidden_sizes = [int(input_size * ls) for ls in layer_sizes]

  # Calculate the number of training batches:
  n_train_batches = examples.get_value(borrow=True).shape[0]
  n_train_batches //= batch_size

  # Set up the stacked denoising autoencoders:
  numpy_rng = numpy.random.RandomState(465746)
  net = AEStack(
    numpy_rng=numpy_rng,
    input_size = input_size,
    layer_sizes = hidden_sizes
  )

  # Visualize the network pre-training:
  vis_network(
    net,
    fake_palette(palette_size),
    window_size=window_size,
    outfile="vis-pre.png"
  )

  # Train the network for autoencoding:
  debug("... training the network ...")
  start_time = timeit.default_timer()
  net.train(
    examples,
    training_epochs,
    corruption_rates,
    learning_rates
  )
  end_time = timeit.default_timer()
  debug(
    "... training finished in {} ...".format(
      str(datetime.timedelta(seconds=end_time - start_time))
    )
  )

  return net

def write_image(data, palette, outdir, outfile):
  size = data.shape
  img = Image.new("RGB", size)
  pixels = img.load()

  for x in range(size[0]):
    for y in range(size[1]):
      idx = int(data[x, y])
      if idx in palette:
        pixels[x, y] = palette[idx]
      else:
        pixels[x, y] = (255, 0, 255)

  img.save(os.path.join(outdir, outfile))

def write_grayscale(data, outdir, outfile, nbest=50):
  rs = data.reshape(-1)
  sqside = int((len(rs)**0.5) + 0.99999)
  shape = (sqside, sqside)

  normed = data / numpy.max(data)
  best = numpy.argsort(normed, axis=None)[-nbest:]

  img = Image.new("RGBA", shape)
  pixels = img.load()

  i = 0
  for x in range(sqside):
    for y in range(sqside):
      if i < len(normed):
        g = int(normed[i] * 256)
        r = g
        a = 255
        if i in best:
          r = 0
      else:
        g = 0
        a = 0
      i += 1
      pixels[x, y] = (r, g, g, a)

  img.save(os.path.join(outdir, outfile))

def vis_network(
  net,
  palette,
  window_size=8,
  show=(12, 12),
  outdir="out",
  outfile="vis.png"
):
  palette_size = len(palette)
  input = T.vector(name="input", dtype=theano.config.floatX)
  output = T.vector(name="output", dtype=theano.config.floatX)

  enc = theano.function(
    inputs=[input],
    outputs=net.get_deconstruct(input)
  )

  dec = theano.function(
    inputs=[output],
    outputs=net.get_reconstruct(output)
  )

  encoded = enc(
    numpy.zeros(
      (window_size, window_size, palette_size),
      dtype=theano.config.floatX
    ).reshape(-1)
  )

  exemplars = []

  for i in range(show[0]*show[1]):
    fake = numpy.zeros(encoded.shape, dtype=theano.config.floatX)
    fake = fake.reshape(-1)
    if i >= fake.shape[0]:
      continue
    fake[i] = 1
    fake = fake.reshape(encoded.shape)
    exemplars.append(
      implode_result(
        dec(fake).reshape((window_size, window_size, palette_size))
      )
    )

  result = numpy.full(
    ((window_size+1) * show[0], (window_size+1) * show[1]),
    palette_size,
    dtype=theano.config.floatX
  )
  i = 0
  for x in range(show[0]):
    for y in range(show[1]):
      if i < len(exemplars):
        result[
          x*(window_size+1):(x+1)*(window_size+1) - 1,
          y*(window_size+1):(y+1)*(window_size+1) - 1
        ] = exemplars[i]
      i += 1

  fp = copy.deepcopy(palette)
  fp[palette_size] = (0, 0, 0)

  write_image(
    result,
    fp,
    outdir,
    outfile
  )

def build_munge(examples, net, nbest=2):
  # Express our inputs in terms of the last layer of our neural net, and get
  # the values using the net in its current state:
  n_ex = examples.shape[0]
  exreps, _ = theano.scan(
    fn=lambda i: net.get_deconstruct(examples[i].reshape([-1])),
    sequences=[T.arange(n_ex)]
  )
  exf = theano.function([], exreps)
  exconst = T.constant(exf())

  # An input variable:
  input = T.tensor3(name="input", dtype=theano.config.floatX)

  # Build an expression for computing the net's deconstruction of a variable
  # input, and putting it into a column shape:
  irepcol = net.get_deconstruct(input.reshape([-1])).reshape([-1, 1])

  # An expression for getting the dot products between our representations of
  # each example and our representation of the input:
  dot_products = T.dot(exconst, irepcol)

  # The "best" examples are the ones which are most similar to the encoding of
  # the input:
  whichbest = T.argsort(dot_products, axis=None)[-nbest:].reshape([nbest])
  best = exconst[whichbest,:]
  bestweights = dot_products[whichbest].reshape([nbest])

  # Normalize the nbest entries and combine them:
  norm = bestweights / T.sum(bestweights)
  combined = T.dot(norm, best)

  rec = net.get_reconstruct(combined).reshape(input.shape)

  munge = theano.function(
    name="munge",
    inputs=[input],
    outputs=[dot_products, rec]
  )

  # TODO: Get rid of this?
  #munge = theano.function(
  #  inputs=[input],
  #  outputs=net.get_reconstruct(
  #    net.get_deconstruct(input.reshape(-1))
  #  ).reshape(input.shape)
  #)

  return munge
  

def get_net(data=None, outdir="data", outfile="network.pkl.gz", rebuild=False):
  fn = os.path.join(outdir, outfile)
  if not data:
    # Load data:
    data = load_data()

  ws = data["window_size"]
  hws = int(ws/2)
  ps = len(data["palette"])
  r_palette = data["r_palette"]

  if rebuild or not os.path.exists(fn):
    debug("... building network from scratch ...")
    # Build network:
    net = build_network(
      data["examples"],
      window_size = ws,
      palette_size = ps
    )

    debug("... pickling trained network ...")
    with gzip.open(fn, 'wb') as fout:
      pickle.dump(net, fout)

    debug("... visualizing trained network ...")
    vis_network(
      net,
      r_palette,
      window_size=ws,
      outdir=outdir
    )
  else:
    debug("... loading pickled network ...")
    with gzip.open(fn, 'rb') as fin:
      net = pickle.load(fin)

  return net


def generate_image(
  outdir="out",
  outfile = "result.lvl.png",
  size=(128,64),
  cycles=1,
  show_best_examples=False
):
  # Load data:
  data = load_data()

  ws = data["window_size"]
  hws = int(ws/2)
  ps = len(data["palette"])
  r_palette = data["r_palette"]

  net = get_net(data=data, rebuild=False)

  # TODO: More realistic frequency distribution here?
  result = numpy.random.random_integers(
    0,
    ps - 1,
    (size[0], size[1])
  )

  write_image(result, r_palette, outdir, "pre.lvl.png")

  result = explode_example(result, ps)

  indices = []
  for x in range(0, size[0] - hws, hws):
    for y in range(0, size[1] - hws, hws):
      indices.append((x, y))
  #for x in range(0, size[0] - ws+1, ws):
  #  for y in range(0, size[1] - ws+1, ws):
  #    indices.append((x, y))

  debug("... starting image generation ...")
  munge = build_munge(data["examples"], net)

  for epoch in range(cycles):
    numpy.random.shuffle(indices)
    patch = 0
    for x, y in indices:
      if (patch % 50 == 0):
        debug("... generating patch {}/{} ...".format(patch + 1, len(indices)))
      patch += 1

      if epoch == 0 and patch == 6:
        write_image(
          implode_result(result),
          r_palette,
          outdir,
          "patched.lvl.png"
        )

      dots, result[x:x+ws,y:y+ws,:] = munge(result[x:x+ws,y:y+ws,:])
      #dots, _ = munge(result[x:x+ws,y:y+ws,:])
      #result[x:x+ws,y:y+ws,:] = result[x:x+ws,y:y+ws,:].reshape(-1).reshape((8, 8, 15))
      if epoch == 0 and show_best_examples:
        write_grayscale(
          dots,
          outdir=outdir,
          outfile="dots-{}-{}.png".format(x, y)
        )

    debug("... generation cycle {}/{} completed ...".format(epoch + 1, cycles))

  result = implode_result(result)
  debug("... writing result image ...")
  write_image(result, r_palette, outdir, outfile)
  debug("... done.")

def test_explode(filename="data/examples.pkl.gz"):
  # Load the dataset
  with gzip.open(filename, 'rb') as fin:
    dataset = pickle.load(fin)

  ex = dataset["examples"][0]
  print(ex)
  exr = ex.reshape(8, 8)
  print(exr)
  expl = explode_example(exr, len(dataset["palette"]))
  print(expl)
  impl = implode_result(expl)
  print(impl)

  expl2 = explode_example(ex, len(dataset["palette"]))
  impl2 = implode_result(expl2.reshape((8, 8, 15)))
  print(impl2)
  print(impl2[7, 4], impl2[7, 5])

  img = Image.new("RGB", (8, 8))
  pixels = img.load()

  i = 0
  for x in range(impl2.shape[0]):
    for y in range(impl2.shape[1]):
      g = int(3*impl2[x, y])
      pixels[x, y] = (g, g, g)
      i += 1
      print(impl2[x, y], end=" ")
    print()

  img.save("t.png")

if __name__ == "__main__":
  #test_explode()
  generate_image()
