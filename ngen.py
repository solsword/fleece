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

class NeuroLayer:
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
    self.weights = theano.shared(
      value=numpy.asarray(
        numpy_rng.uniform(
          low=-4 * numpy.sqrt(6. / (input_size + output_size)),
          high=4 * numpy.sqrt(6. / (input_size + output_size)),
          size=(input_size*2, output_size*2)
        )[wsx:wsx+input_size,wsy:wsy+output_size],
        dtype=theano.config.floatX
      ),
      name='weights',
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

    # Offsets for reconstruction (output -> input):
    # Note that the weights are shared for both directions.
    self.re_offsets = theano.shared(
      value=numpy.zeros(
        input_size,
        dtype=theano.config.floatX
      ),
      name='re_offsets',
      borrow=True
    )

    self.params = [
      self.weights,
      self.de_offsets,
      self.re_offsets
    ]

  # Evaluation functions:
  def get_deconstruct(self, input):
    return T.nnet.sigmoid(
      T.dot(input, self.weights) + self.de_offsets
    )

  def get_reconstruct(self, output):
    return T.nnet.sigmoid(
      T.dot(output, self.weights.T) + self.re_offsets
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

class NeuralNet:
  """
  A stack of auto-encoders.
  """
  def __init__(self, numpy_rng, input_size, layer_sizes, output_size):
    self.rng = numpy_rng
    self.input_size = input_size
    self.output_size = output_size
    self.layers = []
    i_size = input_size
    for i in range(len(layer_sizes)):
      o_size = layer_sizes[i]
      self.layers.append(NeuroLayer(numpy_rng, i_size, o_size))
      i_size = o_size
    self.layers.append(NeuroLayer(numpy_rng, i_size, output_size))

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

  def get_training_functions(self, corruption_rates, ae_learning_rates):
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
        ae_learning_rates[i]
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

  def get_specialization_function(self, input, cv_extract, learning_rate):
    """
    Returns a theano function that uses an example to specialize the network by
    training it to predict the region of the input selected by the given
    cv_extract function.
    """
    pfunc = self.get_deconstruct(input)

    compare_to = cv_extract(input)

    cost = T.sum((compare_to - pfunc) ** 2)

    params = []
    for l in self.layers:
      params.extend(l.params[:-1]) # ignore the reconstruction offsets

    gradients = T.grad(cost, params)

    # generate the list of updates
    updates = [
        (param, param - learning_rate * gr)
        for param, gr in zip(params, gradients)
    ]
    # TODO: are these really unnecessary here?
    # + [l.theano_rng.updates() for l in self.layers]

    return theano.function(
      inputs = [input],
      outputs = cost,
      updates = updates,
      name = "specialization_function"
    )

  def pretrain(self, examples, epoch_counts, corruption_rates, learning_rates):
    """
    Trains the network for autoencoding on the given examples, given lists of
    epoch counts, corruption rates, and learning rates each equal in length to
    the number of layers in the stack.
    """
    tfs = self.get_training_functions(corruption_rates, learning_rates)
    indices = list(range(examples.get_value(borrow=True).shape[0]))
    start_time = timeit.default_timer()
    for i in range(len(self.layers)):
      # TODO: batches?
      for epoch in range(epoch_counts[i]):
        self.rng.shuffle(indices)
        costs = []
        for j in indices:
          cost = tfs[i](examples.get_value(borrow=True)[j].reshape(-1))
          costs.append(cost)
        debug(
          "... [{}] epoch {: 3d} at layer {: 2d} done {} ...".format(
            str(datetime.timedelta(seconds=timeit.default_timer()-start_time)),
            epoch + 1,
            i,
            "(min/avg cost {:0.3f}/{:0.3f})".format(
              float(min(costs)),
              float(sum(costs)/float(len(costs))),
            )
          )
        )

  def train(self, examples, cv_extract, epochs, learning_rate):
    """
    Specializes the network for prediction on the given examples, using the
    given center extract function, the given number of epochs, and the given
    learning rate.
    """
    input = T.vector(name="training_input", dtype=theano.config.floatX)
    tf = self.get_specialization_function(input, cv_extract, learning_rate)
    indices = list(range(examples.get_value(borrow=True).shape[0]))
    start_time = timeit.default_timer()
    # TODO: batches?
    for epoch in range(epochs):
      self.rng.shuffle(indices)
      costs = []
      for j in indices:
        cost = tf(examples.get_value(borrow=True)[j].reshape(-1))
        costs.append(cost)
      debug(
        "... [{}] epoch {: 3d} done {} ...".format(
          str(datetime.timedelta(seconds=timeit.default_timer()-start_time)),
          epoch + 1,
          "(min/avg cost {:0.3f}/{:0.3f})".format(
            float(float(min(costs))),
            float(float(sum(costs)/float(len(costs))))
          )
        )
      )

def get_central_values(flat_input, input_size, center_size, palette_size):
  """
  Takes a flat array which is assumed to represent input_size by input_size by
  palette_size data, and returns a flat array that represents the center_size
  by center_size central values of the original array.
  """
  lc = input_size//2 - center_size//2
  rs = flat_input.reshape((input_size, input_size, palette_size))
  sel = rs[lc:lc+center_size, lc:lc+center_size, :]
  return sel.reshape([-1])

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
  predict_size=2,
  palette_size=16,
  batch_size = 1, # TODO: Implement this
  #layer_sizes = (0.2,),
  #ae_epochs = (1,1,),# (30,),
  #corruption_rates = (0.3,0.3,),
  #ae_learning_rates = (0.05,0.05,), # (0.005,)
  #sp_epochs = 1,
  #sp_learning_rate = 0.05,
  #layer_sizes = (0.7,),
  #ae_epochs = (5,5,),# (30,),
  #corruption_rates = (0.3,0.3,),
  #ae_learning_rates = (0.05,0.05,), # (0.005,)
  #sp_epochs = 5,
  #sp_learning_rate = 0.05,
  #layer_sizes = (0.8,0.5),
  #ae_epochs = (12, 12, 12),
  #corruption_rates = (0.3, 0.3, 0.2),
  #ae_learning_rates = (0.05, 0.05, 0.05),
  #sp_epochs = 20,
  #sp_learning_rate = 0.05,
  layer_sizes = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
  ae_epochs = (14, 14, 14, 14, 14, 14, 14, 14, 14, 14),
  corruption_rates = (0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
  ae_learning_rates = (
    0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04
  ),
  sp_epochs = 20,
  sp_learning_rate = 0.05,
):
  """
  Builds and trains a network for recognizing image fragments.
  """
  # Calculate input and layer sizes:
  input_size = window_size * window_size * palette_size
  hidden_sizes = [int(input_size * ls) for ls in layer_sizes]
  final_size = predict_size * predict_size * palette_size

  # Calculate the number of training batches:
  n_train_batches = examples.get_value(borrow=True).shape[0]
  n_train_batches //= batch_size

  # Set up the stacked denoising autoencoders:
  numpy_rng = numpy.random.RandomState(465746)
  net = NeuralNet(
    numpy_rng=numpy_rng,
    input_size = input_size,
    layer_sizes = hidden_sizes,
    output_size = final_size
  )

  # Visualize the network pre-training:
  vis_network(
    net,
    fake_palette(palette_size),
    window_size=window_size,
    outfile="vis-pre.png"
  )

  # Train the network for autoencoding:
  debug("... pretraining the network ...")
  start_time = timeit.default_timer()
  net.pretrain(
    examples,
    ae_epochs,
    corruption_rates,
    ae_learning_rates
  )
  end_time = timeit.default_timer()
  debug(
    "... pretraining finished in {} ...".format(
      str(datetime.timedelta(seconds=end_time - start_time))
    )
  )

  # Specialize the network for generation:
  debug("... specializing the network ...")
  start_time = timeit.default_timer()
  net.train(
    examples,
    lambda a: get_central_values(a, window_size, predict_size, palette_size),
    sp_epochs,
    sp_learning_rate
  )
  end_time = timeit.default_timer()
  debug(
    "... specialization finished in {} ...".format(
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

def build_ae_munge(examples, net, nbest=2):
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

def build_munge(net, patch_size):
  input = T.tensor3(name="input", dtype=theano.config.floatX)
  predict = net.get_deconstruct(input.reshape([-1]))
  result = predict.reshape([patch_size, patch_size, input.shape[2]])
  return theano.function(
    name="munge",
    inputs=[input],
    outputs=result
  )

def get_net(
  data=None,
  outdir="data",
  outfile="network.pkl.gz",
  center_size=2,
  rebuild=False
):
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
      window_size=ws,
      predict_size=center_size,
      palette_size=ps
    )

    debug("... pickling trained network ...")
    with gzip.open(fn, 'wb') as fout:
      pickle.dump(net, fout)

    debug("... visualizing trained network ...")
    vis_network(
      net,
      r_palette,
      window_size=ws
    )
  else:
    debug("... loading pickled network ...")
    with gzip.open(fn, 'rb') as fin:
      net = pickle.load(fin)

  return net


def generate_image(
  outdir="out",
  outfile = "result.lvl.png",
  #size=(128,64),
  size=(32,32),
  patch_size=2,
  step_size=1,
  cycles=1,
  ini="distribution"
):
  # Load data:
  data = load_data()

  ws = data["window_size"]
  hws = int(ws/2)
  ps = len(data["palette"])
  border = data["border"]
  r_palette = data["r_palette"]
  fr_dist = data["fr_dist"]
  exemplar = data["exemplar"]

  net = get_net(data=data, center_size=patch_size, rebuild=False)

  if ini == "random":
    result = numpy.random.random_integers(
      0,
      ps - 2, # avoid the last entry, which is the border value
      (size[0] + 2*ws, size[1] + 2*ws)
    )
  elif ini == "shuffle":
    result = numpy.zeros((size[0] + 2*ws, size[1] + 2*ws))
    ex = exemplar.reshape(-1)
    numpy.random.shuffle(ex)
    ex = ex.reshape(exemplar.shape)[:size[0],:size[1]]
    result[ws:size[0]+ws,ws:size[1]+ws] = ex
  elif ini == "distribution":
    result = numpy.zeros((size[0] + 2*ws, size[1] + 2*ws))
    for x in range(ws, size[0] + ws):
      for y in range(ws, size[1] + ws):
        sofar = 0
        choice = numpy.random.uniform(0, 1)
        for w, v in fr_dist:
          sofar += w
          if sofar >= choice:
            result[x, y] = v
            break

  # Set our border data to the border value:
  for x in range(ws):
    for y in range(size[1] + 2*ws):
      result[x,y] = border
  for x in range(size[0] + ws, size[0] + 2*ws):
    for y in range(size[1] + 2*ws):
      result[x,y] = border
  for y in range(ws):
    for x in range(size[0] + 2*ws):
      result[x,y] = border
  for y in range(size[1] + ws, size[1] + 2*ws):
    for x in range(size[0] + 2*ws):
      result[x,y] = border

  write_image(result, r_palette, outdir, "pre.lvl.png")

  result = explode_example(result, ps)

  indices = []
  for x in range(ws - hws, size[0] + ws - hws, step_size):
    for y in range(ws - hws, size[1] + ws - hws, step_size):
      indices.append((x, y))

  debug("... starting image generation ...")
  munge = build_munge(net, patch_size)

  for epoch in range(cycles):
    numpy.random.shuffle(indices)
    patch = 0
    for x, y in indices:
      if (patch % 50 == 0):
        debug("... generating patch {}/{} ...".format(patch + 1, len(indices)))
      patch += 1

      if epoch == 0 and patch == 20:
        write_image(
          implode_result(result),
          r_palette,
          outdir,
          "patched.lvl.png"
        )

      result[
        x + ws//2 - patch_size//2:x + ws//2 - patch_size//2 + patch_size,
        y + ws//2 - patch_size//2:y + ws//2 - patch_size//2 + patch_size,
        :
      ] = munge(result[x:x+ws,y:y+ws,:])

    debug("... generation cycle {}/{} completed ...".format(epoch + 1, cycles))

  result = implode_result(result)
  debug("... writing result image ...")
  write_image(result, r_palette, outdir, outfile)
  debug("... done.")

def test_explode(filename="data/examples.pkl.gz", size=8):
  # Load the dataset
  with gzip.open(filename, 'rb') as fin:
    dataset = pickle.load(fin)

  ex = dataset["examples"][0]
  print(ex)
  exr = ex.reshape(size, size)
  print(exr)
  expl = explode_example(exr, len(dataset["palette"]))
  print(expl)
  impl = implode_result(expl)
  print(impl)

  expl2 = explode_example(ex, len(dataset["palette"]))
  impl2 = implode_result(expl2.reshape((size, size, 15)))
  print(impl2)
  print(impl2[7, 4], impl2[7, 5])

  img = Image.new("RGB", (size, size))
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
  #generate_image(cycles=1, ini="distribution")
  generate_image(cycles=1, ini="shuffle")
