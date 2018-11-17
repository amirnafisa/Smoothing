# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)


import math
import random
import re
import sys
import numpy as np

BOS = 'BOS'   # special word type for context at Beginning Of Sequence
EOS = 'EOS'   # special word type for observed token at End Of Sequence
OOV = 'OOV'    # special word type for all Out-Of-Vocabulary words
OOL = 'OOL'    # special word type for all Out-Of-Lexicon words
DEFAULT_TRAINING_DIR = "/usr/local/data/cs465/hw-lm/All_Training/"
OOV_THRESHOLD = 3  # minimum number of occurrence for a word to be considered in-vocabulary


# TODO for TA: Maybe we should use inheritance instead of condition on the
# smoother (similar to the Java code).
class LanguageModel:
  def __init__(self):
    # The variables below all correspond to quantities discussed in the assignment.
    # For log-linear or Witten-Bell smoothing, you will need to define some 
    # additional global variables.
    self.smoother = None       # type of smoother we're using
    self.lambdap = None        # lambda or C parameter used by some smoothers

    # The word vector for w can be found at self.vectors[w].
    # You can check if a word is contained in the lexicon using
    #    if w in self.vectors:
    self.vectors = None    # loaded using read_vectors()

    self.vocab = None    # set of words included in the vocabulary
    self.vocab_size = None  # V: the total vocab size including OOV.

    self.tokens = None      # the c(...) function
    self.types_after = None # the T(...) function

    self.progress = 0        # for the progress bar

    self.bigrams = None
    self.trigrams = None
    
    # the two weight matrices X and Y used in log linear model
    # They are initialized in train() function and represented as two
    # dimensional lists.
    self.U, self.V = None, None
    self.Skiptwogram, self.Skipthreegram = None, None

    # self.tokens[(x, y, z)] = # of times that xyz was observed during training.
    # self.tokens[(y, z)]    = # of times that yz was observed during training.
    # self.tokens[z]         = # of times that z was observed during training.
    # self.tokens[""]        = # of tokens observed during training.
    #
    # self.types_after[(x, y)]  = # of distinct word types that were
    #                             observed to follow xy during training.
    # self.types_after[y]       = # of distinct word types that were
    #                             observed to follow y during training.
    # self.types_after[""]      = # of distinct word types observed during training.

    #Adding for WB Smoothing
    self.sum_tr = {}
    self.sum_bi = {}
    #10h question for t exact once match change
    self.dict_t_once = {}

  def prob(self, x, y, z, s = None, t = None):
    """Computes a smoothed estimate of the trigram probability p(z | x,y)
    according to the language model.
    """

    if self.smoother == "UNIFORM":
      return float(1) / self.vocab_size
    elif self.smoother == "ADDL":
      if x not in self.vocab:
        x = OOV
      if y not in self.vocab:
        y = OOV
      if z not in self.vocab:
        z = OOV
      return ((self.tokens.get((x, y, z), 0) + self.lambdap) /
        (self.tokens.get((x, y), 0) + self.lambdap * self.vocab_size))

      # Notice that summing the numerator over all values of typeZ
      # will give the denominator.  Therefore, summing up the quotient
      # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
      # as is required for any probability function.

    elif self.smoother == "BACKOFF_ADDL":
      if x not in self.vocab:
        x = OOV
      if y not in self.vocab:
        y = OOV
      if z not in self.vocab:
        z = OOV

      p_oov = 1 / self.vocab_size

      p_z   = (self.tokens.get((z), 0) + self.lambdap*self.vocab_size * p_oov) / (self.tokens[""] + self.lambdap * self.vocab_size)
      
      p_yz  = (self.tokens.get((y, z), 0) + self.lambdap*self.vocab_size * p_z) / (self.tokens.get((y), 0) + self.lambdap * self.vocab_size)
      
      p_xyz = (self.tokens.get((x, y, z), 0) + self.lambdap*self.vocab_size * p_yz) / (self.tokens.get((x, y), 0) + self.lambdap * self.vocab_size)
      
      return p_xyz

      
    elif self.smoother == "BACKOFF_WB":

      if x not in self.vocab:
        x = OOV
      if y not in self.vocab:
        y = OOV
      if z not in self.vocab:
        z = OOV

      #c_* is the counts of *
      #N is the total number of tokens
      c_xyz = float( self.tokens.get((x, y, z), 0))
      c_xy  = float( self.tokens.get((x, y), 0))
      c_yz  = float( self.tokens.get((y, z), 0))
      c_y   = float( self.tokens.get((y), 0))
      c_z   = float( self.tokens.get((z), 0))
      N     = float( self.tokens[""])

      #types afters
      t_xy = float(self.types_after.get((x, y), 0) )   
      t_y  = float(self.types_after.get((y), 0) )
      t    = float(self.types_after[""] - 1) #currently t includes oov and hence 1 is subtracted here


      #types afters for exact once change
      #Modification used for only testing the last part of the question 10 with witten bell 
      #Uncomment the following lines of code, but beware it will return 0 probability becasue of which the log will throw an error
      #################################################
      #if (x,y) not in self.dict_t_once:
      #  t_xy = 0
      #else:
      #  t_xy = self.dict_t_once[(x,y)]
      #################################################
     
      

      #Alphas and probabilities
      a = (t / (t + N)) * (1 / (self.vocab_size - t))   
      if c_z > 0:
        p_z = (c_z / (N + t))
      else:
        p_z = a

        
      if c_y == 0:
        a_y = 1
        p_yz = p_z
      else:
        a_y = self.sum_bi[(y)] / (c_y + self.sum_bi[(y)])
        if c_yz > 0:
          p_yz = c_yz / (c_y + self.sum_bi[(y)])
        else:
          p_yz = a_y * p_z

      
      if c_xy == 0:
        a_xy = 1
        p_xyz = p_z
      else:
        a_xy = self.sum_tr[(x,y)] / (c_xy + self.sum_tr[(x,y)])
        if c_xyz > 0:
          p_xyz = c_xyz / (c_xy + self.sum_tr[(x,y)])
        else:
          p_xyz = a_xy * p_yz

      return p_xyz

    elif self.smoother == 'LOGLINEAR':

      if x not in self.vectors:
        x = OOL
      if y not in self.vectors:
        y = OOL
      if z not in self.vectors:
        z = OOL

      try:
        x_vector, y_vector, z_vector = self.vectors[x], self.vectors[y], self.vectors[z]
      except:
        print("exception caught while loading vectors")
        sys.exit()
      denom,num_dict =  self.calc_num_denom_loglinear(x_vector, y_vector)
      return num_dict[z]/denom

    elif self.smoother == 'IMPROVED':


      if x not in self.vectors:
        x = OOL
      if y not in self.vectors:
        y = OOL
      if z not in self.vectors:
        z = OOL
      if s not in self.vectors:
        s = OOL
      if t not in self.vectors:
        t = OOL

      x_vector, y_vector, z_vector = self.vectors[x], self.vectors[y], self.vectors[z]
      s_vector, t_vector = self.vectors[s], self.vectors[t]

      denom,num_dict =  self.calc_num_denom_loglinear(x_vector, y_vector, s_vector, t_vector)
      return num_dict[z]/denom

    else:
      sys.exit("%s has some weird value" % self.smoother)

  def filelogprob(self, filename):
    """Compute the log probability of the sequence of tokens in file.
    NOTE: we use natural log for our internal computation.  You will want to
    divide this number by log(2) when reporting log probabilities.
    """
    logprob = 0.0
    x, y = BOS, BOS
    corpus = self.open_corpus(filename)
    for line in corpus:
      for z in line.split():
        prob = self.prob(x, y, z)
        logprob += math.log(prob)
        x = y
        y = z
    logprob += math.log(self.prob(x, y, EOS))
    corpus.close()
    return logprob

  def read_vectors(self, filename):
    """Read word vectors from an external file.  The vectors are saved as
    arrays in a dictionary self.vectors.
    """
    with open(filename) as infile:
      header = infile.readline()
      self.dim = int(header.split()[-1])
      self.vectors = {}
      for line in infile:
        arr = line.split()
        word = arr.pop(0)
        self.vectors[word] = np.asarray([float(x) for x in arr])

  def calc_num_denom_loglinear(self, x_vector, y_vector, s_vector = None, t_vector = None):
    denom = 0
    z_prime_num_dict = {}
    if s_vector is None:
      s_vector = x_vector *0
      t_vector = x_vector *0
    for z_prime, z_prime_vector in self.vectors.items():
      try:
        total = np.dot(np.dot(np.transpose(x_vector), self.U), z_prime_vector) + \
          np.dot(np.dot(np.transpose(y_vector), self.V), z_prime_vector)
        if self.smoother ==  'IMPROVED':
          total += np.dot(np.dot(np.transpose(t_vector), self.Skiptwogram), z_prime_vector) + \
          np.dot(np.dot(np.transpose(s_vector), self.Skipthreegram), z_prime_vector)
        z_prime_num = math.exp(total)
      except OverflowError:
        z_prime_num = float("inf")
      z_prime_num_dict[z_prime] = (z_prime_num)

      denom += z_prime_num
    return denom, z_prime_num_dict

  def train (self, filename, epochs = 10):
    """Read the training corpus and collect any information that will be needed
    by the prob function later on.  Tokens are whitespace-delimited.

    Note: In a real system, you wouldn't do this work every time you ran the
    testing program. You'd do it only once and save the trained model to disk
    in some format.
    """
    sys.stderr.write("Training from corpus %s\n" % filename)

    # Clear out any previous training
    self.tokens = { }
    self.types_after = { }
    self.bigrams = []
    self.trigrams = []

    # While training, we'll keep track of all the trigram and bigram types
    # we observe.  You'll need these lists only for Witten-Bell backoff.
    # The real work:
    # accumulate the type and token counts into the global hash tables.

    # If vocab size has not been set, build the vocabulary from training corpus
    if self.vocab_size is None:
      self.set_vocab_size(filename)

    # We save the corpus in memory to a list tokens_list.  Notice that we
    # appended two BOS at the front of the list and a EOS at the end.  You
    # will need to add more BOS tokens if you want to use a longer context than
    # trigram.
    x, y = BOS, BOS  # Previous two words.  Initialized as "beginning of sequence"
    # count the BOS context
    self.tokens[(x, y)] = 1
    self.tokens[y] = 1

    tokens_list = [x, y]  # the corpus saved as a list
    corpus = self.open_corpus(filename)
    for line in corpus:
      for z in line.split():
        # substitute out-of-vocabulary words with OOV symbol
        if z not in self.vocab:
          z = OOV
        # substitute out-of-lexicon words with OOL symbol (only for log-linear models)
        if (self.smoother == 'LOGLINEAR' or self.smoother ==  'IMPROVED') and z not in self.vectors:
          z = OOL
        self.count(x, y, z)
        self.show_progress()
        x=y; y=z
        tokens_list.append(z)
    tokens_list.append(EOS)   # append a end-of-sequence symbol 
    sys.stderr.write('\n')    # done printing progress dots "...."
    self.count(x, y, EOS)     # count EOS "end of sequence" token after the final context
    corpus.close()

    #Adding some code for WB Smoothing

    
    for tr in self.trigrams:
      tr_xy = (tr[0], tr[1])
      tr_yz = (tr[1], tr[2])
      if tr_xy not in self.sum_tr:
        self.sum_tr[tr_xy] = 0
      self.sum_tr[tr_xy] += self.tokens.get(tr_yz, 0)
      
      #Added for last part of Q10, to try if t for "exact once" counts
      if self.tokens.get(tr, 0) == 1:
        if tr_xy not in self.dict_t_once:
          self.dict_t_once[tr_xy] = 0
        self.dict_t_once[tr_xy] += 1
      ##############

    for bi in self.bigrams:
      bi_y = (bi[0])
      bi_z = (bi[1])
      if bi_y not in self.sum_bi:
        self.sum_bi[bi_y] = 0
      self.sum_bi[bi_y] += self.tokens.get(bi_z, 0)

    if self.smoother == 'LOGLINEAR' or self.smoother ==  'IMPROVED' :
      # Train the log-linear model using SGD.

      # Initialize parameters
      self.U = np.zeros([self.dim,self.dim])
      self.V = np.zeros([self.dim,self.dim])
      self.Skiptwogram= np.zeros([self.dim,self.dim])
      self.Skipthreegram = np.zeros([self.dim,self.dim])

      # Optimization parameters
      gamma0 = 0.01  # initial learning rate, used to compute actual learning rate
      epochs = epochs  # number of passes

      self.N = len(tokens_list) - 2  # number of training instances

      # ******** COMMENT *********
      # In log-linear model, you will have to do some additional computation at
      # this point.  You can enumerate over all training trigrams as following.
      #
      # for i in range(2, len(tokens_list)):
      #   x, y, z = self.vectors[tokens_list[i - 2]], self.vectors[tokens_list[i - 1]], self.vectors[tokens_list[i]]
      #   prob = self.calculate_loglinprob(x, y, z)

      #
      # Note1: self.lambdap is the regularizer constant C
      # Note2: You can use self.show_progress() to log progress.
      #
      # **************************

      sys.stderr.write("Start optimizing.\n")

      #####################
      # TODO: Implement your SGD here
      #####################

      itert_count = 0
      for epoch in range(epochs):

        #loop over N
        ngram = 2
        if self.smoother == ('IMPROVED'):
          ngram = 4
        for i in range(ngram, len(tokens_list)):

          # print("(iter {} of {})\n".nformat(i, len(tokens_list)))

          gamma = gamma0 / (1 + gamma0 * (2*self.lambdap / self.N) * itert_count)

          #read next trigram
          x, y, z = tokens_list[i - 2], tokens_list[i - 1], tokens_list[i]
          x_vector, y_vector, z_vector = self.vectors[x], self.vectors[y], self.vectors[z]

          if self.smoother == ('IMPROVED'):
            s, t = tokens_list[i - 4], tokens_list[i - 3]
            s_vector, t_vector = self.vectors[s], self.vectors[t]

          # observed_count = self.tokens.get((x, y, z), 0)
          # prob, numerator, denominator, num_prime_list = self.calculate_loglinprob(x_vector, y_vector, z_vector)
          updated_U = self.U
          updated_V = self.V
          updated_Skiptwogram = self.Skiptwogram
          updated_Skipthreegram = self.Skipthreegram

          denom, z_prime_num_dict = self.calc_num_denom_loglinear(x_vector, y_vector)


          for j in range(self.dim):
            for m in range(self.dim):
              grad_Fu_SUMz = 0
              grad_Fv_SUMz = 0
              grad_FSkiptwogram_SUMz = 0
              grad_FSkipthreegram_SUMz =0

              for z_prime, z_prime_vector in self.vectors.items():
                prob_z_prime = z_prime_num_dict[z_prime] / denom

                grad_Fu_SUMz += prob_z_prime * x_vector[j] * z_prime_vector[m]
                grad_Fv_SUMz += prob_z_prime * y_vector[j] * z_prime_vector[m]
                if self.smoother == ('IMPROVED'):
                  grad_FSkiptwogram_SUMz += prob_z_prime * t_vector[j] * z_prime_vector[m]
                  grad_FSkipthreegram_SUMz += prob_z_prime * s_vector[j] * z_prime_vector[m]

              grad_Fu_jm = x_vector[j] * z_vector[m] - grad_Fu_SUMz - (2*self.lambdap/self.N * self.U[j][m])
              grad_Fv_jm = y_vector[j] * z_vector[m] - grad_Fv_SUMz - (2*self.lambdap/self.N * self.V[j][m])
              if self.smoother == ('IMPROVED'):
                grad_FSkiptwogram_jm   = t_vector[j] * z_vector[m] - grad_FSkiptwogram_SUMz - (2*self.lambdap/self.N * self.Skiptwogram[j][m])
                grad_FSkipthreegram_jm = s_vector[j] * z_vector[m] - grad_FSkipthreegram_SUMz - (2*self.lambdap/self.N * self.Skipthreegram[j][m])
                updated_Skiptwogram[j][m] = self.Skiptwogram[j][m] + gamma * grad_FSkiptwogram_jm
                updated_Skipthreegram[j][m] = self.Skipthreegram[j][m] + gamma * grad_FSkipthreegram_jm

              updated_U[j][m] = self.U[j][m] + gamma * grad_Fu_jm
              updated_V[j][m] = self.V[j][m] + gamma * grad_Fv_jm

          self.U = updated_U
          self.V = updated_V
          if self.smoother == ('IMPROVED'):
            self.Skiptwogram = updated_Skiptwogram
            self.Skipthreegram = updated_Skipthreegram

          itert_count+=1
        # print("Finished Epoch %d\n" % epoch)
        ll = 0
        reg_sq = np.power(self.U, 2) + np.power(self.V, 2)
        if self.smoother == ('IMPROVED'):
          reg_sq+= np.power(self.Skiptwogram, 2) + np.power(self.Skipthreegram, 2)

        reg = self.lambdap * np.sum(reg_sq , keepdims=False)

        for i in range(ngram, len(tokens_list)):
          x, y, z = tokens_list[i - 2], tokens_list[i - 1], tokens_list[i]
          s,t = None, None
          if self.smoother == ('IMPROVED'):
            s, t = tokens_list[i - 4], tokens_list[i - 3]
          ll += math.log(self.prob(x, y, z, s,t))

        F = (ll -reg ) /self.N

        print("Epoch "+str(epoch)+": F=" + str(F))

    sys.stderr.write("Finished training on %d tokens\n" % self.tokens[""])

  def count(self, x, y, z):
    """Count the n-grams.  In the perl version, this was an inner function.
    For now, I am just using a class variable to store the found tri-
    and bi- grams.
    """
    self.tokens[(x, y, z)] = self.tokens.get((x, y, z), 0) + 1
    if self.tokens[(x, y, z)] == 1:       # first time we've seen trigram xyz
      self.trigrams.append((x, y, z))
      self.types_after[(x, y)] = self.types_after.get((x, y), 0) + 1

    self.tokens[(y, z)] = self.tokens.get((y, z), 0) + 1
    if self.tokens[(y, z)] == 1:        # first time we've seen bigram yz
      self.bigrams.append((y, z))
      self.types_after[y] = self.types_after.get(y, 0) + 1

    self.tokens[z] = self.tokens.get(z, 0) + 1
    if self.tokens[z] == 1:         # first time we've seen unigram z
      self.types_after[''] = self.types_after.get('', 0) + 1
    #  self.vocab_size += 1

    self.tokens[''] = self.tokens.get('', 0) + 1  # the zero-gram


  def set_vocab_size(self, *files):
    """When you do text categorization, call this function on the two
    corpora in order to set the global vocab_size to the size
    of the single common vocabulary.

     """
    count = {} # count of each word

    for filename in files:
      corpus = self.open_corpus(filename)
      for line in corpus:
        for z in line.split():
          count[z] = count.get(z, 0) + 1
          self.show_progress();
      corpus.close()
    self.vocab = set(w for w in count if count[w] >= OOV_THRESHOLD)

    self.vocab.add(OOV)  # add OOV to vocab
    self.vocab.add(EOS)  # add EOS to vocab (but not BOS, which is never a possible outcome but only a context)
    sys.stderr.write('\n')    # done printing progress dots "...."

    if self.vocab_size is not None:
      sys.stderr.write("Warning: vocab_size already set; set_vocab_size changing it\n")
    self.vocab_size = len(self.vocab)
    sys.stderr.write("Vocabulary size is %d types including OOV and EOS\n"
                      % self.vocab_size)

  def set_smoother(self, arg):
    """Sets smoother type and lambda from a string passed in by the user on the
    command line.
    """
    r = re.compile('^(.*?)-?([0-9.]*)$')
    m = r.match(arg)
    
    if not m.lastindex:
      sys.exit("Smoother regular expression failed for %s" % arg)
    else:
      smoother_name = m.group(1)
      if m.lastindex >= 2 and len(m.group(2)):
        lambda_arg = m.group(2)
        self.lambdap = float(lambda_arg)
      else:
        self.lambdap = None

    if smoother_name.lower() == 'uniform':
      self.smoother = "UNIFORM"
    elif smoother_name.lower() == 'add':
      self.smoother = "ADDL"
    elif smoother_name.lower() == 'backoff_add':
      self.smoother = "BACKOFF_ADDL"
    elif smoother_name.lower() == 'backoff_wb':
      self.smoother = "BACKOFF_WB"
    elif smoother_name.lower() == ('loglinear' or "loglin"):
      self.smoother = "LOGLINEAR"
    elif smoother_name.lower() == 'improved':
      self.smoother = "IMPROVED"
    else:
      sys.exit("Don't recognize smoother name '%s'" % smoother_name)
    
    if self.lambdap is None and self.smoother.find('ADDL') != -1:
      sys.exit('You must include a non-negative lambda value in smoother name "%s"' % arg)

  def open_corpus(self, filename):
    """Associates handle CORPUS with the training corpus named by filename."""
    try:
      corpus = open(filename, "r")
    except IOError as err:
      try:
        corpus = open(DEFAULT_TRAINING_DIR + filename, "r")
      except IOError as err:
        sys.exit("Couldn't open corpus at %s or %s" % (filename,
                 DEFAULT_TRAINING_DIR + filename))
    return corpus

  def num_tokens(self, filename):
    corpus = self.open_corpus(filename)
    num_tokens = sum([len(l.split()) for l in corpus]) + 1

    return num_tokens

  def show_progress(self, freq=5000):
    """Print a dot to stderr every 5000 calls (frequency can be changed)."""
    self.progress += 1
    if self.progress % freq == 1:
      sys.stderr.write('.')

  @classmethod
  def load(cls, fname):
    try:
      import cPickle as pickle
    except:
      import pickle
    fh = open(fname, mode='rb')
    loaded = pickle.load(fh)
    fh.close()
    return loaded

  def save(self, fname):
    try:
      import cPickle as pickle
    except:
      import pickle
    with open(fname, mode='wb') as fh:
      pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

  def probbi(self, y, z):
    """Computes a smoothed estimate of the trigram probability p(z |y)
    according to the language model.
    """

    if self.smoother == "UNIFORM":
      return float(1) / self.vocab_size
    elif self.smoother == "ADDL":
      if y not in self.vocab:
        y = OOV
      if z not in self.vocab:
        z = OOV
      return ((self.tokens.get((y, z), 0) + self.lambdap) /
        (self.tokens.get((y), 0) + self.lambdap * self.vocab_size))
    elif self.smoother == "BACKOFF_ADDL":
      if y not in self.vocab:
        y = OOV
      if z not in self.vocab:
        z = OOV

      p_oov = 1 / self.vocab_size

      p_z   = (self.tokens.get((z), 0) + self.lambdap*self.vocab_size * p_oov) / (self.tokens[""] + self.lambdap * self.vocab_size)
      
      p_yz  = (self.tokens.get((y, z), 0) + self.lambdap*self.vocab_size * p_z) / (self.tokens.get((y), 0) + self.lambdap * self.vocab_size)
      
      
      return p_yz

      
    elif self.smoother == "BACKOFF_WB":

      #N is the total number of tokens
      c_yz  = float( self.tokens.get((y, z), 0))
      c_y   = float( self.tokens.get((y), 0))
      c_z   = float( self.tokens.get((z), 0))
      N     = float( self.tokens[""])

      #types afters
      t_y  = float(self.types_after.get((y), 0) )
      t    = float(self.types_after[""] - 1) #currently t includes oov and hence 1 is subtracted here

      #Alphas and probabilities
      a = (t / (t + N)) * (1 / (self.vocab_size - t))   
      if c_z > 0:
        p_z = (c_z / (N + t))
      else:
        p_z = a

      if c_y == 0:
        a_y = 1
        p_yz = p_z
      else:
        a_y = self.sum_bi[(y)] / (c_y + self.sum_bi[(y)])
        if c_yz > 0:
          p_yz = c_yz / (c_y + self.sum_bi[(y)])
        else:
          p_yz = a_y * p_z

      return p_yz

    elif self.smoother == "LOGLINEAR":
      sys.exit("LOGLINEAR is not implemented yet (that's your job!)")
    else:
      sys.exit("%s has some weird value" % self.smoother)

  def probuni(self, z):
    """Computes a smoothed estimate of the trigram probability p(z)
    according to the language model.
    """

    if self.smoother == "UNIFORM":
      return float(1) / self.vocab_size
    elif self.smoother == "ADDL":
      if z not in self.vocab:
        z = OOV
      return ((self.tokens.get((z), 0) + self.lambdap) /
        (self.tokens[""] + self.lambdap * self.vocab_size))

    elif self.smoother == "BACKOFF_ADDL":
      if z not in self.vocab:
        z = OOV

      p_oov = 1 / self.vocab_size

      p_z   = (self.tokens.get((z), 0) + self.lambdap*self.vocab_size * p_oov) / (self.tokens[""] + self.lambdap * self.vocab_size)
      
      
      return p_z

      
    elif self.smoother == "BACKOFF_WB":

      #c_* is the counts of *
      #N is the total number of tokens
      c_z   = float( self.tokens.get((z), 0))
      N     = float( self.tokens[""])

      #types afters
      t    = float(self.types_after[""] - 1) #currently t includes oov and hence 1 is subtracted here
      #Alphas and probabilities
      a = (t / (t + N)) * (1 / (self.vocab_size - t))   
      if c_z > 0:
        p_z = (c_z / (N + t))
      else:
        p_z = a

      return p_z

    elif self.smoother == "LOGLINEAR":
      sys.exit("LOGLINEAR is not implemented yet (that's your job!)")
    else:
      sys.exit("%s has some weird value" % self.smoother)

#Log trigram Probabilities for speech rec problem
  def logprob(self, line):
    """Compute the log probability of the sequence of tokens in file.
    NOTE: we use natural log for our internal computation.  You will want to
    divide this number by log(2) when reporting log probabilities.
    """
    logprob = 0.0
    x, y = BOS, BOS
    for z in line.split():
      prob = self.prob(x, y, z)
      logprob += math.log(prob)
      x = y
      y = z
    logprob += math.log(self.prob(x, y, EOS))
    return logprob

#Log bigram Probabilities for speech rec problem
  def logprobbi(self, line):
    """Compute the log probability of the sequence of tokens in file.
    NOTE: we use natural log for our internal computation.  You will want to
    divide this number by log(2) when reporting log probabilities.
    """
    logprob = 0.0
    y = BOS
    for z in line.split():
      prob = self.probbi(y, z)
      logprob += math.log(prob)
      y = z
    logprob += math.log(self.probbi(y, EOS))
    return logprob

#log unigram probabilities for speech rec problem
  def logprobuni(self, line):
    """Compute the log probability of the sequence of tokens in file.
    NOTE: we use natural log for our internal computation.  You will want to
    divide this number by log(2) when reporting log probabilities.
    """
    logprob = 0.0
    for z in line.split():
      prob = self.probuni(z)
      logprob += math.log(prob)
    logprob += math.log(self.probuni(EOS))
    return logprob
