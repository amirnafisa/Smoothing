#!/usr/bin/env python

# Sample program for hw-lm
# CS465 at Johns Hopkins University.

# Converted to python by Eric Perlman <eric@cs.jhu.edu>

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

from __future__ import print_function

import math
import sys
import re
import Probs

from collections import defaultdict
import hashlib
from os.path import basename
# Computes the log probability of the sequence of tokens in file,
# according to a trigram model.  The training source is specified by
# the currently open corpus, and the smoothing method used by
# prob() is specified by the global variable "smoother". 

def get_model_filename(smoother, lexicon, train_file):
    train_hash = basename(train_file)
    lexicon_hash = basename(lexicon)
    filename = '{}_{}_{}.model'.format(smoother, lexicon_hash, train_hash)
    return filename

def evaluate(testfile, gen):

    if ((re.search(r'dev\/english\/', testfile) and gen == 'spanish') or (re.search(r'dev\/spanish\/', testfile) and gen == 'english')):
        #Return wrong_prediction and length of the file
        return 1, (re.search('length-(.*)/', testfile)).group(1)
    #Return not wrong_prediction and length of the file
    return 0, (re.search('length-(.*)/', testfile)).group(1)

def main():
  course_dir = '/usr/local/data/cs465/'

  if len(sys.argv) < 6 or (sys.argv[1] == 'TRAIN' and len(sys.argv) != 6):

#     print("""
# Prints the log-probability of each file under a smoothed n-gram model.
#
# Usage:   {} TRAIN smoother lexicon trainpath
#          {} TEST smoother lexicon trainpath files...
# Example: {} TRAIN add0.01 {}hw-lm/lexicons/words-10.txt switchboard-small
#          {} TEST add0.01 {}hw-lm/lexicons/words-10.txt switchboard-small {}hw-lm/speech/sample*
#
# Possible values for smoother: uniform, add1, backoff_add1, backoff_wb, loglinear1
#   (the \"1\" in add1/backoff_add1 can be replaced with any real lambda >= 0
#    the \"1\" in loglinear1 can be replaced with any C >= 0 )
# lexicon is the location of the word vector file, which is only used in the loglinear model
# trainpath is the location of the training corpus
#   (the search path for this includes "{}")
# """.format(sys.argv[0], sys.argv[0], sys.argv[0], course_dir, sys.argv[0], course_dir, course_dir, Probs.DEFAULT_TRAINING_DIR))

  mode = sys.argv[1]
  argv = sys.argv[2:]

  smoother = argv.pop(0)
  lexicon = argv.pop(0)
  train_file1 = argv.pop(0)
  train_file2 = argv.pop(0)
  epochs = 10
  if mode == 'TRAIN':

    #Train Model1
    lm1 = Probs.LanguageModel()
    #Comment following line when you want cross entropy reading
    lm1.set_vocab_size(train_file1, train_file2)
    lm1.set_smoother(smoother)
    lm1.read_vectors(lexicon)
    lm1.train(train_file1,epochs)
    lm1.save(get_model_filename(smoother, lexicon, train_file1))

    #Train Model2
    lm2 = Probs.LanguageModel()
    #Comment following line when you want cross entropy reading
    lm2.set_vocab_size(train_file1, train_file2)
    lm2.set_smoother(smoother)
    lm2.read_vectors(lexicon)
    lm2.train(train_file2, epochs)
    lm2.save(get_model_filename(smoother, lexicon, train_file2))
  elif mode == 'TEST':
    if not argv:
      print("warning: no input files specified")
    
    priorprob_corpus1 = float(argv.pop(0))

    #Load parameters of the trained models
    lm1 = Probs.LanguageModel.load(get_model_filename(smoother, lexicon, train_file1))
    lm2 = Probs.LanguageModel.load(get_model_filename(smoother, lexicon, train_file2))

    # We use natural log for our internal computations and that's
    # the kind of log-probability that fileLogProb returns.
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    #Class counters to keep track of number of predictions in each class
    class1_counter = 0
    class2_counter = 0
    #Counter of wrong predictions for evaluation
    wrong_predictions = 0
    total_cross_entropy1 = 0.
    total_cross_entropy2 = 0.
    total_cross_entropy = 0.
    files_length_accuracy = defaultdict(list)
    #Loop for predicting each dev/test file
    for testfile in argv:
      ce1 =  lm1.filelogprob(testfile) / math.log(2)
#      print("#{:g}\t{}".format(ce1, testfile))
      #Number of tokens in the test file used for averaging probability
      token_count = lm1.num_tokens(testfile)
      #Compute posterior probability for class 1
      map1 = ((math.log(priorprob_corpus1) + lm1.filelogprob(testfile)) / math.log(2) ) / token_count


      #Compute posterior probability for class 2
      map2 = ((math.log(1 - priorprob_corpus1) +  lm2.filelogprob(testfile)) / math.log(2)) / token_count
      ce2 =  lm2.filelogprob(testfile) / math.log(2)
#      print("#{:g}\t{}".format(ce2, testfile))

      total_cross_entropy1 -= ce1
      total_cross_entropy2 -= ce2

      #Compare probabilities for prediction
      if map1 > map2:
          print(train_file1,"\t",testfile)
          class1_counter += 1
          prediction, filelength = evaluate(testfile, 'english')
          wrong_predictions += prediction
      else:
          print(train_file2, "\t", testfile)
          class2_counter += 1
          prediction, filelength = evaluate(testfile, 'spanish')
          wrong_predictions += prediction
    
      #files_length_accuracy[filelength].append(1-prediction)

    #Print Outputs for Class 1
    print(class1_counter,"files were more probably",train_file1,"({percent:.2f}%)".format(percent = 100*class1_counter/
                                                                                         (class1_counter + class2_counter)))
    #Print Outputs for Class 2
    print(class2_counter, "files were more probably", train_file2, "({percent:.2f}%)".format(percent = 100 * class2_counter/
                                                                                                (class1_counter + class2_counter)))
    print("#",wrong_predictions,"Error Rate: ", " ({percent:.2f}%)".format(percent = 100 * wrong_predictions/(class1_counter + class2_counter)))
    
    #filename = 'P3_{}_{}_{}_{}_data.txt'.format(smoother, basename(lexicon), basename(train_file1), basename(train_file2))
    #f = open(filename, "w")
    #for key, val in  files_length_accuracy.items():
    #    print("#File of length ", key," were ", 100*sum(val)/len(val), "% accurate.")
    #    f.write(str(key)+" "+str(100*sum(val)/len(val))+"\n")
    #f.close()


    # for p1,p2 in zip(ce1_list, ce2_list):
    #     if p1> p2:

    total_cross_entropy2 -= ce2
    
    total_cross_entropy = total_cross_entropy1 + total_cross_entropy2
#    print('#Overall cross-entropy:\t{0:.5f}'.format(total_cross_entropy1/sum([lm1.num_tokens(testfile) for testfile in argv])))
#    print('#Overall cross-entropy:\t{0:.5f}'.format(total_cross_entropy2/sum([lm2.num_tokens(testfile) for testfile in argv])))
    print('#Overall cross-entropy:\t{0:.5f}'.format(0.5*total_cross_entropy/sum([lm1.num_tokens(testfile) for testfile in argv])))

  else:
    sys.exit(-1)

if __name__ ==  "__main__":
  main()

