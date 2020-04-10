from tqdm import tqdm
import os
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import html
import tensorflow as tf
from matplotlib import pyplot as plt




"""
Reads movie_conversations.txt to get the right pairs
Sorts movie_lines.txt using the data from movie_conversations.txt
"""

CONVERSE_FILEPATH = "./data/movie_conversations.txt"
LINES_FILEPATH = "./data/movie_lines.txt"

def sort_data(converse_filepath, lines_filepath):
  seperator = " +++$+++ "
  """
  - movie_conversations.txt
  - the structure of the conversations
  - fields
    - characterID of the first character involved in the conversation
    - characterID of the second character involved in the conversation
    - movieID of the movie in which the conversation occurred
    - list of the utterances that make the conversation, in chronological 
      order: ['lineID1','lineID2',É,'lineIDN']

  - movie_lines.txt
  - contains the actual text of each utterance
  - fields:
    - lineID
    - characterID (who uttered this phrase)
    - movieID
    - character name
    - text of the utterance

  output data: data[mov1[line1[name, converse], line2[...]], mov2[...], ...]
  """
  with open(converse_filepath, "r") as cf:
    cf_lines = [l for l in cf.read().split("\n") if l != ""]
    cf_fields = [f.split(seperator) for f in cf_lines]

  with open(lines_filepath, "r") as lf:
    lf_lines = [l for l in lf.read().split("\n") if l != ""]
    lf_fields = [f.split(seperator) for f in lf_lines]
    lf_dict = dict()
    for f in lf_fields:
      lf_dict[f[0]] = f[3:5]

  data = list()
  movie_batch = list()
  converse_batch = list()
  line_id1 = cf_fields[0][0]
  line_id2 = cf_fields[0][1]
  movie_id = cf_fields[0][2]

  for f in tqdm(cf_fields):
    # print(f)
    if movie_id == f[2]:

      if line_id1 == f[0] and line_id2 == f[1]:
        for idx in eval(f[3]):
          converse_batch.append(lf_dict[idx])

      else:
        movie_batch.append(converse_batch)
        converse_batch = list()
        for idx in eval(f[3]):
          converse_batch.append(lf_dict[idx])

      line_id1 = f[0]
      line_id2 = f[1]

    else:
      data.append(movie_batch)
      movie_batch = list()
      movie_id = f[2]

  return data

def pull_twitter(twitter_filepath, shuffle=True):
  with open(twitter_filepath, "r", encoding="utf-8") as twt_f:
    lines = twt_f.read().split("\n")

  inputs, outputs = list(), list()
  for i, l in enumerate(tqdm(lines)):
    if i % 2 == 0:
      inputs.append(bytes(html.unescape(l).lower(), "utf-8"))
    else:
      outputs.append(bytes(html.unescape(l).lower(), "utf-8"))

  popped = 0
  for i, (ins, outs) in enumerate(zip(inputs, outputs)):
    if not ins or not outs:
      ins.pop(i)
      outs.pop(i)
      popped += 1

  print(f"Pairs popped: {popped}")
  if shuffle:
    print("\nShuffling...")
    inputs, outputs = shuffle_inputs_outputs(inputs, outputs)

  return inputs, outputs

def shuffle_inputs_outputs(inputs, outputs):
  inputs_outputs = list(zip(inputs, outputs))
  random.shuffle(inputs_outputs)
  inputs, outputs = zip(*inputs_outputs)
  return inputs, outputs

def create_tokenizers(inputs_outputs, inputs_outputs_savepaths, target_vocab_size):
  inputs, outputs = inputs_outputs
  inputs_savepath, outputs_savepath = inputs_outputs_savepaths

  # create tokens using tf subword tokenizer
  inputs_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    inputs, target_vocab_size=target_vocab_size)
  outputs_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    outputs, target_vocab_size=target_vocab_size)
  # save tokenizers to savepaths
  print("Saving tokenizers...")
  inputs_tokenizer.save_to_file(inputs_savepath)
  outputs_tokenizer.save_to_file(outputs_savepath)

  return inputs_tokenizer, outputs_tokenizer

def load_tokenizers(inputs_outputs_savepaths):
  print("Loading tokenizers...")
  inputs_savepath, outputs_savepath = inputs_outputs_savepaths
  inputs_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(inputs_savepath)
  outputs_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(outputs_savepath)

  return inputs_tokenizer, outputs_tokenizer

def encode(inputs_outputs, inputs_outputs_tokenizer):
  inputs, outputs = inputs_outputs
  inputs_tokenizer, outputs_tokenizer = inputs_outputs_tokenizer

  inputs = [inputs_tokenizer.vocab_size] + inputs_tokenizer.encode(
      inputs) + [inputs_tokenizer.vocab_size+1]

  outputs = [outputs_tokenizer.vocab_size] + outputs_tokenizer.encode(
      outputs) + [outputs_tokenizer.vocab_size+1]
  
  return inputs, outputs

def tf_encode(inputs_outputs, inputs_outputs_tokenizer):
  result_in, result_out = tf.py_function(encode, [inputs_outputs, inputs_outputs_tokenizer], [tf.int64, tf.int64])
  result_in.set_shape([None])
  result_out.set_shape([None])

  return result_in, result_out

def prepare_data(batch_size, inputs_outputs, inputs_outputs_tokenizer, max_length):
  print("Preparing data...")
  inputs, outputs = inputs_outputs
  if len(inputs) == len(outputs):
    batches_in = list()
    batches_out = list()
    curr_batch_in = list()
    curr_batch_out = list()
    skipped = 0
    for (ins, outs) in zip(inputs, outputs):
      ins, outs = encode([ins, outs], inputs_outputs_tokenizer)
      if len(ins) > max_length or len(outs) > max_length:
        skipped += 1
        continue
      else:
        ins = pad_sequences(sequences=[ins], maxlen=max_length,
          padding="post", truncating='post', value=0.0)[0]
        outs = pad_sequences(sequences=[outs], maxlen=max_length,
          padding="post", truncating='post', value=0.0)[0]
        curr_batch_in.append(ins)
        curr_batch_out.append(outs)

        if len(curr_batch_in) % batch_size == 0:
          batches_in.append(tf.convert_to_tensor(curr_batch_in, dtype=tf.int64))
          batches_out.append(tf.convert_to_tensor(curr_batch_out, dtype=tf.int64))
          curr_batch_in = list()
          curr_batch_out = list()

    if curr_batch_in:
      batches_in.append(tf.convert_to_tensor(curr_batch_in, dtype=tf.int64))
      batches_out.append(tf.convert_to_tensor(curr_batch_out, dtype=tf.int64))

    print(f"Total batches per epoch: {len(batches_in)}")
    print(f"Total pairs skipped: {skipped}")

    return batches_in, batches_out

  else:
    print("Given `inputs` length is not same as `outputs` length")

def plot_attention_weights(inputs_outputs_tokenizer, attention, sentence, result, layer):
  inputs_tokenizer, outputs_tokenizer = inputs_outputs_tokenizer
  fig = plt.figure(figsize=(16, 8))
  
  sentence = inputs_tokenizer.encode(sentence)
  
  attention = tf.squeeze(attention[layer], axis=0)
  
  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # plot the attention weights
    ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))
    
    ax.set_ylim(len(result)-1.5, -0.5)
        
    ax.set_xticklabels(
        ['<start>']+[inputs_tokenizer.decode([i]) for i in sentence]+['<end>'], 
        fontdict=fontdict, rotation=90)
    
    ax.set_yticklabels([outputs_tokenizer.decode([i]) for i in result 
                        if i < outputs_tokenizer.vocab_size], 
                       fontdict=fontdict)
    
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  # inputs, outputs = pull_twitter("./data/chat.txt")
  # print(f"Total inputs: {len(inputs)}, Total outputs: {len(outputs)}")
  # for i in range(20):
  #   print(f"""Input: {inputs[i].decode("utf-8")}""")
  #   print(f"""Output: {outputs[i].decode("utf-8")}""")

  srt_dt = sort_data(CONVERSE_FILEPATH, LINES_FILEPATH)
  print(srt_dt[0])

