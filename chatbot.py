import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from chatbot_utils import *
from chatbot_layers import *
import os
import time
import yaml

print(f"using tensorflow v{tf.__version__}")
print(f"using tensorflow.keras v{tf.keras.__version__}")




class TransformerChatbot(object):

  def __init__(self, config_path):

    with open(config_path) as cf:
      chatbot_config = yaml.load(cf, Loader=yaml.FullLoader)

    self.num_layers = chatbot_config["num_layers"]
    self.d_model = chatbot_config["d_model"]
    self.dff = chatbot_config["dff"]
    self.num_heads = chatbot_config["num_heads"]
    self.dropout_rate = chatbot_config["dropout_rate"]
    self.max_length = chatbot_config["max_length"]
    self.epochs = chatbot_config["epochs"]
    self.batch_size = chatbot_config["batch_size"]
    self.target_vocab_size = chatbot_config["target_vocab_size"]
    self.checkpoint = chatbot_config["checkpoint"]
    self.max_checkpoint = chatbot_config["max_checkpoint"]
    self.custom_checkpoint = chatbot_config["custom_checkpoint"]
    self.eval_limit = chatbot_config["eval_limit"]
    self.exit_phrase = chatbot_config["exit_phrase"]

    if chatbot_config["storage_path"] != None:
      self.storage_path = storage_path
    else:
      self.storage_path = "./"

    if not self.storage_path.endswith("/"):
      self.storage_path += "/"

    self.data_path = f"{self.storage_path}data"
    self.checkpoint_path = f"{self.storage_path}checkpoints/train"
    self.tokenizer_path = f"{self.storage_path}tokenizers"
    self.inputs_savepath = f"{self.tokenizer_path}/inputs_token"
    self.outputs_savepath = f"{self.tokenizer_path}/outputs_token"

    # create folders if they don't exists to prevent errors
    if not os.path.exists(f"{self.storage_path}checkpoints"):
      os.mkdir(f"{self.storage_path}checkpoints")
    if not os.path.exists(f"{self.storage_path}checkpoints/train"):
      os.mkdir(f"{self.storage_path}checkpoints/train")
    if not os.path.exists(f"{self.storage_path}tokenizers"):
      os.mkdir(f"{self.storage_path}tokenizers")
    if not os.path.exists(f"{self.storage_path}models"):
      os.mkdir(f"{self.storage_path}models")

    # preparing tokenizers and twitter data
    self.inputs, self.outputs = pull_twitter(f"{self.data_path}/chat.txt")
    try:
      self.inputs_tokenizer, self.outputs_tokenizer = load_tokenizers(
        inputs_outputs_savepaths=[self.inputs_savepath, self.outputs_savepath])
    except:
      print("No tokenizers has been created yet, creating new tokenizers...")
      self.inputs_tokenizer, self.outputs_tokenizer = create_tokenizers(
        inputs_outputs=[self.inputs, self.outputs],
        inputs_outputs_savepaths=[self.inputs_savepath, self.outputs_savepath],
        target_vocab_size=self.target_vocab_size)

    self.input_vocab_size = self.inputs_tokenizer.vocab_size + 2
    self.target_vocab_size = self.outputs_tokenizer.vocab_size + 2

    self.learning_rate = CustomSchedule(self.d_model)
    self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    self.transformer = Transformer(
      self.num_layers, self.d_model,
      self.num_heads, self.dff,
      self.input_vocab_size,
      self.target_vocab_size,
      pe_input=self.input_vocab_size,
      pe_target=self.target_vocab_size,
      rate=self.dropout_rate)

    self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                               optimizer=self.optimizer)
    self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=self.max_checkpoint)

    if self.custom_checkpoint:
      self.ckpt.restore(self.custom_checkpoint)
      print(f"Custom checkpoint restored: {self.custom_checkpoint}")
    # if a checkpoint exists, restore the latest checkpoint.
    elif self.ckpt_manager.latest_checkpoint:
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
      print (f"Latest checkpoint restored: {self.ckpt_manager.latest_checkpoint}")

    if chatbot_config["mode"] == "train":
      print("\nMODE: train\n===========\n")
      self.train_dataset = prepare_data(self.batch_size, [self.inputs, self.outputs],
        [self.inputs_tokenizer, self.outputs_tokenizer], self.max_length)

      self.train()
      # do some simple evaluation after training
      for (ins, outs) in zip(self.inputs, self.outputs):
        predicted_sentence, attention_weights, sentence, result = self.translate(ins)
        print(f"\nInput: {ins}")
        print(f"Predicted: {predicted_sentence}")
        print(f"Sample output: {outs}")
      plot_attention_weights([self.inputs_tokenizer, self.outputs_tokenizer],
        attention_weights, sentence, result, "decoder_layer4_block2")

    elif chatbot_config["mode"] == "eval":
      print("\nMODE: eval\n==========\n")
      self.inputs = self.inputs[:self.eval_limit]
      self.outputs = self.outputs[:self.eval_limit]

      for (ins, outs) in zip(self.inputs, self.outputs):
        predicted_sentence, attention_weights, sentence, result = self.translate(ins)
        print(f"\nInput: {ins}")
        print(f"Predicted: {predicted_sentence}")
        print(f"Sample output: {outs}")
      plot_attention_weights([self.inputs_tokenizer, self.outputs_tokenizer],
        attention_weights, sentence, result, "decoder_layer4_block2")

    elif chatbot_config["mode"] == "test":
      print("\nMODE: test\n==========\n")
      while True:
        usr_input = input("[USER]: ")
        if usr_input == self.exit_phrase:
          print("Exiting test mode...")
          break
        else:
          predicted_sentence, _, _, _ = self.translate(usr_input)
          print(f"[CHABOT]: {predicted_sentence}")

  def train(self):
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
      tar_inp = tar[:, :-1]
      tar_real = tar[:, 1:]
      
      enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
      
      with tf.GradientTape() as tape:
        predictions, _ = self.transformer(inp, tar_inp, 
                                     True, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

      gradients = tape.gradient(loss, self.transformer.trainable_variables)    
      self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
      
      self.train_loss(loss)
      self.train_accuracy(tar_real, predictions)

    for epoch in range(self.epochs):
      start = time.time()
      
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()
      
      # inp -> portuguese, tar -> english
      batches_in, batches_out = self.train_dataset
      for (batch, (inp, tar)) in enumerate(zip(batches_in, batches_out)):
        train_step(inp, tar)
        
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
              epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))
          
      if (epoch + 1) % self.checkpoint == 0:
        ckpt_save_path = self.ckpt_manager.save()
        print (f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}")

      print("Epoch {} Loss {:.4f} Accuracy {:.4f}".format(
        epoch + 1, self.train_loss.result(), self.train_accuracy.result()))
      print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  def evaluate(self, inp_sentence):
    start_token = [self.inputs_tokenizer.vocab_size]
    end_token = [self.inputs_tokenizer.vocab_size + 1]
    
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + self.inputs_tokenizer.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [self.outputs_tokenizer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)
      
    for i in range(self.max_length):
      enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
          encoder_input, output)
    
      # predictions.shape == (batch_size, seq_len, vocab_size)
      predictions, attention_weights = self.transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
      
      # select the last word from the seq_len dimension
      predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      
      # return the result if the predicted_id is equal to the end token
      if predicted_id == self.outputs_tokenizer.vocab_size+1:
        return tf.squeeze(output, axis=0), attention_weights
      
      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

  def translate(self, sentence):
    result, attention_weights = self.evaluate(sentence)
    
    predicted_sentence = self.outputs_tokenizer.decode([i for i in result 
                                              if i < self.outputs_tokenizer.vocab_size])

    return predicted_sentence, attention_weights, sentence, result




if __name__ == "__main__":
  CONFIG_PATH = "./chatbot_config.yml"
  transformer_chatbot = TransformerChatbot(CONFIG_PATH)