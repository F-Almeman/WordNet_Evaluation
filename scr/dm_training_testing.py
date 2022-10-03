import os
import sys
sys.path.append('lib')
from tqdm import tqdm
import pandas as pd
import argparse

from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from sklearn.model_selection import train_test_split
import numpy as np
import logging

def apply_prompt(df, prompt_opt):
  if prompt_opt == 'lemma:':
    df['EXAMPLE'] = df.LEMMA+': '+df.EXAMPLE
    
  elif prompt_opt == 'target':
    for idx in range(len(df)):
      lemma = df.LEMMA.iloc[idx]
      for w in df.EXAMPLE.iloc[idx].split():
        if w.strip().startswith(lemma.strip()):
          ex = df.EXAMPLE.iloc[idx].replace(w, f'<target> {w} </target>')
      df.EXAMPLE.iloc[idx] = ex
  return df


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-train','--train',help='train_data lemmas, definitions, and their examples',required=True)
  parser.add_argument('-val','--val',help='val_data lemmas, definitions, and their examples',required=True)
  parser.add_argument('-test','--test',help='test_data lemmas, definitions, and their examples',required=True)
  parser.add_argument('-o','--output_path',help='Path to save results',required=True)
  parser.add_argument('-p','--prompt_type',help='Prompt to use', choices = ['in','for','lemma:', 'target'], required=True)

  # Parse the argument
  args = parser.parse_args()

  # Read the csv file for training data
  df_train = pd.read_csv(args.train)
  df_train = df_train.dropna()
  
  # apply prompt
  df_train = apply_prompt(df_train, args.prompt_type)

  df_train = df_train['DEFINITION EXAMPLE'.split()]
  df_train.rename(columns={'DEFINITION':"target_text",'EXAMPLE':"input_text"}, inplace=True)
  
  # Read the csv file for validation data
  df_val = pd.read_csv(args.val)
  df_val = df_val.dropna()
  
  # apply prompt
  df_val = apply_prompt(df_val, args.prompt_type)

  df_val = df_val['DEFINITION EXAMPLE'.split()]
  df_val.rename(columns={'DEFINITION':"target_text",'EXAMPLE':"input_text"}, inplace=True)
  
  
  # Read the csv file for test data
  df_test = pd.read_csv(args.test)
  df_test = df_test.dropna()
  
  # apply prompt
  df_test = apply_prompt(df_test, args.prompt_type)

  df_test = df_test['DEFINITION EXAMPLE'.split()]
  df_test.rename(columns={'DEFINITION':"target_text",'EXAMPLE':"input_text"}, inplace=True)
  
 
  #############################################################

  logging.basicConfig(level=logging.INFO)
  transformers_logger = logging.getLogger("transformers")
  transformers_logger.setLevel(logging.WARNING)

  train_epochs = 20

  # Configure the model
  model_args = Seq2SeqArgs()
  model_args.num_train_epochs = train_epochs
  model_args.evaluate_generated_text = True
  model_args.evaluate_during_training = True
  model_args.evaluate_during_training_verbose = True
  model_args.overwrite_output_dir = True 
  model_args.use_cached_eval_features = True
  model_args.use_multiprocessing = False
  # disable saving to outputs
  model_args.save_eval_checkpoints = False
  model_args.save_steps = -1
  #model_args.train_batch_size = 2
  model_args.learning_rate = 4e-6
  
  if args.prompt_type == 'target':
    model_args.special_tokens = [r"<target>", r"<\target>"]

  model = Seq2SeqModel(
      encoder_decoder_type="bart",
      encoder_decoder_name="facebook/bart",
      args=model_args,
      use_cuda=True
  )

  # Configure Validation
  # Use train_test_split to split our data into train and validation sets for training

  # Num steps in epoch = num training samples / batch size
  steps_per_epoch = int(np.ceil(len(df_train) / float(model.args.train_batch_size)))
  print('Each epoch will have {:,} steps.'.format(steps_per_epoch))

  model.args.evaluate_during_training_steps = steps_per_epoch

  ##############################################################

  # Turn on early stopping.
  model.args.use_early_stopping = True

  # "The improvement over best_eval_loss necessary to count as a better checkpoint."
  model.args.early_stopping_delta = 0.0001

  # What metric to use in calculating score for evaluation set (plus whether a low
  # vs. high value is better for this metric).

  #model.args.early_stopping_metric = "mcc"
  #model.args.early_stopping_metric_minimize = False

  model.args.early_stopping_metric = "eval_loss"
  model.args.early_stopping_metric_minimize = True

  # "Terminate training after this many evaluations without an improvement in the
  #  evaluation metric greater then early_stopping_delta."
  model.args.early_stopping_patience = 7

  print('Training on {:,} samples...'.format(len(df_train)))
  
  out = model.train_model(df_train, eval_data=df_val)

  print("Done from training")
  #############################################################

  # Evaluate on test set

  model = Seq2SeqModel(
      encoder_decoder_type="bart",
      encoder_decoder_name="outputs/best_model",
      use_cuda=True,
  )
  

  preds = model.predict(df_test.input_text.tolist())
  
  print("Done from testing")


  ##########################################################

  results =  []

  for idx in range(len(df_test)):
    source = df_test.input_text.iloc[idx]
    gold = df_test.target_text.iloc[idx]
    pred = preds[idx]

   
    results.append({
              'source':source,
              'gold':gold,
              'pred':pred,
          })

  results_df = pd.DataFrame(results)

  results_df.to_csv (os.path.join(args.output_path, "dm_output.csv"), index = False, header=True)
