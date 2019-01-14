import os
import collections
import tokenization

import numpy as np
import tensorflow as tf 

from run_classifier import ColaProcessor
from run_classifier import MnliProcessor
from run_classifier import MrpcProcessor
from run_classifier import XnliProcessor
from run_classifier import convert_single_example

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "model_path", None,
    "The input model path. Should contain the .pb files "
    "for the task.")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")



def load_model(model_path):
    """Load the frozen pb file then build the graph
    Args:
        model_path: path of the frozen pb file
    return:
        sess: the session of this bert model
    """
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    tf.saved_model.loader.load(
                sess, [tf.saved_model.tag_constants.SERVING], model_path)
    return sess


def bert_prediction(vocab_file, do_lower_case, task_name, sess, data_dir, max_seq_length):
    processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
    }
    
    tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    predict_examples = processor.get_test_examples(data_dir)
    frozen_input_ids = sess.graph.get_tensor_by_name("input_ids:0")
    frozen_input_mask = sess.graph.get_tensor_by_name("input_mask:0")
    frozen_segment_ids = sess.graph.get_tensor_by_name("segment_ids:0")
    predict_info = sess.graph.get_tensor_by_name("predict_probabilities:0")
    for (ex_index, example) in enumerate(predict_examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(predict_examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                        max_seq_length, tokenizer)
        probabilities = sess.run(predict_info, feed_dict={frozen_input_ids: np.expand_dims(feature.input_ids, 0), frozen_input_mask: np.expand_dims(feature.input_mask, 0),frozen_segment_ids:np.expand_dims(feature.segment_ids, 0)})
        print("[INFO]the class_probability is: ", probabilities)

def main():
    sess = load_model(FLAGS.model_path)
    bert_prediction(FLAGS.vocab_file, FLAGS.do_lower_case, FLAGS.task_name, sess, FLAGS.data_dir, FLAGS.max_seq_length)

if  __name__ =="__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()

