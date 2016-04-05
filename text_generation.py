import tensorflow as tf
import numpy as np
import random
from tensorflow.models.rnn import rnn, rnn_cell

class ReadTextFile(object):
    def __init__(self, filename, batch_size=None):
        self.file = open(filename, 'r')
        self.current_batch_num = 0
        self.batch_size = batch_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def batch(self):
        data = None
        while True:
            if self.batch_size:
                data = self.file.read(self.batch_size)
            else:
                data = self.file.read()
            yield data

def build_categories(filename):
    with ReadTextFile("scripts.txt") as reader:
        data = reader.batch().next()
        data_size = len(data)
        classes = list(set(data))
        # print classes
        # print len(classes)
        return data_size, classes

# build_categories("scripts.txt")

# TODO Use word2vec after getting initial version working
class LSTM(object):
    def __init__(self, num_nodes=128, data_file="scripts.txt", batch_size=4096):
        self.num_nodes = num_nodes
        self.data_file = data_file
        self.batch_size = batch_size

    def one_hot(self, indices, num_classes):
        output = []
        for i in indices:
            vect = [0.0]*num_classes
            vect[i] = 1.0
            output.append(vect)
        return np.array(output)

    def train(self, num_epochs):
        # Build classes from data
        try:
            self.data_size, self.classes = build_categories(self.data_file)
        except IOError:
            print "Could not read file {}" %self.data_file
        self.num_classes = len(self.classes)
        self.num_epochs = num_epochs
        forget_bias = 1.0
        learning_rate = 0.2
        next_char_sample_size = 5
        keep_prob = 0.5


        with tf.Graph().as_default(), tf.Session() as session:
            # Placeholder for input data feed
            x = tf.placeholder(tf.float32, shape=[None, self.num_classes])
            # Weights and biases for first hidden layer
            W_hidden = tf.Variable(tf.truncated_normal(shape=[self.num_classes, self.num_nodes]))
            b_hidden = tf.Variable(tf.truncated_normal(shape=[self.num_nodes]))
            # Placeholder for label data feed
            y = tf.placeholder(tf.float32, shape=[None, self.num_classes])

            # LSTM cell with dropout and initial state = 0
            lstm = rnn_cell.BasicLSTMCell(num_units=self.num_nodes, forget_bias=forget_bias)
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            state = tf.zeros(shape=[self.batch_size, lstm.state_size])

            # Output of first fully connected layer
            output = tf.matmul(x, W_hidden) + b_hidden
            # Output of LSTM layer
            output, state = lstm(output, state)

            # Weights and biases for second hidden layer
            W_output = tf.Variable(tf.truncated_normal(shape=[self.num_nodes, self.num_classes]))
            b_output = tf.Variable(tf.truncated_normal(shape=[self.num_classes]))

            # Output of second fully connected layer
            logits = tf.matmul(output, W_output) + b_output
            # Calculate class probabilities
            probs = tf.nn.softmax(logits)
            # Calculate loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

            # Optimization step for training the model
            # training_step = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
            training_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9).minimize(loss)
            # Initialize variables
            session.run(tf.initialize_all_variables())

            # Train the model
            for epoch in range(self.num_epochs):
                # print 'Epoch number:',epoch
                # Process data in batches
                with ReadTextFile("stories.txt", self.batch_size+1) as reader:
                    for step in range(int(self.data_size/(self.batch_size+1))):
                        # Reuse variables for training
                        if step > 0:
                            tf.get_variable_scope().reuse_variables()
                        # Read in next batch of data
                        data = reader.batch().next()

                        # Build training set and one-hot encode it
                        training_data_indices = [self.classes.index(char) for char in list(data[:-1])]
                        training_data = self.one_hot(training_data_indices, self.num_classes)

                        # Build testing set and one-hot encode it
                        training_labels_indices = [self.classes.index(char) for char in list(data[1:])]
                        training_labels = self.one_hot(training_labels_indices, self.num_classes)

                        # Run session to train model
                        session.run(training_step, feed_dict={x: training_data, y: training_labels})
                    # Display loss
                    print "Loss for epoch {} = {}".format(epoch+1, loss.eval(feed_dict={x: training_data, y: training_labels}))

                # TODO Look into using Savers for checkpointing the model
                # Sample text at regular intervals
                if epoch % 2 == 0:
                    # Number of chars to generate
                    num_steps = 1000
                    # Start with random seed char
                    seed = random.choice(self.classes)
                    out = []
                    for step in range(num_steps):
                        # print seed
                        out.append(seed)
                        # Hack to pass Y into feed dict - Possible issue?
                        evaled = probs.eval(feed_dict={x: self.one_hot([self.classes.index(seed)]*self.batch_size, self.num_classes),
                                                       y: self.one_hot([2]*self.batch_size, self.num_classes)})
                        prob = random.choice(evaled[0].argsort()[-next_char_sample_size:][::-1])
                        # print prob
                        seed = self.classes[prob]

                    # Write generated text to outputs file with expt parameters
                    contents = "".join(out)
                    with open("outputs.txt", "a") as output_file:
                        output_file.write("Number of epochs: {}".format(epoch)+"\n")
                        output_file.write("Number of lstm nodes: {}".format(self.num_nodes)+"\n")
                        output_file.write("Keep prob: {}".format(keep_prob)+"\n")
                        output_file.write("Optimizer: {}".format(training_step.name)+"\n")
                        output_file.write("Number of classes: {}".format(self.num_classes)+"\n")
                        output_file.write("Batch size: {}".format(self.batch_size)+"\n")
                        output_file.write("Sample size for probabilistic generation: {}".format(next_char_sample_size)+"\n")
                        output_file.write("Learning rate: {}".format(learning_rate)+"\n\n")
                        for chunk in range(0, len(out)-100, 100):
                            output_file.write(contents[chunk:chunk+100]+"\n")
                            # print contents[chunk:chunk+100]
                        output_file.write("\n")


def main():
    for num_nodes in [128]:
        model = LSTM(num_nodes)
        for num_epochs in [1000]:
            model.train(num_epochs)

if __name__ == '__main__':
    main()
