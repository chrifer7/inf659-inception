import tensorflow as tf
import sys
import math
import os
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from getvector import getvector
from tensorflow.python.platform import gfile
from progress.bar import Bar

if len(sys.argv) <= 1:
    print('enter option: "train" or "test <image file name>"')
    exit()

data_inputs = []
data_labels = []

# Checking if the 2048-dimensional vector representations of the training images are already available
if os.path.isfile('./data_inputs.txt') and os.path.isfile('./data_labels.txt'):
    data_inputs = np.loadtxt('data_inputs.txt')
    data_labels = np.loadtxt('data_labels.txt')

else:     
    #tf.reset_default_graph()
    # add in your images here if you want to train the model on your own images
    data_dir = 'data/DM Images/train'
    
    #Cuenta la cantidad de directorios/clases
    n_classes = 0
    for o in os.listdir(data_dir):
        if not o.startswith('.'):
            n_classes = n_classes + 1
    
    i_class = 0
    for o in os.listdir(data_dir):
        if not o.startswith('.'):     
            class_name = os.path.join(o)
            print("Clase: ", class_name, "Indice: ",  i_class)

            file_list = []
            file_glob = os.path.join(data_dir, class_name, '*.jpg');
            print('File glob: ', file_glob)
            file_list.extend(gfile.Glob(file_glob))
            print('Samples: ', len(file_list))

            #file_list = file_list[0:2]
            #bar = Bar('Inception-V3 is processing images:', max=300)
            bar = Bar('Inception-V3 is processing images:', max=len(file_list))
            
            one_hot_row = np.zeros(n_classes)             
            one_hot_row[i_class] = 1
            print('one hot: ', one_hot_row)
            for file_name in file_list:
                print('\nProcessing: ', file_name)
                #Extrae caracteristicas
                data_inputs.append(getvector(file_name))
                data_labels.append(one_hot_row)
                
                bar.next()
            i_class = i_class + 1
    bar.finish()
    
    print('Datos cargados de un archivo.')

    np.savetxt('data_inputs.txt', data_inputs)
    np.savetxt('data_labels.txt', data_labels)

# Splitting into train, val, and test
train_inputs, valtest_inputs, train_labels, valtest_labels = train_test_split(data_inputs, data_labels, test_size=0.3,
                                                                              random_state=42)
val_inputs, test_inputs, val_labels, test_labels = train_test_split(valtest_inputs, valtest_labels, test_size=0.4,
                                                                    random_state=43)

# Setting hyperparameters
learning_rate = 0.01
batch_size = 64
epochs = 10
log_batch_step = 50

# useful info
n_features = np.size(train_inputs, 1)
n_labels = np.size(train_labels, 1)

# Placeholders for input features and labels
inputs = tf.placeholder(tf.float32, (None, n_features))
labels = tf.placeholder(tf.float32, (None, n_labels))

# Setting up weights and bias
weights = tf.Variable(tf.truncated_normal((n_features, n_labels), stddev=0.1), name='weights')
bias = tf.Variable(tf.zeros(n_labels), name='bias')
tf.add_to_collection('vars', weights)
tf.add_to_collection('vars', bias)

# Setting up operation in fully connected layer
logits = tf.add(tf.matmul(inputs, weights), bias)
prediction = tf.nn.softmax(logits)
tf.add_to_collection('pred', prediction)

# Defining loss of network
difference = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss = tf.reduce_sum(difference)

# Setting optimiser
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Define accuracy
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

saver = tf.train.Saver((weights, bias))

if sys.argv[1] == 'train':
    # Run tensorflow session
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Running the training in batches
        batch_count = int(math.ceil(len(train_inputs) / batch_size))

        for epoch_i in range(epochs):
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')
            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_inputs = train_inputs[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]
                # Run optimizer
                _ = sess.run(optimizer, feed_dict={inputs: batch_inputs, labels: batch_labels})

            # Check accuracy against validation data
            val_accuracy, val_loss = sess.run([accuracy, loss], feed_dict={inputs: val_inputs, labels: val_labels})
            print("After epoch {}, Loss: {}, Accuracy: {}".format(epoch_i + 1, val_loss, val_accuracy))

        g = tf.get_default_graph()
        saver.save(sess, 'testsave')


elif sys.argv[1] == 'test':
    try:
        file_name = sys.argv[2]
    except IndexError:
        print('please enter image file path.........')
        exit()
    image_input = getvector(file_name).reshape((1, 2048))
    if 'cat' in file_name:
        image_label = [[1, 0]]
    else:
        image_label = [[0, 1]]

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('testsave.meta') #verificar despues
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))

        prediction = sess.run(prediction, feed_dict={inputs: image_input})

        print('It\'s a cat: {}, It\'s a dog: {}'.format(prediction[0][0], prediction[0][1]))

else:
    print('type either train or test!')
    exit()
