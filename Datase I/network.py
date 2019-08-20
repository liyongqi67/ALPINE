# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell


class Settings(object):
    def __init__(self):
        self.model_name = 'lstm'
        self.timestep = 300

        self.item_dim = 64


        self.hidden_size =  64
        self.n_layer = 1

        self.batch_size=2048
        self.dnn_size=128

class Lstm(object):

    def __init__(self, W_embedding,user_embedding,settings):
        self.model_name = settings.model_name
        self.timestep = settings.timestep
        self.item_dim = settings.item_dim

        self.hidden_size = settings.hidden_size
        self.n_layer = settings.n_layer

        self.dnn_size=settings.dnn_size


        self.global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        # placeholders
        self.tst = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = settings.batch_size
        self.visual_embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.user_embedding = tf.get_variable(name='user_embedding', shape=user_embedding.shape,
                                             initializer=tf.constant_initializer(user_embedding), trainable=True)
        with tf.name_scope('Inputs'):
            self.pos_inputs = tf.placeholder(tf.int32, [None, self.timestep], name='pos_inputs')   #postive history
            self.pos=tf.nn.embedding_lookup(self.visual_embedding, self.pos_inputs)  
            self.pos_mask_inputs=tf.placeholder(tf.int32, [None,], name='pos_mask_inputs')         #postive history sequence mask
            self.pos_edge_inputs=tf.placeholder(tf.int32, [None,self.timestep,1], name='pos_edge_inputs')   #postive history edge

            self.neg_inputs = tf.placeholder(tf.int32, [None, self.timestep], name='neg_inputs')   #negtive history
            self.neg=tf.nn.embedding_lookup(self.visual_embedding, self.neg_inputs) 
            self.neg_mask_inputs=tf.placeholder(tf.int32, [None,], name='neg_mask_inputs')         
            self.neg_edge_inputs=tf.placeholder(tf.int32, [None,self.timestep,1], name='neg_edge_inputs')


            self.item_inputs = tf.placeholder(tf.int32, [None], name='item_inputs')     # video id
            self.item=tf.nn.embedding_lookup(self.visual_embedding, self.item_inputs)

            self.y_inputs = tf.placeholder(tf.float32, [None], name='y_input')          #label

            self.user_id_inputs= tf.placeholder(tf.int32, [None], name='user_id_inputs')
            self.user_feature=tf.nn.embedding_lookup(self.user_embedding, self.user_id_inputs)


        with tf.variable_scope('pos_encoder'):
            self.pos_em=self.temporal_graph_based_LSTM_layer(self.pos,self.pos_edge_inputs,self.pos_mask_inputs)

        with tf.variable_scope('neg_encoder'):
            self.neg_em=self.temporal_graph_based_LSTM_layer(self.neg,self.neg_edge_inputs,self.neg_mask_inputs)
       
        with tf.variable_scope('pos_dnn'):
            w1=self.weight_variable([self.item_dim*2 , self.dnn_size], name='w1')
            b1=self.bias_variable([self.dnn_size],name='b1')

            output=tf.matmul(tf.concat([self.pos_em,self.item],axis=1),w1)+b1

            output=tf.nn.dropout(output,self.keep_prob)
            output=tf.nn.relu(output)

            w2=self.weight_variable([self.dnn_size, 1], name='w2')
            b2=self.bias_variable([1],name='b2')

            output=tf.matmul(output,w2)+b2


            self.pos_output=tf.reshape(output,[-1])


        with tf.variable_scope('neg_dnn'):
            w1=self.weight_variable([self.item_dim*2 , self.dnn_size], name='w1')
            b1=self.bias_variable([self.dnn_size],name='b1')

            output=tf.matmul(tf.concat([self.neg_em,self.item],axis=1),w1)+b1

            output=tf.nn.dropout(output,self.keep_prob)
            output=tf.nn.relu(output)

            w2=self.weight_variable([self.dnn_size, 1], name='w2')
            b2=self.bias_variable([1],name='b2')

            output=tf.matmul(output,w2)+b2


            self.neg_output=tf.reshape(output,[-1])

        with tf.variable_scope('multi_level_interest_layer'):
            w1=self.weight_variable([self.item_dim*2 , self.dnn_size], name='w1')
            b1=self.bias_variable([self.dnn_size],name='b1')

            output=tf.matmul(tf.concat([self.item,self.user_feature],axis=1),w1)+b1

            output=tf.nn.dropout(output,self.keep_prob)
            output=tf.nn.relu(output)

            w2=self.weight_variable([self.dnn_size, 1], name='w2')
            b2=self.bias_variable([1],name='b2')

            output=tf.matmul(output,w2)+b2

            self.enhance_output=tf.reshape(output,[-1])

        #
        self.output=0.58*self.pos_output+0.18*self.neg_output+0.24*self.enhance_output
        self.evaoutput=tf.nn.sigmoid(self.output)

        with tf.name_scope('loss'):
            l2_norm = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.y_inputs))+l2_norm * 0.00005




    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)


    def vanilla_attention(self,queries, keys, keys_length):

        queries = tf.expand_dims(queries, 1) # [B, 1, H]
  # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) # [B, 1, T]

  # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
        key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return tf.reshape(outputs,[-1,self.item_dim])


    def lstm_cell_forward(self,xt, h_prev, c_prev,h_edge,c_edge):

            with tf.variable_scope('rnn_cell', reuse=True):
###################################################################

                #输入门

                Wix = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wix')
                Wih = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wih')
                Bi = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bi')

                input_gate=tf.nn.sigmoid(tf.matmul(xt,Wix)+tf.matmul(h_prev,Wih)+Bi)


                #遗忘门
                Wfx = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wfx')
                Wfh = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wfh')
                Bf = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bf')

                forget_gate=tf.nn.sigmoid(tf.matmul(xt,Wfx)+tf.matmul(h_prev,Wfh)+Bf)

                #输出门
                Wox = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wox')
                Woh = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Woh')
                Bo = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bo')

                output_gate=tf.nn.sigmoid(tf.matmul(xt,Wox)+tf.matmul(h_prev,Woh)+Bo)


                #当前记忆
                Wmx = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wmx')
                Wmh = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wmh')
                Bm = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bm')
                
                memory_gate=tf.nn.tanh(tf.matmul(xt,Wmx)+tf.matmul(h_prev,Wmh)+Bm)
                

                #输出记忆
                c=tf.multiply(input_gate,memory_gate)+tf.multiply(forget_gate,c_prev)
##############################################################################################################################
                #输入门

                Wix2 = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wix2')
                Wih2 = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wih2')
                Bi2 = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bi2')

                input_gate2=tf.nn.sigmoid(tf.matmul(xt,Wix2)+tf.matmul(h_edge,Wih2)+Bi2)


                #遗忘门
                Wfx2 = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wfx2')
                Wfh2 = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wfh2')
                Bf2 = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bf2')

                forget_gate2=tf.nn.sigmoid(tf.matmul(xt,Wfx2)+tf.matmul(h_edge,Wfh2)+Bf2)

                #输出门
                Wox2 = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wox2')
                Woh2 = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Woh2')
                Bo2 = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bo2')

                output_gate2=tf.nn.sigmoid(tf.matmul(xt,Wox2)+tf.matmul(h_edge,Woh2)+Bo2)


                #当前记忆
                Wmx2 = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wmx2')
                Wmh2 = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wmh2')
                Bm2 = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bm2')

                memory_gate2=tf.nn.tanh(tf.matmul(xt,Wmx2)+tf.matmul(h_edge,Wmh2)+Bm2)
                
                c2=tf.multiply(input_gate2,memory_gate2)+tf.multiply(forget_gate2,c_edge)

################################################################################################################################

                #隐层输出
                h=tf.multiply(output_gate,tf.nn.tanh(c))+tf.multiply(output_gate2,tf.nn.tanh(c2))

                return c,c2,h

    def temporal_graph_based_LSTM_layer(self,sequence_input,edge_input,mask_input):

            with tf.variable_scope('rnn_cell'):
                    #输入门

                    Wix = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wix')
                    Wih = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wih')
                    Bi = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bi')




                    #遗忘门
                    Wfx = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wfx')
                    Wfh = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wfh')
                    Bf = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bf')



                    #输出门
                    Wox = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wox')
                    Woh = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Woh')
                    Bo = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bo')




                    #当前记忆
                    Wmx = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wmx')
                    Wmh = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wmh')
                    Bm = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bm')

                    #输入门

                    Wix2 = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wix2')
                    Wih2 = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wih2')
                    Bi2 = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bi2')




                    #遗忘门
                    Wfx2 = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wfx2')
                    Wfh2 = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wfh2')
                    Bf2 = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bf2')



                    #输出门
                    Wox2 = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wox2')
                    Woh2 = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Woh2')
                    Bo2 = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bo2')




                    #当前记忆
                    Wmx2 = tf.get_variable(shape=[self.item_dim, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wmx2')
                    Wmh2 = tf.get_variable(shape=[self.hidden_size, self.hidden_size],initializer=tf.orthogonal_initializer(),name='Wmh2')
                    Bm2 = tf.get_variable(shape=[self.hidden_size],initializer=tf.constant_initializer(1.0),name='Bm2')

            sequence_input=tf.unstack(sequence_input, axis=1)
            edge_input=tf.unstack(edge_input,axis=1)

            h0=tf.Variable(tf.constant(0.0, shape=[self.batch_size,self.hidden_size]))
            c0=tf.Variable(tf.constant(0.0, shape=[self.batch_size,self.hidden_size]))

            c,c2,h= self.lstm_cell_forward(sequence_input[0], h0,c0,h0,c0)

            rnn_outputs = h
            rnn_outputs = tf.reshape(rnn_outputs,[-1,1,self.hidden_size])
            batch_index=tf.reshape(tf.range(self.batch_size),[-1,1]) #  B 1
            for i in range(1,len(sequence_input)):
                index=tf.concat([batch_index,tf.reshape(edge_input[i],[-1,1])],1)     # B 2
                rnn_input=tf.gather_nd(rnn_outputs,index)

                c,c2,h= self.lstm_cell_forward(sequence_input[i], h,c,rnn_input,c2)

                rnn_outputs = tf.concat([rnn_outputs,tf.reshape(h,[-1,1,self.hidden_size])],1)
            rnn_outputs=tf.nn.dropout(rnn_outputs,self.keep_prob)

            rnn_outputs=self.vanilla_attention(self.item,rnn_outputs,mask_input)

            return rnn_outputs