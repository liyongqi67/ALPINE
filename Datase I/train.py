# coding=UTF-8
import heapq
import random
import math
import tensorflow as tf
import numpy as np
import os
import io
import time
import datetime
import network
from tensorflow.contrib import learn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA 
# Parameters
# ==================================================
import sys

import pickle

#data file

time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

print(time)





#下面是模型的超参
lr=0.001

settings = network.Settings()
timestep = 300
train_batchSize=2048
val_batchsize=2048
item_dim=64
# Load data

#下面是视觉特征
visual="/home/share/liyongqi/kuaishou/process_data3/visual64_select.npy"
visual=np.load(visual)
print(visual.shape)
user_em=np.load("/home/share/liyongqi/kuaishou/process_data4/user_like.npy")
print(user_em.shape)
with open("/home/share/liyongqi/kuaishou/process_data4/dataset.pkl", 'rb') as f:
  train_interaction_data= pickle.load(f)
  test_interaction_data= pickle.load(f)
  pos_his_data= pickle.load(f)
  neg_his_data=pickle.load(f)
  pos_edge_data=pickle.load(f)
  neg_edge_data=pickle.load(f)

def getTrainBatch(batchSize,num):
        
        label=np.zeros([batchSize],dtype=np.int)
        pos_input=np.zeros([batchSize,timestep],dtype=np.int)
        pos_mask_input=np.zeros([batchSize],dtype=np.int)
        pos_edge_input=np.zeros([batchSize,timestep],dtype=np.int)
        neg_input=np.zeros([batchSize,timestep],dtype=np.int)
        neg_mask_input=np.zeros([batchSize],dtype=np.int)
        neg_edge_input=np.zeros([batchSize,timestep],dtype=np.int)
        item_input=np.zeros([batchSize],dtype=np.int)
        user_id_input=np.zeros([batchSize],dtype=np.int)
        for i in range(num*batchSize,(num+1)*batchSize):
             user_id=train_interaction_data[i][0][0]
             pos_len=train_interaction_data[i][1]
             neg_len=train_interaction_data[i][2]

             label[i%batchSize]=train_interaction_data[i][0][2]
             pos_input[i%batchSize]=pos_his_data[user_id][:pos_len]+[len(visual)-1]*(max(timestep,pos_len)-pos_len)
             pos_mask_input[i%batchSize]=pos_len
             pos_edge_input[i%batchSize]=pos_edge_data[user_id][:pos_len]+[0]*(max(timestep,pos_len)-pos_len)

             neg_input[i%batchSize]=neg_his_data[user_id][:neg_len]+[len(visual)-1]*(max(timestep,neg_len)-neg_len)
             neg_mask_input[i%batchSize]=neg_len

             neg_edge_input[i%batchSize]=neg_edge_data[user_id][:neg_len]+[0]*(max(timestep,neg_len)-neg_len)

             item_input[i%batchSize]=train_interaction_data[i][0][1]
             user_id_input[i%batchSize]=user_id




        return label,pos_input,pos_mask_input,pos_edge_input.reshape([-1,timestep,1]),neg_input,neg_mask_input,neg_edge_input.reshape([-1,timestep,1]),item_input,user_id_input
        
def getValBatch(batchSize,num):
        label=np.zeros([batchSize],dtype=np.int)
        pos_input=np.zeros([batchSize,timestep],dtype=np.int)
        pos_mask_input=np.zeros([batchSize],dtype=np.int)
        pos_edge_input=np.zeros([batchSize,timestep],dtype=np.int)
        neg_input=np.zeros([batchSize,timestep],dtype=np.int)
        neg_mask_input=np.zeros([batchSize],dtype=np.int)
        neg_edge_input=np.zeros([batchSize,timestep],dtype=np.int)
        item_input=np.zeros([batchSize],dtype=np.int)
        user_id_input=np.zeros([batchSize],dtype=np.int)
        for i in range(num*batchSize,(num+1)*batchSize):
             user_id=test_interaction_data[i][0][0]
             pos_len=test_interaction_data[i][1]
             neg_len=test_interaction_data[i][2]

             label[i%batchSize]=test_interaction_data[i][0][2]
             pos_input[i%batchSize]=pos_his_data[user_id][:pos_len]+[len(visual)-1]*(max(timestep,pos_len)-pos_len)
             pos_mask_input[i%batchSize]=pos_len
             pos_edge_input[i%batchSize]=pos_edge_data[user_id][:pos_len]+[0]*(max(timestep,pos_len)-pos_len)

             neg_input[i%batchSize]=neg_his_data[user_id][:neg_len]+[len(visual)-1]*(max(timestep,neg_len)-neg_len)
             neg_mask_input[i%batchSize]=neg_len
             neg_edge_input[i%batchSize]=neg_edge_data[user_id][:neg_len]+[0]*(max(timestep,neg_len)-neg_len)

             item_input[i%batchSize]=test_interaction_data[i][0][1]
             user_id_input[i%batchSize]=user_id




        return label,pos_input,pos_mask_input,pos_edge_input.reshape([-1,timestep,1]),neg_input,neg_mask_input,neg_edge_input.reshape([-1,timestep,1]),item_input,user_id_input

def eva(sess,model):
                valLoop=len(test_interaction_data)/val_batchsize
                valLoop=int(valLoop)
                result=[]
                result_ans=[]

                print("start eva")
                for valNum in tqdm(range(valLoop)):
                    label,pos_input,pos_mask_input,pos_edge_input,neg_input,neg_mask_input,neg_edge_input,item_input,user_id_input=getValBatch(val_batchsize,valNum)

                    loss,y_predVal= sess.run([model.loss,model.evaoutput],  feed_dict = {
                           model.pos_inputs: pos_input, 
                           model.pos_mask_inputs: pos_mask_input,  
                           model.pos_edge_inputs: pos_edge_input, 

                           model.neg_inputs: neg_input, 
                           model.neg_mask_inputs: neg_mask_input,  
                           model.neg_edge_inputs: neg_edge_input,

                           model.item_inputs:item_input,
                           model.y_inputs:label,
                           model.user_id_inputs:user_id_input,
                           model.tst: True, 
                           model.keep_prob: 1.0})
                    result.extend(y_predVal)
                    result_ans.extend(label)
                return roc_auc_score(result_ans,result)
# Placeholders for input, output and dropout
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True 
sess = tf.InteractiveSession(config=config)




model = network.Lstm(visual,user_em,settings)



train_op=tf.train.AdamOptimizer(0.001).minimize(model.loss,global_step=model.global_step)
update_op = tf.group(*model.update_emas)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saver.restore(sess,  "./model_save/model.ckpt")
auc_eva=eva(sess,model)
print(auc_eva)
with open(time+".txt", "a") as f:

    bestPerformance=0.0
    best_auceva=0
    auc_eva=eva(sess,model)
    for epoch in range(10000000):
        num=len(train_interaction_data)/train_batchSize
        num=int(num)
        

        for num in tqdm(range(num)):

            
            label,pos_input,pos_mask_input,pos_edge_input,neg_input,neg_mask_input,neg_edge_input,item_input,user_id_input=getTrainBatch(train_batchSize,num)

            _,_,loss,y_pred =sess.run([train_op,update_op,model.loss,model.evaoutput],
              feed_dict = {
                           model.pos_inputs: pos_input, 
                           model.pos_mask_inputs: pos_mask_input,  
                           model.pos_edge_inputs: pos_edge_input, 

                           model.neg_inputs: neg_input, 
                           model.neg_mask_inputs: neg_mask_input,  
                           model.neg_edge_inputs: neg_edge_input,

                           model.item_inputs:item_input,
                           model.y_inputs:label,
                           model.user_id_inputs:user_id_input,
                           model.tst: False, 
                           model.keep_prob: 1.0})
            if num%100==1:

              auc=roc_auc_score(label,y_pred)

              print('auc in train:',auc)
              print("epoch:",epoch,' num:',num,' loss:',loss)
              f.write("epoch:"+str(epoch)+' num:'+str(num)+' loss:'+str(loss)+' auc:'+str(auc)+"\n")
              f.flush()
            if num%2000==1 and not num==1:
                      auc_eva=eva(sess,model)
                      if(auc_eva>best_auceva):
                          save_path = saver.save(sess, "./model_save/model.ckpt")
                          best_auceva=auc_eva
                      print('epoch:',epoch,'auc on val:',auc_eva,'best_auceva',best_auceva)
                      f.write('epoch:'+str(epoch)+' auc on val:'+str(auc_eva)+' best_auceva:'+str(best_auceva)+"\n" )
                      f.flush()


        auc_eva=eva(sess,model)
        if(auc_eva>best_auceva):
            save_path = saver.save(sess, "./model_save/model.ckpt")
            best_auceva=auc_eva
        print('epoch:',epoch,'auc on val:',auc_eva,'best_auceva',best_auceva)
        f.write('epoch:'+str(epoch)+' auc on val:'+str(auc_eva)+' best_auceva:'+str(best_auceva)+"\n" )
        f.flush()

     
sess.close()

