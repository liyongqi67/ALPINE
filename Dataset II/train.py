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

#下面是视觉特征
train_visual="/home/share/liyongqi/kuaishou/dataset2/visual_feature.npy"
train_visual=np.load(train_visual)
print(train_visual.shape)

with open("/home/share/liyongqi/kuaishou/dataset2/train_dataset.pkl", 'rb') as f:
  train_interaction_data= pickle.load(f)
  pos_his_data= pickle.load(f)
  neg_his_data=pickle.load(f)
  pos_edge_data=pickle.load(f)
  neg_edge_data=pickle.load(f)
with open("/home/share/liyongqi/MicroVideo-1.7M/test_data.csv", 'r') as reader:
    reader.readline()
    val_interaction_data = []
    for s in tqdm(reader.readlines()):
        line=s.strip('\n').split(',')
        line = list(map(int, line))
        line[1]=line[1]+984984
        val_interaction_data.append(line)
def getTrainBatch(batchSize,num):
        
        label=np.zeros([batchSize],dtype=np.int)
        pos_input=np.zeros([batchSize,timestep],dtype=np.int)
        pos_mask_input=np.zeros([batchSize],dtype=np.int)
        pos_edge_input=np.zeros([batchSize,timestep],dtype=np.int)
        neg_input=np.zeros([batchSize,timestep],dtype=np.int)
        neg_mask_input=np.zeros([batchSize],dtype=np.int)
        neg_edge_input=np.zeros([batchSize,timestep],dtype=np.int)
        item_input=np.zeros([batchSize],dtype=np.int)

        for i in range(num*batchSize,(num+1)*batchSize):
             user_id=train_interaction_data[i][0][0]
             pos_len=train_interaction_data[i][1]
             neg_len=train_interaction_data[i][2]

             label[i%batchSize]=train_interaction_data[i][0][3]
             pos_input[i%batchSize]=list(map(lambda n:n+1,pos_his_data[user_id][:pos_len]))+[len(train_visual)-1]*(max(timestep,pos_len)-pos_len)
             pos_mask_input[i%batchSize]=pos_len
             pos_edge_input[i%batchSize]=pos_edge_data[user_id][:pos_len]+[0]*(max(timestep,pos_len)-pos_len)

             neg_input[i%batchSize]=list(map(lambda n:n+1,neg_his_data[user_id][:neg_len]))+[len(train_visual)-1]*(max(timestep,neg_len)-neg_len)
             neg_mask_input[i%batchSize]=neg_len

             neg_edge_input[i%batchSize]=neg_edge_data[user_id][:neg_len]+[0]*(max(timestep,neg_len)-neg_len)

             item_input[i%batchSize]=train_interaction_data[i][0][1]+1




        return label,pos_input,pos_mask_input,pos_edge_input.reshape([-1,timestep,1]),neg_input,neg_mask_input,neg_edge_input.reshape([-1,timestep,1]),item_input
        
def getValBatch(batchSize,num):
        label=np.zeros([batchSize],dtype=np.int)
        pos_input=np.zeros([batchSize,timestep],dtype=np.int)
        pos_mask_input=np.zeros([batchSize],dtype=np.int)
        pos_edge_input=np.zeros([batchSize,timestep],dtype=np.int)
        neg_input=np.zeros([batchSize,timestep],dtype=np.int)
        neg_mask_input=np.zeros([batchSize],dtype=np.int)
        neg_edge_input=np.zeros([batchSize,timestep],dtype=np.int)
        item_input=np.zeros([batchSize],dtype=np.int)

        for i in range(num*batchSize,(num+1)*batchSize):
             user_id=val_interaction_data[i][0]
             pos_len=len(pos_his_data[user_id])
             neg_len=len(neg_his_data[user_id])

             label[i%batchSize]=val_interaction_data[i][3]
             pos_input[i%batchSize]=list(map(lambda n:n+1,pos_his_data[user_id][:pos_len]))+[len(train_visual)-1]*(max(timestep,pos_len)-pos_len)
             pos_mask_input[i%batchSize]=pos_len
             pos_edge_input[i%batchSize]=pos_edge_data[user_id][:pos_len]+[0]*(max(timestep,pos_len)-pos_len)

             neg_input[i%batchSize]=list(map(lambda n:n+1,neg_his_data[user_id][:neg_len]))+[len(train_visual)-1]*(max(timestep,neg_len)-neg_len)
             neg_mask_input[i%batchSize]=neg_len
             neg_edge_input[i%batchSize]=neg_edge_data[user_id][:neg_len]+[0]*(max(timestep,neg_len)-neg_len)

             item_input[i%batchSize]=val_interaction_data[i][1]



             
        return label,pos_input,pos_mask_input,pos_edge_input.reshape([-1,timestep,1]),neg_input,neg_mask_input,neg_edge_input.reshape([-1,timestep,1]),item_input

def pos_eva(sess,model):
                valLoop=len(val_interaction_data)/val_batchsize
                valLoop=int(valLoop)
                result=[]
                result_ans=[]
                print("start eva")
                for valNum in tqdm(range(valLoop)):
                    label,pos_input,pos_mask_input,pos_edge_input,neg_input,neg_mask_input,neg_edge_input,item_input=getValBatch(val_batchsize,valNum)

                    loss,y_predVal= sess.run([model.pos_loss,model.pos_evaoutput],  feed_dict = {
                           model.pos_inputs: pos_input, 
                           model.pos_mask_inputs: pos_mask_input,  
                           model.pos_edge_inputs: pos_edge_input, 

                           model.neg_inputs: neg_input, 
                           model.neg_mask_inputs: neg_mask_input,  
                           model.neg_edge_inputs: neg_edge_input,

                           model.item_inputs:item_input,
                           model.y_inputs:label,
                           model.tst: True, 
                           model.keep_prob: 1.0})
                    result.extend(y_predVal)
                    result_ans.extend(label)
                return roc_auc_score(result_ans,result),np.array(result),np.array(result_ans)
def neg_eva(sess,model):
                valLoop=len(val_interaction_data)/val_batchsize
                valLoop=int(valLoop)
                result=[]
                result_ans=[]
                print("start eva")
                for valNum in tqdm(range(valLoop)):
                    label,pos_input,pos_mask_input,pos_edge_input,neg_input,neg_mask_input,neg_edge_input,item_input=getValBatch(val_batchsize,valNum)

                    loss,y_predVal= sess.run([model.neg_loss,model.neg_evaoutput],  feed_dict = {
                           model.pos_inputs: pos_input, 
                           model.pos_mask_inputs: pos_mask_input,  
                           model.pos_edge_inputs: pos_edge_input, 

                           model.neg_inputs: neg_input, 
                           model.neg_mask_inputs: neg_mask_input,  
                           model.neg_edge_inputs: neg_edge_input,

                           model.item_inputs:item_input,
                           model.y_inputs:label,
                           model.tst: True, 
                           model.keep_prob: 1.0})
                    result.extend(y_predVal)
                    result_ans.extend(label)

                return roc_auc_score(result_ans,result),np.array(result),np.array(result_ans)
def joint_eva(sess,model):
                valLoop=len(val_interaction_data)/val_batchsize
                valLoop=int(valLoop)
                result=[]
                result_ans=[]

                print("start eva")
                for valNum in tqdm(range(valLoop)):
                    label,pos_input,pos_mask_input,pos_edge_input,neg_input,neg_mask_input,neg_edge_input,item_input=getValBatch(val_batchsize,valNum)

                    loss,y_predVal= sess.run([model.joint_loss,model.joint_evaoutput],  feed_dict = {
                           model.pos_inputs: pos_input, 
                           model.pos_mask_inputs: pos_mask_input,  
                           model.pos_edge_inputs: pos_edge_input, 

                           model.neg_inputs: neg_input, 
                           model.neg_mask_inputs: neg_mask_input,  
                           model.neg_edge_inputs: neg_edge_input,

                           model.item_inputs:item_input,
                           model.y_inputs:label,
                           model.tst: True, 
                           model.keep_prob: 1.0})
                    result.extend(y_predVal)
                    result_ans.extend(label)

                return roc_auc_score(result_ans,result),np.array(result),np.array(result_ans)
# Placeholders for input, output and dropout
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True 
sess = tf.InteractiveSession(config=config)




model = network.Lstm(train_visual,settings)



pos_train_op=tf.train.AdamOptimizer(0.001).minimize(model.pos_loss)
neg_train_op=tf.train.AdamOptimizer(0.001).minimize(model.neg_loss)
joint_train_op=tf.train.AdamOptimizer(0.001).minimize(model.joint_loss)


sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

#saver.restore(sess,  "./model_save/model.ckpt")

with open(time+".txt", "a") as f:

    bestPerformance=0.0
    best_auceva=0
    '''
    for epoch in range(5):
        num=len(train_interaction_data)/train_batchSize
        num=int(num)
        

        for num in tqdm(range(num)):

            
            label,pos_input,pos_mask_input,pos_edge_input,neg_input,neg_mask_input,neg_edge_input,item_input=getTrainBatch(train_batchSize,num)

            _,loss,y_pred=sess.run([pos_train_op,model.pos_loss,model.pos_evaoutput],
              feed_dict = {
                           model.pos_inputs: pos_input, 
                           model.pos_mask_inputs: pos_mask_input,  
                           model.pos_edge_inputs: pos_edge_input, 

                           model.neg_inputs: neg_input, 
                           model.neg_mask_inputs: neg_mask_input,  
                           model.neg_edge_inputs: neg_edge_input,

                           model.item_inputs:item_input,
                           model.y_inputs:label,
                           model.tst: False, 
                           model.keep_prob: 1.0})

            if num%100==1:

              auc=roc_auc_score(label,y_pred)

              print('auc in train:',auc)
              print("epoch:",epoch,' num:',num,' loss:',loss)
              f.write("epoch:"+str(epoch)+' num:'+str(num)+' loss:'+str(loss)+' auc:'+str(auc)+"\n")
              f.flush()
        
        auc_eva,pos_result,result_ans=pos_eva(sess,model)
        if(auc_eva>best_auceva):
            np.save("pos_result.npy",pos_result)
            np.save("result_ans.npy",result_ans)
            save_path = saver.save(sess, "./model_save/model.ckpt")
            best_auceva=auc_eva
        print('epoch:',epoch,'auc on val:',auc_eva,'best_auceva',best_auceva)
        f.write('epoch:'+str(epoch)+' auc on val:'+str(auc_eva)+' best_auceva:'+str(best_auceva)+"\n" )
        f.flush()
       
    for epoch in range(5):

        num=len(train_interaction_data)/train_batchSize
        num=int(num)
        

        for num in tqdm(range(num)):

            
            label,pos_input,pos_mask_input,pos_edge_input,neg_input,neg_mask_input,neg_edge_input,item_input=getTrainBatch(train_batchSize,num)

            _,loss,y_pred =sess.run([neg_train_op,model.neg_loss,model.neg_evaoutput],
              feed_dict = {
                           model.pos_inputs: pos_input, 
                           model.pos_mask_inputs: pos_mask_input,  
                           model.pos_edge_inputs: pos_edge_input, 

                           model.neg_inputs: neg_input, 
                           model.neg_mask_inputs: neg_mask_input,  
                           model.neg_edge_inputs: neg_edge_input,

                           model.item_inputs:item_input,
                           model.y_inputs:label,
                           model.tst: False, 
                           model.keep_prob: 1.0})

            if num%100==1:

              auc=roc_auc_score(label,y_pred)

              print('auc in train:',auc)
              print("epoch:",epoch,' num:',num,' loss:',loss)
              f.write("epoch:"+str(epoch)+' num:'+str(num)+' loss:'+str(loss)+' auc:'+str(auc)+"\n")
              f.flush()

       
        auc_eva,neg_result,result_ans=neg_eva(sess,model)
        if(auc_eva>best_auceva):
            np.save("neg_result.npy",neg_result)
            np.save("result_ans.npy",result_ans)
            save_path = saver.save(sess, "./model_save/model.ckpt")
            best_auceva=auc_eva
        print('epoch:',epoch,'auc on val:',auc_eva,'best_auceva',best_auceva)
        f.write('epoch:'+str(epoch)+' auc on val:'+str(auc_eva)+' best_auceva:'+str(best_auceva)+"\n" )
        f.flush()  
        
    auc_eva,joint_result,result_ans=joint_eva(sess,model)
    if(auc_eva>best_auceva):
        np.save("jointresult.npy",joint_result)
        np.save("result_ans.npy",result_ans)
        save_path = saver.save(sess, "./model_save/model.ckpt")
        best_auceva=auc_eva
    print('epoch:',epoch,'auc on val:',auc_eva,'best_auceva',best_auceva)
    f.write('epoch:'+str(epoch)+' auc on val:'+str(auc_eva)+' best_auceva:'+str(best_auceva)+"\n" )
    f.flush() 
    ''' 
    for epoch in range(100):
        num=len(train_interaction_data)/2048
        num=int(num)
        

        for num in tqdm(range(num)):

            
            label,pos_input,pos_mask_input,pos_edge_input,neg_input,neg_mask_input,neg_edge_input,item_input=getTrainBatch(2048,num)

            _,loss,y_pred =sess.run([joint_train_op,model.joint_loss,model.joint_evaoutput],
              feed_dict = {
                           model.pos_inputs: pos_input, 
                           model.pos_mask_inputs: pos_mask_input,  
                           model.pos_edge_inputs: pos_edge_input, 

                           model.neg_inputs: neg_input, 
                           model.neg_mask_inputs: neg_mask_input,  
                           model.neg_edge_inputs: neg_edge_input,

                           model.item_inputs:item_input,
                           model.y_inputs:label,
                           model.tst: False, 
                           model.keep_prob: 1.0})

            if num%100==1:

              auc=roc_auc_score(label,y_pred)

              print('auc in train:',auc)
              print("epoch:",epoch,' num:',num,' loss:',loss)
              f.write("epoch:"+str(epoch)+' num:'+str(num)+' loss:'+str(loss)+' auc:'+str(auc)+"\n")
              f.flush()


        auc_eva,joint_result,result_ans=joint_eva(sess,model)
        if(auc_eva>best_auceva):
            np.save("jointresult.npy",joint_result)
            np.save("result_ans.npy",result_ans)
            save_path = saver.save(sess, "./model_save/model.ckpt")
            best_auceva=auc_eva
        print('epoch:',epoch,'auc on val:',auc_eva,'best_auceva',best_auceva)
        f.write('epoch:'+str(epoch)+' auc on val:'+str(auc_eva)+' best_auceva:'+str(best_auceva)+"\n" )
        f.flush()  
sess.close()

