from GNN_data_loader.data_utils import gen_batch
from GNN_utils.math_utils import evaluation
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time
from GNN_utils.math_graph import *
from GNN_data_loader.data_utils import *

class GNN():
    def __init__(self,W_file,V_file):
        
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        from os.path import join as pjoin
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        args = self.set_args()

        self.n, self.n_his, self.n_pred = args.n_route, args.n_his, args.n_pred
        #228,12,9
        Ks, Kt = args.ks, args.kt
        #3,3
        blocks = [[args.channel, 32, 64], [64, 32, 128]]
        # Load wighted adjacency matrix W
        W = weight_matrix(W_file)
        
        L = scaled_laplacian(W)
        Lk = cheb_poly_approx(L, Ks, self.n)
        tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
        
        # Data Preprocessing
        n_train, n_val, n_test = 34, 5, 5
        PeMS = data_gen(V_file, (n_train, n_val, n_test,args.channel), 
                        self.n, self.n_his + self.n_pred, 48)

        inputs, batch_size,  inf_mode = PeMS, PeMS.get_len('test'),  args.inf_mode
    
        load_path='GNN_output/models/'
        model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
        test_graph = tf.Graph()
        with test_graph.as_default():
            saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

        self.sess = tf.Session(graph=test_graph)

        saver.restore(self.sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')
        #./output/models/STGCN-9150

        self.pred = test_graph.get_collection('y_pred')
        
        self.step_idx = self.n_pred - 1
        
        

    def set_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--n_route', type=int, default=16)
        parser.add_argument('--n_his', type=int, default=10)
        parser.add_argument('--n_pred', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--epoch', type=int, default=11)
        parser.add_argument('--save', type=int, default=10)
        parser.add_argument('--ks', type=int, default=3)
        parser.add_argument('--kt', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--opt', type=str, default='RMSProp')
        parser.add_argument('--graph', type=str, default='default')
        parser.add_argument('--inf_mode', type=str, default='merge')
        parser.add_argument('--channel', type=str, default=8)
        args = parser.parse_args()
        return args

    def single_pred(self,sess, y_pred, seq, n_his, n_pred, step_idx, dynamic_batch=True):
        '''
        seq n_his+n_pred
        '''
        pred_list = []
        i = seq

        test_seq = np.copy(i[:, 0:self.n_his + 1, :, :])
        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            
            if isinstance(pred, list):
                pred = np.array(pred[0])
            #print(pred.shape,test_seq.shape)
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
        #  pred_array -> [n_pred, batch_size, n_route, C_0)
        pred_array = np.concatenate(pred_list, axis=1)
        return pred_array[step_idx], pred_array.shape[1]

    def get_predict(self,x_test):
        #x_test 长度 应该是 n_his+n_pred
        x_stats = {'mean': np.mean(x_test), 'std': np.std(x_test)}        
        x_test = (x_test - x_stats['mean'])/x_stats['std']
        x_test = np.array([x_test])
        
        y_test, len_test = self.single_pred(self.sess,self.pred,x_test,self.n_his,self.n_pred,self.step_idx)

        y_test = y_test*x_stats['std']+x_stats['mean']
        return y_test[0]
    
    def get_history(self,env,time_duration,single_his):
        keep_lane = [0,2,3,5,6,8,9,11]
                    #0,1,2,3,4,5,6,7
        #0 3
        #6 9
        #2 5
        #8 11
        ##0,2;4,6;1,3;5,7
        print(len(env.list_inter_log))
        time_now = int(env.get_current_time())
        his = []
        for T in range(time_now-time_duration,time_now,single_his):
            his_period = []
            for t in range(T,T+single_his):
                if env.list_inter_log[0][t]['action']<0:
                    continue
                his_t = []
                for inter in env.list_inter_log:
                    his_t.append([inter[t]['state']['lane_num_vehicle'][l] for l in keep_lane])
                his_t = np.array(his_t) #(n,8)
                his_period.append(his_t)
            his_period = np.array(his_period) #(*,n,8)
            his.append(np.max(his_period,axis = 0)) #sum --> (n,8)
        his = np.array(his)#(T,n,8)
        
        a,b,c = his.shape
        his_add = np.zeros((a+1,b,c))
        his_add[:a,:,:] = his
        his_add[a,:,:] = his[-1,:,:]
        return his_add

    def get_lanenum(self,env,action_num=4,time_duration = 600,single_his = 60):
        his = self.get_history(env,time_duration,single_his)
        pred = self.get_predict(his) # (n,8)
        #pred = np.average(his[:-1,:,:],axis=0)
        action_2_lane = [0,4,1,5]
        lane_num = []
        for inter in range(env.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            l=[]
            y = pred[inter]
            for a in range(action_num):
                press = (y[action_2_lane[a]]+y[action_2_lane[a]+2])/2
            l.append(press)
            lane_num.append(l)
        return lane_num
    def get_maxpred(self,env,action_num=4,time_duration = 600,single_his = 60):
        lane_num = self.get_lanenum(env,action_num=4,time_duration = 600,single_his = 60)
        return np.max(np.array(lane_num))


def addnew(x,num):
    '''
    在每两个数之间新加num个

    '''
    
    num = num+1
    x_new = []
    for i in range(len(x)-1):
        
        dif = x[i+1] - x[i]
        x_new.extend([x[i]+dif/num*k for k in range(num)])
    return x_new
        
    
if __name__ == "__main__":
    pass
    #gnn = GNN('W.csv','V.csv')
    #y = gnn.get_predict(np.random.uniform(size=(11,1,8)))
    #至少是 n_his+1
    