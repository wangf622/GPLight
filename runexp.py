import config
import copy
from pipeline import Pipeline
import os
import time
from multiprocessing import Process
import argparse
import os
import matplotlib
# matplotlib.use('TkAgg')

from script import get_traffic_volume

multi_process = True
TOP_K_ADJACENCY=-1
TOP_K_ADJACENCY_LANE=-1
PRETRAIN=False
NUM_ROUNDS=100
EARLY_STOP=False 
NEIGHBOR=False
SAVEREPLAY=False
ADJACENCY_BY_CONNECTION_OR_GEO=False
hangzhou_archive=True
ANON_PHASE_REPRE=[]

def parse_args():
    parser = argparse.ArgumentParser()
    # The file folder to create/log in
    parser.add_argument("--memo", type=str, default='0515_afternoon_Colight_6_6_bi')#1_3,2_2,3_3,4_4
    parser.add_argument("--env", type=int, default=1) #env=1 means you will run CityFlow
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--road_net", type=str, default='6_6')#'1_2') # which road net you are going to run
    parser.add_argument("--volume", type=str, default='300')#'300'
    parser.add_argument("--suffix", type=str, default="0.3_bi")#0.3

    global hangzhou_archive
    hangzhou_archive=False
    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY=5
    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE=5
    global NUM_ROUNDS
    NUM_ROUNDS=100
    global EARLY_STOP
    EARLY_STOP=False
    global NEIGHBOR
    # TAKE CARE
    NEIGHBOR=False
    global SAVEREPLAY # if you want to relay your simulation, set it to be True
    SAVEREPLAY=False
    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE
    ADJACENCY_BY_CONNECTION_OR_GEO=False

    #modify:TOP_K_ADJACENCY in line 154
    global PRETRAIN
    PRETRAIN=False
    parser.add_argument("--mod", type=str, default='CoLight')#SimpleDQN,SimpleDQNOne,GCN,CoLight,Lit
    parser.add_argument("--cnt",type=int, default=3600)#3600
    parser.add_argument("--gen",type=int, default=4)#4

    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers",type=int, default=7)
    parser.add_argument("--onemodel",type=bool, default=False)

    parser.add_argument("--visible_gpu", type=str, default="-1")
    global ANON_PHASE_REPRE
    tt=parser.parse_args()
    if 'CoLight_Signal' in tt.mod:
        #12dim
        ANON_PHASE_REPRE={
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
        }
    else:
        #12dim
        ANON_PHASE_REPRE={
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        }
    print('agent_name:%s',tt.mod)
    print('ANON_PHASE_REPRE:',ANON_PHASE_REPRE)
    

    return parser.parse_args()


def memo_rename(traffic_file_list):
    new_name = ""
    for traffic_file in traffic_file_list:
        if "synthetic" in traffic_file:
            sta = traffic_file.rfind("-") + 1
            print(traffic_file, int(traffic_file[sta:-4]))
            new_name = new_name + "syn" + traffic_file[sta:-4] + "_"
        elif "cross" in traffic_file:
            sta = traffic_file.find("equal_") + len("equal_")
            end = traffic_file.find(".xml")
            new_name = new_name + "uniform" + traffic_file[sta:end] + "_"
        elif "flow" in traffic_file:
            new_name = traffic_file[:-4]
    new_name = new_name[:-1]
    return new_name

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1

def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf, # experiment config
                   dic_agent_conf=dic_agent_conf, # RL agent config
                   dic_traffic_env_conf=dic_traffic_env_conf, # the simolation configuration
                   dic_path=dic_path # where should I save the logs?
                   )
    global multi_process
    ppl.run(multi_process=multi_process)

    print("pipeline_wrapper end")
    return



def main(memo, traffic_file_list,roadnet_file,num_row,num_col,path_to_data,workers):
    num_intersections = num_row * num_col
    print('num_intersections:',num_intersections)

    ENVIRONMENT = "anon"

    process_list = []
    n_workers = workers     #len(traffic_file_list)
    multi_process = True

    global PRETRAIN
    global NUM_ROUNDS
    global EARLY_STOP
    for traffic_file in traffic_file_list:
        global TOP_K_ADJACENCY
        global TOP_K_ADJACENCY_LANE
        global NEIGHBOR
        global SAVEREPLAY
        global ADJACENCY_BY_CONNECTION_OR_GEO
        global ANON_PHASE_REPRE
        global hangzhou_archive
        deploy_dic_traffic_env_conf = {
            "NUM_INTERSECTIONS": num_intersections,
            "NUM_ROW": num_row,
            "NUM_COL": num_col,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": roadnet_file,
            "ACTION_PATTERN": "set",
            "MIN_ACTION_TIME": 10,
            "YELLOW_TIME": 5,
            "ALL_RED_TIME": 0,
            "NUM_PHASES": 2,
            "NUM_LANES": 1,
            "ACTION_DIM": 2,
            "MEASURE_TIME": 10,
            "IF_GUI": false,
            "DEBUG": false,
            "INTERVAL": 1,
            "THREADNUM": 8,
            "SAVEREPLAY": false,
            "RLTRAFFICLIGHT": true,
            "DIC_FEATURE_DIM": {
                "D_LANE_QUEUE_LENGTH": [4],
                "D_LANE_NUM_VEHICLE": [4],
                "D_COMING_VEHICLE": [12],
                "D_LEAVING_VEHICLE": [12],
                "D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1": [4],
                "D_CUR_PHASE": [8],
                "D_NEXT_PHASE": [1],
                "D_TIME_THIS_PHASE": [1],
                "D_TERMINAL": [1],
                "D_LANE_SUM_WAITING_TIME": [4],
                "D_VEHICLE_POSITION_IMG": [4,60],
                "D_VEHICLE_SPEED_IMG": [4,60],
                "D_VEHICLE_WAITING_TIME_IMG": [4,60],
                "D_PRESSURE": [1],
                "D_ADJACENCY_MATRIX": [5],
                "D_ADJACENCY_MATRIX_LANE": [5],
                "D_CUR_PHASE_0": [1],
                "D_LANE_NUM_VEHICLE_0": [4],
                "D_CUR_PHASE_1": [1],
                "D_LANE_NUM_VEHICLE_1": [4],
                "D_CUR_PHASE_2": [1],
                "D_LANE_NUM_VEHICLE_2": [4],
                "D_CUR_PHASE_3": [1],
                "D_LANE_NUM_VEHICLE_3": [4]
            },
            "LIST_STATE_FEATURE": ["cur_phase","lane_num_vehicle","adjacency_matrix","adjacency_matrix_lane"],
            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0
            },
            "LANE_NUM": {"LEFT": 1,"RIGHT": 1,"STRAIGHT": 1},
            "PHASE": {
                "anon": {
                    "1": [0,1,0,1,0,0,0,0],
                    "2": [0,0,0,0,0,1,0,1],
                    "3": [1,0,1,0,0,0,0,0],
                    "4": [0,0,0,0,1,0,1,0]
                }
            },
            "USE_LANE_ADJACENCY": true,
            "ONE_MODEL": false,
            "NUM_AGENTS": 1,
            "TOP_K_ADJACENCY": 5,
            "ADJACENCY_BY_CONNECTION_OR_GEO": false,
            "TOP_K_ADJACENCY_LANE": 5,
            "SIMULATOR_TYPE": "anon",
            "BINARY_PHASE_EXPANSION": true,
            "FAST_COMPUTE": true,
            "NEIGHBOR": false,
            "MODEL_NAME": "CoLight",
            "VOLUME": "300",
            "phase_expansion": {
                "1": [0,1,0,1,0,0,0,0],
                "2": [0,0,0,0,0,1,0,1],
                "3": [1,0,1,0,0,0,0,0],
                "4": [0,0,0,0,1,0,1,0],
                "5": [1,1,0,0,0,0,0,0],
                "6": [0,0,1,1,0,0,0,0],
                "7": [0,0,0,0,0,0,1,1],
                "8": [0,0,0,0,1,1,0,0]
            },
            "phase_expansion_4_lane": {
                "1": [1,1,0,0],
                "2": [0,0,1,1]
            }
        }

        deploy_dic_exp_conf = {
            "TRAFFIC_FILE": [traffic_file],
            "ROADNET_FILE": roadnet_file,
            "RUN_COUNTS": 3600,
            "MODEL_NAME": "CoLight",
            "NUM_ROUNDS": 100,
            "NUM_GENERATORS": 4,
            "LIST_MODEL": ["Fixedtime","SOTL","Deeplight","SimpleDQN"],
            "LIST_MODEL_NEED_TO_UPDATE": ["Deeplight","SimpleDQN","CoLight","GCN","SimpleDQNOne","Lit"],
            "MODEL_POOL": false,
            "NUM_BEST_MODEL": 3,
            "PRETRAIN": false,
            "PRETRAIN_MODEL_NAME": "CoLight",
            "PRETRAIN_NUM_ROUNDS": 0,
            "PRETRAIN_NUM_GENERATORS": 15,
            "AGGREGATE": false,
            "DEBUG": false,
            "EARLY_STOP": false,
            "MULTI_TRAFFIC": false,
            "MULTI_RANDOM": false
        }
        deploy_dic_agent_conf = {
            "TRAFFIC_FILE": traffic_file,
            "CNN_layers": [[32,32]],
            "att_regularization": false,
            "rularization_rate": 0.03,
            "LEARNING_RATE": 0.001,
            "SAMPLE_SIZE": 1000,
            "BATCH_SIZE": 20,
            "EPOCHS": 100,
            "UPDATE_Q_BAR_FREQ": 5,
            "UPDATE_Q_BAR_EVERY_C_ROUND": false,
            "GAMMA": 0.8,
            "MAX_MEMORY_LEN": 10000,
            "PATIENCE": 10,
            "D_DENSE": 20,
            "N_LAYER": 2,
            "EPSILON": 0.8,
            "EPSILON_DECAY": 0.95,
            "MIN_EPSILON": 0.2,
            "LOSS_FUNCTION": "mean_squared_error",
            "SEPARATE_MEMORY": false,
            "NORMAL_FACTOR": 20,
        }
        
        deploy_dic_path = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_DATA": path_to_data,
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            "PATH_TO_ERROR": os.path.join("errors", memo)
            "PATH_TO_PRETRAIN_DATA": "data/template",
            "PATH_TO_AGGREGATE_SAMPLES": "records/initial"
        }

        deploy_dic_path["PATH_TO_DATA"] = 'data/Hangzhou/4_4'
        

        if multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_exp_conf,
                                deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                             dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if multi_process:
        for i in range(0, len(process_list), n_workers):
            i_max = min(len(process_list), i + n_workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)


    return memo


if __name__ == "__main__":
    args = parse_args()
    #memo = "multi_phase/optimal_search_new/new_headway_anon"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    main(args.memo, ['hangzhou.json'],'roadnet_4_4.json',4,4,'data/Hangzhou',args.workers)