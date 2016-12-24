# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:15:10 2016

@author: li

optimize both throughput and connections
"""

#import sys
#sys.path.insert(0, '/home/li/Dropbox/KTH/numerical_analysis/ILPs')
from sdm import *
from gurobipy import *

np.random.seed(2010)

num_cores=3
num_slots=80

i = 4 
time_limit_routing = 1800 # 1000
time_limit_sa = 108 # 10800

filename = 'traffic_matrix_4.csv'
#    print filename
tm = []
with open(filename) as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        row = [float(u) for u in row]
        tm.append(row)

tm = np.array(tm)
#%% arch2
betav = np.array([0, 
    1e-5, 2e-5, 4e-5, 8e-5, 
    1e-4, 2e-4, 4e-4, 8e-4, 
    1e-3, 2e-3, 4e-3, 8e-3, 
    1e-2, 2e-2, 4e-2, 8e-2, 
    1e-1, 2e-1, 4e-1, 1, 10])
#betav = np.array([0, 0.04, 10])
connection_ub = []
throughput_ub = []
obj_ub = []

connection_lb = []
throughput_lb = []
obj_lb = []

connection_he = []
throughput_he = []
obj_he = []

for beta in betav:        
    m = Arch2_decompose(tm, num_slots=num_slots, num_cores=num_cores, 
        alpha=1,beta=beta)

    m.create_model_routing(mipfocus=1,timelimit=time_limit_routing,mipgap=0.01, method=2)
    connection_ub.append(m.connection_ub_)
    throughput_ub.append(m.throughput_ub_)
    obj_ub.append(m.obj_ub_)

#    m.create_model_sa(mipfocus=1,timelimit=10800,mipgap=0.01, method=2, 
#        SubMIPNodes=2000, heuristics=0.8)
#    connection_lb.append(m.connection_lb_)
#    throughput_lb.append(m.throughput_lb_)
#    obj_lb.append(m.obj_lb_)
#    m.write_result_csv('cnklist_lb_%d_%.2e.csv'%(i,beta), m.cnklist_sa)
    
    connection_lb.append(0)
    throughput_lb.append(0)
    obj_lb.append(0)

    m.heuristic()
    connection_he.append(m.obj_heuristic_connection_)
    throughput_he.append(m.obj_heuristic_throughput_)
    obj_he.append(m.obj_heuristic_)
    # write results
    m.write_result_csv('cnklist_heuristic_{}_{:2}.csv'.format(i, beta), m.cnklist_heuristic_)

result = np.array([betav,
                   connection_ub,throughput_ub,obj_ub,
                   connection_lb,throughput_lb,obj_lb,
                   connection_he,throughput_he,obj_he]).T
file_name = 'result_A2_4.csv'
with open(file_name, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['beta', 'connection_ub', 'throughput_ub', 
    'obj_ub', 'connection_lb', 'throughput_lb', 'obj_lb',
    'connection_he', 'throughput_he', 'obj_he'])
    writer.writerows(result)
