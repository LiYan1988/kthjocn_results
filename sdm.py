# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:09:29 2016

@author: liyan

Generate random traffic matrices
"""

import numpy as np
#import pandas as pd
from gurobipy import *
import numpy as np
import time
import itertools
import copy
import csv
from scipy.linalg import toeplitz

class Traffic(object):
    
    def __init__(self, num_pods, max_pod_connected, min_pod_connected=0,
                 capacity_choices=[1, 10, 100, 200, 400, 1000], 
                 capacity_probs=None):
        # total number of PODs
        self.num_pods = num_pods 
        # max value of PODs each POD is connected, one POD can be connected to 
        # at most num_pods-1 other PODs
        if max_pod_connected >= num_pods:
            max_pod_connected = num_pods-1
        elif max_pod_connected < 0:
            max_pod_connected = min_pod_connected
        self.max_pod_connected = max_pod_connected
        # min value of PODs connected
        if min_pod_connected < 0:
            min_pod_connected = 0
        elif min_pod_connected > max_pod_connected:
            min_pod_connected = max_pod_connected
        self.min_pod_connected = min_pod_connected
        # choices of capacity values
        self.capacity_choices = capacity_choices
        # POD id list
        self.pod_id_list = ['pod_%d' % i for i in range(self.num_pods)]
        # capacity probabilities
        if capacity_probs is None:
            self.capacity_probs = (np.ones((1, len(capacity_choices)))*1.0
                                   /len(capacity_choices))
        else:
            self.capacity_probs = capacity_probs
        
    def generate_traffic(self):
        """Generate random traffic matrix
        """
        # each POD is connected to x_i other PODs, where
        self.pod_connectivity = np.random.randint(self.min_pod_connected, 
                                                  self.max_pod_connected+1, 
                                                self.num_pods)
        self.traffic_matrix = np.zeros((self.num_pods, self.num_pods))
        for i in range(self.num_pods):
            # POD i cannot connect to itself
            pod_choice = np.delete(np.arange(self.num_pods), i)
            connected_pods = np.random.choice(pod_choice, 
                                              self.pod_connectivity[i], 
                                             replace=False)
            connected_capacities = np.random.choice(self.capacity_choices,
                                                    (1, len(connected_pods)),
                                                    replace=True,
                                                    p=self.capacity_probs)
            self.traffic_matrix[i, connected_pods] = connected_capacities
            
            
    def value_count(self):
        """Count value frequency and throughput
        """
        self.number_count = {i:0 for i in self.capacity_choices}
        self.throughput_count = {i:0 for i in self.capacity_choices}
        for i in range(self.num_pods):
            for j in range(self.num_pods):
                for k in self.capacity_choices:
                    if self.traffic_matrix[i][j] == k:
                        self.number_count[k] += 1
                        self.throughput_count[k] += k

            
# Architecture 1
class Arch1_decompose(object):
    """Create models for different SDM DCN architectures
    """
    def __init__(self, traffic_matrix, num_slots=320, num_cores=10,
                 slot_capacity =25, num_guard_slot=1, alpha=1, beta=0):
        """Initialize 
        """
        # traffic matrix
        self.traffic_matrix = traffic_matrix
        # number of PODs
        self.num_pods = traffic_matrix.shape[0]
        # capacity per spectrum slot, Gbps
        self.slot_capacity = slot_capacity
        # number of slot as guardband
        self.num_guard_slot = num_guard_slot
        # number of slots
        self.num_slots = num_slots
        # number of cores
        self.num_cores = num_cores
        # number of total demands
        self.total_demands = sum(self.traffic_matrix.flatten()>0)
        # weight factor
        self.alpha = alpha
        self.beta = beta
        
        c_max = self.num_slots * self.slot_capacity
        self.tm = self.traffic_matrix.copy()
        # number of blocked demands at the begining
        self.num_blocked2 = sum(self.tm.flatten()>c_max)
        # remove those demands with too large capacities
        self.tm[self.tm>c_max] = 0
        
        # Model data
        # set of pods, pod_0, ..., pod_(N_p-1)
        pods = list(range(self.num_pods))
        # pairs of traffic demands
        traffic_pairs = tuplelist([(i, j) for i in pods for j in pods
                            if self.tm[i, j]>0])
        # traffic capacities
        traffic_capacities = {}
        for u in traffic_pairs:
            if self.tm[u[0],u[1]] > 0:
                traffic_slot = int(np.ceil(float(self.tm[u[0],u[1]]) / 
                            self.slot_capacity) + self.num_guard_slot)
                traffic_capacities[u] = traffic_slot
#                print(traffic_slot)
                
        # set of cores
        cores = list(range(self.num_cores))
        
        self.pods = pods
        self.traffic_pairs = traffic_pairs
        self.traffic_capacities = traffic_capacities
        self.cores = cores
        
    def create_model_routing(self, **kwargs):
        """ILP
        """
        # Model
        tic = time.clock()
        model = Model('arch1_routing')
        
        # binary variable: c_i,u,k = 1 if connection u uses core k in POD i
        core_usage = {}
        for u in self.traffic_pairs:
            for k in self.cores:
                for i in u:
                    core_usage[u,i,k] = model.addVar(vtype=GRB.BINARY)
                  
        # the absolute difference between the core index chosen by a traffic
#        core_diff = {}
#        for u in self.traffic_pairs:
#            for k in self.cores:
#                core_diff[u] = model.addVar(vtype=GRB.INTEGER, obj=0.1)
                    
        suc = {}
        for u in self.traffic_pairs:
            suc[u] = model.addVar(vtype=GRB.BINARY, obj=-(self.alpha+self.beta*self.tm[u[0],u[1]]))

        model.update()
        
        # one connection uses one core
        for u in self.traffic_pairs:
            model.addConstr(quicksum(core_usage[u,u[0],k] for k in self.cores)==suc[u])
            model.addConstr(quicksum(core_usage[u,u[1],k] for k in self.cores)==suc[u])
            
        # flow per core
        for i in self.pods:
            tmp = list((i, j) for (i, j) in self.traffic_pairs.select(i, '*'))
            tmp0 = list((j, i) for (j, i) in self.traffic_pairs.select('*', i))
            tmp.extend(tmp0)
            for k in self.cores:
                model.addConstr(quicksum(self.traffic_capacities[u]*
                core_usage[u, i, k] for u in tmp)<=self.num_slots)

        # limit core switches
#        for u in self.traffic_pairs:
#            model.addConstr(core_diff[u] >= 
#                quicksum(core_usage[u,u[0],k] for k in self.cores)-
#                quicksum(core_usage[u,u[1],k] for k in self.cores))
#            model.addConstr(core_diff[u] >= 
#                quicksum(core_usage[u,u[0],k] for k in self.cores)-
#                quicksum(core_usage[u,u[1],k] for k in self.cores))
        
        # params
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(model.params, key, value)
        
        model.optimize()
        toc = time.clock()
        
        self.model_routing = model
        self.runtime = toc-tic
        
        pcset = {} # set of connections using pod i, core k
        for i in self.pods:
            for k in self.cores:
                pcset[i,k] = []
        
        for u in self.traffic_pairs:
            for k in self.cores:
                for i in u:
                    if core_usage[u,i,k].x==1:
                        pcset[i,k].append(u)
        self.pcset_dc = pcset
                
        suclist = [] # set of successfully allocated connections
        for u in self.traffic_pairs:
            if suc[u].x==1:
               suclist.append(u)
        self.suclist_dc = suclist
        
        core_usagex = {} # core allocation
        for u in self.traffic_pairs:
            if u in suclist:
                for i in u:
                    for k in self.cores:
                        if core_usage[u,i,k].x==1:
                            core_usagex[u,i] = k
                            break
            else:
                core_usagex[u,u[0]] = -1
                core_usagex[u,u[1]] = -1
        self.core_usagex = core_usagex
        
        self.connection_ub_ = len(suclist)
        self.throughput_ub_ = sum(self.tm[u[0],u[1]] for u in self.suclist_dc)
        self.obj_ub_ = self.alpha*self.connection_ub_+self.beta*self.throughput_ub_
                
    def create_model_sa(self, **kwargs):
        """Spectrum assignment ILP
        """

        smallM = self.num_slots
        bigM = 10*smallM
        
        # Model
        tic = time.clock()
        model_sa = Model('arch1_sa')

        # binary variable: spectrum order
        spec_order = {}
        for i in self.pods:
            for k in self.cores:
                for c in itertools.combinations(self.pcset_dc[i,k],2):
                    spec_order[c[0],c[1]] = model_sa.addVar(vtype=GRB.BINARY)
        
        # continuous variable: first spectrum slot index
        # binary: fail?
        spec_idx = {}
        isfail = {}
        for u in self.suclist_dc:
            spec_idx[u] = model_sa.addVar(vtype=GRB.CONTINUOUS)
            isfail[u] = model_sa.addVar(vtype=GRB.BINARY, obj=self.alpha+self.beta*self.tm[u[0],u[1]])
            
        model_sa.update()
        
        # constraints: order
        for i in self.pods:
            for k in self.cores:
                for c in itertools.combinations(self.pcset_dc[i,k],2):
                    model_sa.addConstr(spec_idx[c[0]]+self.traffic_capacities[c[0]]-
                    spec_idx[c[1]]+bigM*spec_order[c[0],c[1]]<=bigM)
                    model_sa.addConstr(spec_idx[c[1]]+self.traffic_capacities[c[1]]-
                    spec_idx[c[0]]+bigM*(1-spec_order[c[0],c[1]])<=bigM)

        for u in self.suclist_dc:
            model_sa.addConstr(bigM*isfail[u]>=
            spec_idx[u]+self.traffic_capacities[u]-smallM)

        # params
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(model_sa.params, key, value)
                
        model_sa.optimize()
        toc = time.clock()
        
        self.model_sa = model_sa
        self.runtime_sa = toc-tic
        
        tmp = list(self.suclist_dc)
        for u in self.suclist_dc:
            if isfail[u].x == 1:
                tmp.remove(u)
        self.suclist_sa = tmp
        
        self.spec_idxx = {} # spectrum slots allocation
        for u in self.traffic_pairs:
            if u in self.suclist_dc:
                self.spec_idxx[u] = int(round(spec_idx[u].x))
            else:
                self.spec_idxx[u] = -1
                self.core_usagex[u,u[0]] = -1
                self.core_usagex[u,u[1]] = -1                
            self.connection_lb_ = len(self.suclist_sa)
            self.throughput_lb_ = sum(self.tm[u[0],u[1]] for u in self.suclist_sa)

        # construct the resource tensor
        self.cnklist_sa = []
        tensor_milp = np.ones((self.num_pods, self.num_cores, self.num_slots), dtype=np.int8)
        for u in self.suclist_sa:
            src = u[0]
            dst = u[1]
            core_src = self.core_usagex[u,src]
            core_dst = self.core_usagex[u,dst]
            spec_idx = self.spec_idxx[u]
            spec_bd = self.traffic_capacities[u]
            tmp = [src, dst, spec_idx, spec_bd, core_src, core_dst, self.tm[u]]
            self.cnklist_sa.append(tmp)
            res_src = tensor_milp[src,core_src,spec_idx:(spec_idx+spec_bd)]
            res_dst = tensor_milp[dst,core_dst,spec_idx:(spec_idx+spec_bd)]
            if (sum(res_src)==spec_bd) and (sum(res_dst)==spec_bd):
                res_src[:] = 0
                res_dst[:] = 0
        self.tensor_milp = tensor_milp
        self.efficiency_milp = (float(sum(self.tm[i] for i in self.suclist_sa))/
            sum(self.traffic_capacities[i]*self.slot_capacity 
            for i in self.suclist_sa))
        self.obj_lb_ = self.alpha*self.connection_lb+self.beta*self.throughput_lb
            
    def write_result_csv(self, file_name, suclist):
        with open(file_name, 'w') as f:
            f.write('src,dst,spec,slots_used,core_src,core_dst,tfk_slot\n')
            for c in suclist:
                wstr = '{},{},{},{},{},{},{}\n'.format(c[0], c[1], c[2], 
                    c[3], c[4], c[5], c[6])
                f.write(wstr)

    def one_runs(self, a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        isone = np.concatenate(([0], np.equal(a, 1).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isone))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        ranges[:,1] = ranges[:,1]-ranges[:,0]
        return ranges  
                        
    def check(self, cnklist):
        """Check feasibility of solution
        """
        # check if any two connections are overlapped
        n_overlap=0
        for (u,v) in itertools.combinations(cnklist,2):
            if set(u)&set(v):
                cout_u = self.cnk_resource[u][2]
                cin_u = self.cnk_resource[u][3]
                si_u = self.cnk_resource[u][1]
                sb_u = self.traffic_capacities[u]

                cout_v = self.cnk_resource[v][2]
                cin_v = self.cnk_resource[v][3]
                si_v = self.cnk_resource[v][1]
                sb_v = self.traffic_capacities[v]
                
                if set([(u[0], cout_u), (u[1], cin_u)])&set([(v[0], cout_v), (v[1], cin_v)]):
                    if (si_u>=si_v and si_v+sb_v-1>=si_u) or (si_v>=si_u and si_u+sb_u-1>=si_v):
                        print [(u[0], cout_u), (u[1], cin_u)]
                        print [(v[0], cout_v), (v[1], cin_v)]
                        print 'wrong'
                        n_overlap+=1
        
        # check if any connection is out of range 
        n_oof = 0
        for u in self.cnk_group_suc:
            si = self.cnk_resource[u][1]
            sb = self.traffic_capacities[u]
            if si+sb-1>self.num_slots:
                n_oof+=1
        return (n_overlap, n_oof)
        
    def sa_heuristic(self, ascending1=True,ascending2=True):
        """Spectrum assignment heuristi
        ascending1: order of allocating connections in suclist
        ascending2: order of allocating connections in remain list
        """
        suclist = list(self.suclist_dc)
        suclist_tm = [self.traffic_capacities[u] for u in suclist]
        if ascending1:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
            
        IS_list = {} # independent set
        IS_list[0] = []
        cl_list = {}
        cl_list[0] = set()
        i = 0
        while len(suclist):
            tmplist = list(suclist)
            for u in tmplist:
                src = u[0]
                dst = u[1]
                src_core = self.core_usagex[u,src]
                dst_core = self.core_usagex[u,dst]
                if ((src,src_core) not in cl_list[i]) and ((dst, dst_core) not in cl_list[i]):
                    # add connection if it's independent to element in IS_list[i]
                    IS_list[i].append(u)
                    cl_list[i].add((src,src_core))
                    cl_list[i].add((dst,dst_core))
                    tmplist.remove(u)
            i += 1
            IS_list[i] = []
            cl_list[i] = set()
            suclist = tmplist
            
        del cl_list[i]
        del IS_list[i]

        self.obj_sah_ = 0
        self.obj_sah_connection_ = 0
        self.obj_sah_throughput_ = 0
        suclist = []
        cnklist = []
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots),dtype=np.int0)        
        for i in range(len(IS_list)):
            for u in IS_list[i]:
                src = u[0]
                dst = u[1]
                src_core = self.core_usagex[u,src]
                dst_core = self.core_usagex[u,dst]
                tmpsrc = restensor[src,src_core,:]
                tmpdst = restensor[dst,dst_core,:]
                tmp = tmpsrc*tmpdst
                tmpavail = self.one_runs(tmp)
                tmpidx = np.where(tmpavail[:,1]>=self.traffic_capacities[u])[0]
                if tmpidx.size:
                   spec_idx = tmpavail[tmpidx[0],0]
                   restensor[src,src_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
                   restensor[dst,dst_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
                   self.obj_sah_ += self.alpha+self.beta*self.tm[src,dst]
                   self.obj_sah_connection_ += 1
                   self.obj_sah_throughput_ += self.tm[src,dst]
                   tmp = [src, dst, spec_idx, self.traffic_capacities[u], 
                      src_core, dst_core, self.tm[u]]
                   cnklist.append(tmp)
                   suclist.append(u)

        remain_cnk = [u for u in self.traffic_pairs if u not in suclist]
        remain_tm = [self.traffic_capacities[u] for u in remain_cnk]
        if ascending2:
            remain_cnk = [x for (y,x) in sorted(zip(remain_tm,remain_cnk))]
        else:
            remain_cnk = [x for (y,x) in sorted(zip(remain_tm,remain_cnk), reverse=False)]
            
        for u in remain_cnk:
            src = u[0]
            dst = u[1]
            tmpsrc = restensor[src,:,:]
            tmpdst = restensor[dst,:,:]
            tmpcmb = np.zeros((self.num_cores**2, self.num_slots))
            k = 0
            avail_slots = {}
            for ksrc in self.cores:
                for kdst in self.cores:
                    tmpcmb[k,:] = tmpsrc[ksrc,:]*tmpdst[kdst,:]
                    tmpavail = self.one_runs(tmpcmb[k,:])
                    tmpidx = np.where(tmpavail[:,1]>=self.traffic_capacities[u])[0]
                    if not tmpidx.size:
                        avail_slots[ksrc,kdst] = np.array([-1, self.num_slots+1])
                    else:
                        idxm = np.argmin(tmpavail[tmpidx,1])
                        avail_slots[ksrc,kdst] = np.array(tmpavail[tmpidx[idxm],:])
                    k += 1
            avail_slots = list(sorted(avail_slots.iteritems(), key=lambda (x,y):y[1]))
            # avail_slots[0] has the form of ((core_out,core_in), [spec_idx,available_slots])
            if avail_slots[0][1][1]<=self.num_slots:
                src_core = avail_slots[0][0][0]
                dst_core = avail_slots[0][0][1]
                spec_idx = avail_slots[0][1][0]
                spec_bd = self.traffic_capacities[u]
                restensor[src,src_core,spec_idx:(spec_idx+spec_bd)] = 0
                restensor[dst,dst_core,spec_idx:(spec_idx+spec_bd)] = 0
                self.obj_sah_ += self.alpha+self.beta*self.tm[src,dst]
                self.obj_sah_connection_ += 1
                self.obj_sah_throughput_ += self.tm[src,dst]
                tmp = [src, dst, spec_idx, self.traffic_capacities[u], 
                   src_core, dst_core, self.tm[u]]
                cnklist.append(tmp)
        self.tensor_heuristic = restensor
#        self.efficiency_heuristic = (float(sum(self.tm[i] for i in suclist))/
#            sum(self.traffic_capacities[i]*self.slot_capacity 
#            for i in suclist))
        self.suclist_heuristic = cnklist
        
    def sa_heuristic_aff(self, ascending=True):
        """First fit with optimized core allocation
        """
        # ordering the connections
        suclist = list(self.suclist_dc)
        suclist_tm = [self.traffic_capacities[u] for u in suclist]
        if ascending:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
        
        # first fit
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots), dtype=np.int0)
        self.obj_affopt_ = 0
        self.obj_affopt_connection_ = 0
        self.obj_affopt_throughput_ = 0
        self.cnklist_affopt = [] # list of successfully allocated connections
        for i,u in enumerate(suclist):
            src = u[0]
            dst = u[1]
            src_core = self.core_usagex[u,src]
            dst_core = self.core_usagex[u,dst]
            tmpsrc = restensor[src,src_core,:]
            tmpdst = restensor[dst,dst_core,:]
            tmp = tmpsrc*tmpdst
            tmpavail = self.one_runs(tmp)
            tmpidx = np.where(tmpavail[:,1]>=self.traffic_capacities[u])[0]
            if tmpidx.size:
               spec_idx = tmpavail[tmpidx[0],0]
               restensor[src,src_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
               restensor[dst,dst_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
               self.obj_affopt_ += self.alpha+self.beta*self.tm[src,dst]
               self.obj_affopt_connection_ += 1
               self.obj_affopt_throughput_ += self.tm[src,dst]
               tmp = [src, dst, spec_idx, self.traffic_capacities[u], 
                      src_core, dst_core, self.tm[u]]
               self.cnklist_affopt.append(tmp)
        
    def heuristic(self):
        objbest = 0
        objcnk = 0
        objthp = 0
        cnklist = []
        self.sa_heuristic(ascending1=True, ascending2=True)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.suclist_heuristic
        self.sa_heuristic(ascending1=True, ascending2=False)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.suclist_heuristic
        self.sa_heuristic(ascending1=False, ascending2=True)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.suclist_heuristic
        self.sa_heuristic(ascending1=False, ascending2=False)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.suclist_heuristic
            
        self.sa_heuristic_aff(ascending=True)
        if objbest < self.obj_affopt_:
            objbest = self.obj_affopt_
            objcnk = self.obj_affopt_connection_
            objthp = self.obj_affopt_throughput_
            cnklist = self.cnklist_affopt
        self.sa_heuristic_aff(ascending=False)
        if objbest < self.obj_affopt_:
            objbest = self.obj_affopt_
            objcnk = self.obj_affopt_connection_
            objthp = self.obj_affopt_throughput_
            cnklist = self.cnklist_affopt
            
        self.aff(ascending=True)
        if objbest < self.obj_aff_:
            objbest = self.obj_aff_
            objcnk = self.obj_aff_connection_
            objthp = self.obj_aff_throughput_
            cnklist = self.cnklist_aff
        self.aff(ascending=False)
        if objbest < self.obj_aff_:
            objbest = self.obj_aff_
            objcnk = self.obj_aff_connection_
            objthp = self.obj_aff_throughput_
            cnklist = self.cnklist_aff
        
        self.obj_heuristic_ = objbest
        self.obj_heuristic_connection_ = objcnk
        self.obj_heuristic_throughput_ = objthp
        self.cnklist_heuristic_ = cnklist
        
    def aff(self, ascending=True):
        """First fit according to the given connection list
        """
        suclist = list(self.traffic_pairs)
        suclist_tm = [self.traffic_capacities[u] for u in suclist]
        if ascending:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
        
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots), dtype=np.int0)
        self.obj_aff_ = 0
        self.obj_aff_connection_ = 0
        self.obj_aff_throughput_ = 0
        self.cnklist_aff = []
        for i,u in enumerate(suclist):
            src = u[0]
            dst = u[1]
            core_candidates = [(x,y) for x in self.cores for y in self.cores]
            for src_core, dst_core in core_candidates:
                tmpsrc = restensor[src,src_core,:]
                tmpdst = restensor[dst,dst_core,:]
                tmp = tmpsrc*tmpdst
                tmpavail = self.one_runs(tmp)
                tmpidx = np.where(tmpavail[:,1]>=self.traffic_capacities[u])[0]
                if tmpidx.size:
                   spec_idx = tmpavail[tmpidx[0],0]
                   restensor[src,src_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
                   restensor[dst,dst_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
                   self.obj_aff_ += self.alpha+self.beta*self.tm[src,dst]
                   self.obj_aff_connection_ += 1
                   self.obj_aff_throughput_ += self.tm[src,dst]
                   tmp = [src, dst, spec_idx, self.traffic_capacities[u], 
                          src_core, dst_core, self.tm[u]]
                   self.cnklist_aff.append(tmp)
                   break
                
    def save_tensor(self, tensor, filename):
        """Save resource tensor
        save as csv
        """
        tmp = tensor.reshape((-1, self.num_slots))
        np.savetxt(filename, tmp, fmt='%1d',delimiter=',')
        # for load the saved tensor
        # tmp = np.loadtxt(filename, delimiter=',')
        # tensor = tmp.reshape((self.num_pods, self.num_cores, self.num_slots))
        
        
class Arch2_decompose(object):
    """Create models for different SDM DCN architectures
    """
    def __init__(self, traffic_matrix, num_slots=320, num_cores=10,
                 slot_capacity =25, num_guard_slot=1, alpha=1, beta=0):
        """Initialize 
        """
        # traffic matrix
        self.traffic_matrix = traffic_matrix
        # number of PODs
        self.num_pods = traffic_matrix.shape[0]
        # capacity per spectrum slot, Gbps
        self.slot_capacity = slot_capacity*num_cores
        # number of slot as guardband
        self.num_guard_slot = num_guard_slot
        # number of slots
        self.num_slots = num_slots
        # number of cores
        self.num_cores = 1
        # number of total demands
        self.total_demands = sum(self.traffic_matrix.flatten()>0)
        # weight factor
        self.alpha = alpha
        self.beta = beta
        
        c_max = self.num_slots * self.slot_capacity
        self.tm = self.traffic_matrix.copy()
        # number of blocked demands at the begining
        self.num_blocked2 = sum(self.tm.flatten()>c_max)
        # remove those demands with too large capacities
        self.tm[self.tm>c_max] = 0
        
        # Model data
        # set of pods, pod_0, ..., pod_(N_p-1)
        pods = list(range(self.num_pods))
        # pairs of traffic demands
        traffic_pairs = tuplelist([(i, j) for i in pods for j in pods
                            if self.tm[i, j]>0])
        # traffic capacities
        traffic_capacities = {}
        for u in traffic_pairs:
            if self.tm[u[0],u[1]] > 0:
                traffic_slot = int(np.ceil(float(self.tm[u[0],u[1]]) / 
                            self.slot_capacity) + self.num_guard_slot)
                traffic_capacities[u] = traffic_slot
#                print(traffic_slot)
                
        # set of cores
        cores = list(range(self.num_cores))
        
        self.pods = pods
        self.traffic_pairs = traffic_pairs
        self.traffic_capacities = traffic_capacities
        self.cores = cores
        
    def create_model_routing(self, **kwargs):
        """ILP
        """
        # Model
        tic = time.clock()
        model = Model('Arch2_routing')
        
        # binary variable: c_i,u,k = 1 if connection u uses core k in POD i
        core_usage = {}
        for u in self.traffic_pairs:
            for k in self.cores:
                for i in u:
                    core_usage[u,i,k] = model.addVar(vtype=GRB.BINARY)
                  
        # the absolute difference between the core index chosen by a traffic
#        core_diff = {}
#        for u in self.traffic_pairs:
#            for k in self.cores:
#                core_diff[u] = model.addVar(vtype=GRB.INTEGER, obj=0.1)
                    
        suc = {}
        for u in self.traffic_pairs:
            suc[u] = model.addVar(vtype=GRB.BINARY, obj=-(self.alpha+self.beta*self.tm[u[0],u[1]]))

        model.update()
        
        # one connection uses one core
        for u in self.traffic_pairs:
            model.addConstr(quicksum(core_usage[u,u[0],k] for k in self.cores)==suc[u])
            model.addConstr(quicksum(core_usage[u,u[1],k] for k in self.cores)==suc[u])
            
        # flow per core
        for i in self.pods:
            tmp = list((i, j) for (i, j) in self.traffic_pairs.select(i, '*'))
            tmp0 = list((j, i) for (j, i) in self.traffic_pairs.select('*', i))
            tmp.extend(tmp0)
            for k in self.cores:
                model.addConstr(quicksum(self.traffic_capacities[u]*
                core_usage[u, i, k] for u in tmp)<=self.num_slots)

        # limit core switches
#        for u in self.traffic_pairs:
#            model.addConstr(core_diff[u] >= 
#                quicksum(core_usage[u,u[0],k] for k in self.cores)-
#                quicksum(core_usage[u,u[1],k] for k in self.cores))
#            model.addConstr(core_diff[u] >= 
#                quicksum(core_usage[u,u[0],k] for k in self.cores)-
#                quicksum(core_usage[u,u[1],k] for k in self.cores))
        
        # params
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(model.params, key, value)
        
        model.optimize()
        toc = time.clock()
        
        self.model_routing = model
        self.runtime = toc-tic
        
        pcset = {} # set of connections using pod i, core k
        for i in self.pods:
            for k in self.cores:
                pcset[i,k] = []
        
        for u in self.traffic_pairs:
            for k in self.cores:
                for i in u:
                    if core_usage[u,i,k].x==1:
                        pcset[i,k].append(u)
        self.pcset_dc = pcset
                
        suclist = [] # set of successfully allocated connections
        for u in self.traffic_pairs:
            if suc[u].x==1:
               suclist.append(u)
        self.suclist_dc = suclist
        
        core_usagex = {} # core allocation
        for u in self.traffic_pairs:
            if u in suclist:
                for i in u:
                    for k in self.cores:
                        if core_usage[u,i,k].x==1:
                            core_usagex[u,i] = k
                            break
            else:
                core_usagex[u,u[0]] = -1
                core_usagex[u,u[1]] = -1
        self.core_usagex = core_usagex
        
        self.connection_ub_ = len(suclist)
        self.throughput_ub_ = sum(self.tm[u[0],u[1]] for u in self.suclist_dc)
        self.obj_ub_ = self.alpha*self.connection_ub_+self.beta*self.throughput_ub_
                
    def create_model_sa(self, **kwargs):
        """Spectrum assignment ILP
        """

        smallM = self.num_slots
        bigM = 10*smallM
        
        # Model
        tic = time.clock()
        model_sa = Model('Arch2_sa')

        # binary variable: spectrum order
        spec_order = {}
        for i in self.pods:
            for k in self.cores:
                for c in itertools.combinations(self.pcset_dc[i,k],2):
                    spec_order[c[0],c[1]] = model_sa.addVar(vtype=GRB.BINARY)
        
        # continuous variable: first spectrum slot index
        # binary: fail?
        spec_idx = {}
        isfail = {}
        for u in self.suclist_dc:
            spec_idx[u] = model_sa.addVar(vtype=GRB.CONTINUOUS)
            isfail[u] = model_sa.addVar(vtype=GRB.BINARY, obj=self.alpha+self.beta*self.tm[u[0],u[1]])
            
        model_sa.update()
        
        # constraints: order
        for i in self.pods:
            for k in self.cores:
                for c in itertools.combinations(self.pcset_dc[i,k],2):
                    model_sa.addConstr(spec_idx[c[0]]+self.traffic_capacities[c[0]]-
                    spec_idx[c[1]]+bigM*spec_order[c[0],c[1]]<=bigM)
                    model_sa.addConstr(spec_idx[c[1]]+self.traffic_capacities[c[1]]-
                    spec_idx[c[0]]+bigM*(1-spec_order[c[0],c[1]])<=bigM)

        for u in self.suclist_dc:
            model_sa.addConstr(bigM*isfail[u]>=
            spec_idx[u]+self.traffic_capacities[u]-smallM)

        # params
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(model_sa.params, key, value)
                
        model_sa.optimize()
        toc = time.clock()
        
        self.model_sa = model_sa
        self.runtime_sa = toc-tic
        
        try:
            tmp = list(self.suclist_dc)
            for u in self.suclist_dc:
                if isfail[u].x == 1:
                    tmp.remove(u)
            self.suclist_sa = tmp
            
            self.spec_idxx = {} # spectrum slots allocation
            for u in self.traffic_pairs:
                if u in self.suclist_dc:
                    self.spec_idxx[u] = int(round(spec_idx[u].x))
                else:
                    self.spec_idxx[u] = -1
                    self.core_usagex[u,u[0]] = -1
                    self.core_usagex[u,u[1]] = -1                
                self.connections_lb = len(self.suclist_sa)
                self.throughput_lb = sum(self.tm[u[0],u[1]] for u in self.suclist_sa)
    
            # construct the resource tensor
            self.cnklist_sa = []
            tensor_milp = np.ones((self.num_pods, self.num_cores, self.num_slots), dtype=np.int8)
            for u in self.suclist_sa:
                src = u[0]
                dst = u[1]
                core_src = self.core_usagex[u,src]
                core_dst = self.core_usagex[u,dst]
                spec_idx = self.spec_idxx[u]
                spec_bd = self.traffic_capacities[u]
                tmp = [src, dst, spec_idx, spec_bd, core_src, core_dst, self.tm[u]]
                self.cnklist_sa.append(tmp)
                res_src = tensor_milp[src,core_src,spec_idx:(spec_idx+spec_bd)]
                res_dst = tensor_milp[dst,core_dst,spec_idx:(spec_idx+spec_bd)]
                if (sum(res_src)==spec_bd) and (sum(res_dst)==spec_bd):
                    res_src[:] = 0
                    res_dst[:] = 0
            self.tensor_milp = tensor_milp
            self.efficiency_milp = (float(sum(self.tm[i] for i in self.suclist_sa))/
                sum(self.traffic_capacities[i]*self.slot_capacity 
                for i in self.suclist_sa))
            self.obj_lb_ = self.alpha*self.connections_lb+self.beta*self.throughput_lb
        except:
            print 'No solution found'
            
    def write_result_csv(self, file_name, suclist):
        with open(file_name, 'w') as f:
            f.write('src,dst,spec,slots_used,core_src,core_dst,tfk_slot\n')
            for c in suclist:
                wstr = '{},{},{},{},{},{},{}\n'.format(c[0], c[1], c[2], 
                    c[3], c[4], c[5], c[6])
                f.write(wstr)

    def one_runs(self, a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        isone = np.concatenate(([0], np.equal(a, 1).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isone))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        ranges[:,1] = ranges[:,1]-ranges[:,0]
        return ranges  
                        
    def check(self, cnklist):
        """Check feasibility of solution
        """
        # check if any two connections are overlapped
        n_overlap=0
        for (u,v) in itertools.combinations(cnklist,2):
            if set(u)&set(v):
                cout_u = self.cnk_resource[u][2]
                cin_u = self.cnk_resource[u][3]
                si_u = self.cnk_resource[u][1]
                sb_u = self.traffic_capacities[u]

                cout_v = self.cnk_resource[v][2]
                cin_v = self.cnk_resource[v][3]
                si_v = self.cnk_resource[v][1]
                sb_v = self.traffic_capacities[v]
                
                if set([(u[0], cout_u), (u[1], cin_u)])&set([(v[0], cout_v), (v[1], cin_v)]):
                    if (si_u>=si_v and si_v+sb_v-1>=si_u) or (si_v>=si_u and si_u+sb_u-1>=si_v):
                        print [(u[0], cout_u), (u[1], cin_u)]
                        print [(v[0], cout_v), (v[1], cin_v)]
                        print 'wrong'
                        n_overlap+=1
        
        # check if any connection is out of range 
        n_oof = 0
        for u in self.cnk_group_suc:
            si = self.cnk_resource[u][1]
            sb = self.traffic_capacities[u]
            if si+sb-1>self.num_slots:
                n_oof+=1
        return (n_overlap, n_oof)
        
    def sa_heuristic(self, ascending1=True,ascending2=True):
        """Spectrum assignment heuristi
        ascending1: order of allocating connections in suclist
        ascending2: order of allocating connections in remain list
        """
        suclist = list(self.suclist_dc)
        suclist_tm = [self.traffic_capacities[u] for u in suclist]
        if ascending1:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
            
        IS_list = {} # independent set
        IS_list[0] = []
        cl_list = {}
        cl_list[0] = set()
        i = 0
        while len(suclist):
            tmplist = list(suclist)
            for u in tmplist:
                src = u[0]
                dst = u[1]
                src_core = self.core_usagex[u,src]
                dst_core = self.core_usagex[u,dst]
                if ((src,src_core) not in cl_list[i]) and ((dst, dst_core) not in cl_list[i]):
                    # add connection if it's independent to element in IS_list[i]
                    IS_list[i].append(u)
                    cl_list[i].add((src,src_core))
                    cl_list[i].add((dst,dst_core))
                    tmplist.remove(u)
            i += 1
            IS_list[i] = []
            cl_list[i] = set()
            suclist = tmplist
            
        del cl_list[i]
        del IS_list[i]

        self.obj_sah_ = 0
        self.obj_sah_connection_ = 0
        self.obj_sah_throughput_ = 0
        suclist = []
        cnklist = []
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots),dtype=np.int0)        
        for i in range(len(IS_list)):
            for u in IS_list[i]:
                src = u[0]
                dst = u[1]
                src_core = self.core_usagex[u,src]
                dst_core = self.core_usagex[u,dst]
                tmpsrc = restensor[src,src_core,:]
                tmpdst = restensor[dst,dst_core,:]
                tmp = tmpsrc*tmpdst
                tmpavail = self.one_runs(tmp)
                tmpidx = np.where(tmpavail[:,1]>=self.traffic_capacities[u])[0]
                if tmpidx.size:
                   spec_idx = tmpavail[tmpidx[0],0]
                   restensor[src,src_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
                   restensor[dst,dst_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
                   self.obj_sah_ += self.alpha+self.beta*self.tm[src,dst]
                   self.obj_sah_connection_ += 1
                   self.obj_sah_throughput_ += self.tm[src,dst]
                   tmp = [src, dst, spec_idx, self.traffic_capacities[u], 
                      src_core, dst_core, self.tm[u]]
                   cnklist.append(tmp)
                   suclist.append(u)

        remain_cnk = [u for u in self.traffic_pairs if u not in suclist]
        remain_tm = [self.traffic_capacities[u] for u in remain_cnk]
        if ascending2:
            remain_cnk = [x for (y,x) in sorted(zip(remain_tm,remain_cnk))]
        else:
            remain_cnk = [x for (y,x) in sorted(zip(remain_tm,remain_cnk), reverse=False)]
            
        for u in remain_cnk:
            src = u[0]
            dst = u[1]
            tmpsrc = restensor[src,:,:]
            tmpdst = restensor[dst,:,:]
            tmpcmb = np.zeros((self.num_cores**2, self.num_slots))
            k = 0
            avail_slots = {}
            for ksrc in self.cores:
                for kdst in self.cores:
                    tmpcmb[k,:] = tmpsrc[ksrc,:]*tmpdst[kdst,:]
                    tmpavail = self.one_runs(tmpcmb[k,:])
                    tmpidx = np.where(tmpavail[:,1]>=self.traffic_capacities[u])[0]
                    if not tmpidx.size:
                        avail_slots[ksrc,kdst] = np.array([-1, self.num_slots+1])
                    else:
                        idxm = np.argmin(tmpavail[tmpidx,1])
                        avail_slots[ksrc,kdst] = np.array(tmpavail[tmpidx[idxm],:])
                    k += 1
            avail_slots = list(sorted(avail_slots.iteritems(), key=lambda (x,y):y[1]))
            # avail_slots[0] has the form of ((core_out,core_in), [spec_idx,available_slots])
            if avail_slots[0][1][1]<=self.num_slots:
                src_core = avail_slots[0][0][0]
                dst_core = avail_slots[0][0][1]
                spec_idx = avail_slots[0][1][0]
                spec_bd = self.traffic_capacities[u]
                restensor[src,src_core,spec_idx:(spec_idx+spec_bd)] = 0
                restensor[dst,dst_core,spec_idx:(spec_idx+spec_bd)] = 0
                self.obj_sah_ += self.alpha+self.beta*self.tm[src,dst]
                self.obj_sah_connection_ += 1
                self.obj_sah_throughput_ += self.tm[src,dst]
                tmp = [src, dst, spec_idx, self.traffic_capacities[u], 
                   src_core, dst_core, self.tm[u]]
                cnklist.append(tmp)
        self.tensor_heuristic = restensor
#        self.efficiency_heuristic = (float(sum(self.tm[i] for i in suclist))/
#            sum(self.traffic_capacities[i]*self.slot_capacity 
#            for i in suclist))
        self.suclist_heuristic = cnklist
        
    def sa_heuristic_aff(self, ascending=True):
        """First fit with optimized core allocation
        """
        # ordering the connections
        suclist = list(self.suclist_dc)
        suclist_tm = [self.traffic_capacities[u] for u in suclist]
        if ascending:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
        
        # first fit
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots), dtype=np.int0)
        self.obj_affopt_ = 0
        self.obj_affopt_connection_ = 0
        self.obj_affopt_throughput_ = 0
        self.cnklist_affopt = [] # list of successfully allocated connections
        for i,u in enumerate(suclist):
            src = u[0]
            dst = u[1]
            src_core = self.core_usagex[u,src]
            dst_core = self.core_usagex[u,dst]
            tmpsrc = restensor[src,src_core,:]
            tmpdst = restensor[dst,dst_core,:]
            tmp = tmpsrc*tmpdst
            tmpavail = self.one_runs(tmp)
            tmpidx = np.where(tmpavail[:,1]>=self.traffic_capacities[u])[0]
            if tmpidx.size:
               spec_idx = tmpavail[tmpidx[0],0]
               restensor[src,src_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
               restensor[dst,dst_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
               self.obj_affopt_ += self.alpha+self.beta*self.tm[src,dst]
               self.obj_affopt_connection_ += 1
               self.obj_affopt_throughput_ += self.tm[src,dst]
               tmp = [src, dst, spec_idx, self.traffic_capacities[u], 
                      src_core, dst_core, self.tm[u]]
               self.cnklist_affopt.append(tmp)
        
    def heuristic(self):
        objbest = 0
        cnklist = []
        self.sa_heuristic(ascending1=True, ascending2=True)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.suclist_heuristic
        self.sa_heuristic(ascending1=True, ascending2=False)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.suclist_heuristic
        self.sa_heuristic(ascending1=False, ascending2=True)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.suclist_heuristic
        self.sa_heuristic(ascending1=False, ascending2=False)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.suclist_heuristic
            
        self.sa_heuristic_aff(ascending=True)
        if objbest < self.obj_affopt_:
            objbest = self.obj_affopt_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.cnklist_affopt
        self.sa_heuristic_aff(ascending=False)
        if objbest < self.obj_affopt_:
            objbest = self.obj_affopt_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.cnklist_affopt
            
        self.aff(ascending=True)
        if objbest < self.obj_aff_:
            objbest = self.obj_aff_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.cnklist_aff
        self.aff(ascending=False)
        if objbest < self.obj_aff_:
            objbest = self.obj_aff_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.cnklist_aff
        
        self.obj_heuristic_ = objbest
        self.obj_heuristic_connection_ = objcnk
        self.obj_heuristic_throughput_ = objthp
        self.cnklist_heuristic_ = cnklist
        
    def aff(self, ascending=True):
        """First fit according to the given connection list
        """
        suclist = list(self.traffic_pairs)
        suclist_tm = [self.traffic_capacities[u] for u in suclist]
        if ascending:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
        
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots), dtype=np.int0)
        self.obj_aff_ = 0
        self.obj_aff_connection_ = 0
        self.obj_aff_throughput_ = 0
        self.cnklist_aff = []
        for i,u in enumerate(suclist):
            src = u[0]
            dst = u[1]
            core_candidates = [(x,y) for x in self.cores for y in self.cores]
            for src_core, dst_core in core_candidates:
                tmpsrc = restensor[src,src_core,:]
                tmpdst = restensor[dst,dst_core,:]
                tmp = tmpsrc*tmpdst
                tmpavail = self.one_runs(tmp)
                tmpidx = np.where(tmpavail[:,1]>=self.traffic_capacities[u])[0]
                if tmpidx.size:
                   spec_idx = tmpavail[tmpidx[0],0]
                   restensor[src,src_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
                   restensor[dst,dst_core,spec_idx:(spec_idx+self.traffic_capacities[u])] = 0
                   self.obj_aff_ += self.alpha+self.beta*self.tm[src,dst]
                   self.obj_aff_connection_ += 1
                   self.obj_aff_throughput_ += self.tm[src,dst]
                   tmp = [src, dst, spec_idx, self.traffic_capacities[u], 
                          src_core, dst_core, self.tm[u]]
                   self.cnklist_aff.append(tmp)
                   break
                
    def save_tensor(self, tensor, filename):
        """Save resource tensor
        save as csv
        """
        tmp = tensor.reshape((-1, self.num_slots))
        np.savetxt(filename, tmp, fmt='%1d',delimiter=',')
        # for load the saved tensor
        # tmp = np.loadtxt(filename, delimiter=',')
        # tensor = tmp.reshape((self.num_pods, self.num_cores, self.num_slots))


class Arch3_decompose(object):
    """Create models for different SDM DCN architectures
    """
    def __init__(self, traffic_matrix, num_slots=320, num_cores=10,
                 slot_capacity =25, num_guard_slot=1, alpha=1, beta=0):
        """Initialize 
        """
        # traffic matrix
        self.traffic_matrix = traffic_matrix
        # number of PODs
        self.num_pods = traffic_matrix.shape[0]
        # capacity per spectrum slot, Gbps
        self.slot_capacity = slot_capacity
        # number of slot as guardband
        self.num_guard_slot = num_guard_slot
        # number of slots
        self.num_slots = num_slots
        # number of cores
        self.num_cores = num_cores
        # number of total demands
        self.total_demands = sum(self.traffic_matrix.flatten()>0)
        
        
        # Need to consider guardbands, no need to consider max capacity 
        # since a traffic can use the whole fiber
        self.tm = self.traffic_matrix.copy()        
        # Model data
        # set of pods
        pods = list(range(self.num_pods))
        # pairs of traffic demands
        traffic_pairs = tuplelist([(i, j) for i in pods for j in pods
                            if self.tm[i, j]>0])
        
        # Set of possible combinations of core and slot numbers
        core_set = {}
        slot_set = {}
        volu_set = {}
        for i, j in traffic_pairs:
            tmp = self.core_slot(self.tm[i, j])
            core_set[(i, j)] = tmp[:, 0]
            slot_set[(i, j)] = tmp[:, 1]
            volu_set[(i, j)] = tmp[:, 2]
        
        # set of cores
        cores = list(range(self.num_cores))
        
        self.pods = pods
        self.cores = cores
        self.core_set = core_set
        self.slot_set = slot_set
        self.volu_set = volu_set
        self.traffic_pairs = traffic_pairs
        # weight factor
        self.alpha = alpha
        self.beta = beta
        
    def volumn_model(self, **kwargs):
        """Estimate the volume of each connection, i.e., the combination of 
        core adn slot numbers.
        """
        # Model
        tic = time.clock()
        model_vol = Model('model_vol')
        
        # variable: choice of core-slot combination
        # variable: succuss?
        vol_choice = {}
        is_suc = {}
        vol_cnk = {}
        for u in self.traffic_pairs:
            is_suc[u] = model_vol.addVar(vtype=GRB.BINARY, obj=-1)
            vol_cnk[u] = model_vol.addVar(vtype=GRB.CONTINUOUS)
            for i in range(self.num_cores):
                vol_choice[u, i] = model_vol.addVar(vtype=GRB.BINARY, obj=-0.00001)
        
        # variable: volumn
        vol_limit = self.num_cores*self.num_slots
        vol_pod = {}
        for i in self.pods:
            vol_pod[i] = model_vol.addVar(vtype=GRB.CONTINUOUS, ub=vol_limit)
                
        model_vol.update()
        
        # constraints: success
        for u in self.traffic_pairs:
            model_vol.addConstr(quicksum(vol_choice[u, i] 
            for i in range(self.num_cores))==is_suc[u])
            model_vol.addConstr(quicksum(vol_choice[u, i]*self.volu_set[u][i]
            for i in range(self.num_cores))==vol_cnk[u])
                
        for i in self.pods:
            tmp = list((i, j) for (i, j) in self.traffic_pairs.select(i, '*'))
            tmp0 = list((j, i) for (j, i) in self.traffic_pairs.select('*', i))
            # all the traffics in link i
            tmp.extend(tmp0)
            model_vol.addConstr(quicksum(vol_cnk[u] for u in tmp)==vol_pod[i])
        
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(model_vol.params, key, value)
        
        model_vol.optimize()
        toc = time.clock()
        
        is_sucx = {}
        for u in self.traffic_pairs:
            is_sucx[u] = is_suc[u].x
        vol_choicex = {}
        for u in self.traffic_pairs:
            for i in range(self.num_cores):
                if(vol_choice[u,i].x==1):
                    vol_choicex[u] = i
        self.is_suc = is_sucx
        self.vol_choice = vol_choicex
        
    def core_slot(self, capacity):
        """Find all the possible combination of core and slot numbers for 
        a traffic demand with given capacity
        The guardband is considered
        
        Output: m * 2 numpy array, the first column is the number of cores, 
        and the second column is the number of slots, m is the number of 
        possible combinations.
        """
        # total number of slots
        n_slots = np.ceil(capacity / self.slot_capacity)
        # list of all combinations of core and slot numbers
        combination = [] 
        for i in range(1, self.num_cores+1):
            u = [i,int(np.ceil(n_slots/i)+self.num_guard_slot)]
            u.append(u[0]*u[1])
            combination.append(tuple(u))
        combination = np.asarray(combination)
                
        return combination
        
    def create_model_routing(self, **kwargs):
        channels_core = []
        group_core = {}
        tmp = 0
        B = np.empty((self.num_cores, 0))
        for n in range(1, self.num_cores+1):
            channels_core.extend(list(range(tmp, tmp+self.num_cores-n+1)))
            group_core[n] = list(range(tmp, tmp+self.num_cores-n+1))
            tmp = tmp+self.num_cores-n+1
            c = np.zeros((self.num_cores,))
            c[:n] = 1
            r = np.zeros((self.num_cores-n+1))
            r[0] = 1
            B = np.hstack((B, toeplitz(c,r)))
        self.B = B
        self.channels_core = channels_core
        self.group_core = group_core
        
        channels_core_nslot = {}
        for u in self.traffic_pairs:
            for n in range(1, self.num_cores+1):
                for i in group_core[n]:
                    channels_core_nslot[u,i] = self.slot_set[u][n-1]

        model_routing = Model('model_routing')
    
        core_choice = {}
        for u in self.traffic_pairs:
            for i in channels_core:
                core_choice[u,u[0],i] = model_routing.addVar(vtype=GRB.BINARY)
                core_choice[u,u[1],i] = model_routing.addVar(vtype=GRB.BINARY)
                
        is_suc = {}
        for u in self.traffic_pairs:
            is_suc[u] = model_routing.addVar(vtype=GRB.BINARY, obj=-(self.alpha+self.beta*self.tm[u[0],u[1]]))
            
        flow_core = {}
        for i in self.pods:
            for j in self.cores:
                flow_core[i,j] = model_routing.addVar(vtype=GRB.CONTINUOUS, ub=self.num_slots)
        
        model_routing.update()
        
        for u in self.traffic_pairs:
            model_routing.addConstr(quicksum(core_choice[u,u[0],i] 
            for i in channels_core)==is_suc[u])
            model_routing.addConstr(quicksum(core_choice[u,u[1],i] 
            for i in channels_core)==is_suc[u])
            #core channel consistent
            for n in range(1, self.num_cores+1):
                model_routing.addConstr(quicksum(core_choice[u,u[0],i] for i in group_core[n]) 
                == quicksum(core_choice[u,u[1],i] for i in group_core[n]))
        
        
        for i in self.pods:
            tmp = list((i, j) for (i, j) in self.traffic_pairs.select(i, '*'))
            tmp0 = list((j, i) for (j, i) in self.traffic_pairs.select('*', i))
            # all the traffics in link i
            tmp.extend(tmp0)
            for j in self.cores:
                model_routing.addConstr(quicksum(
                channels_core_nslot[u,k]*B[j,k]*core_choice[u,i,k]
                for k in channels_core
                for u in tmp)==flow_core[i,j])

        if len(kwargs):
            for key, value in kwargs.items():
                setattr(model_routing.params, key, value)
        
        model_routing.optimize()
        
        core_choicex = {} # which core channel
        nslot_choice = {} # number of spectral slots per core for connection u using channel i
        for u in self.traffic_pairs:
            if is_suc[u].x==1:
                for i in channels_core:
                    if core_choice[u,u[0],i].x==1:
                        core_choicex[u,u[0]] = i
                        nslot_choice[u] = channels_core_nslot[u,i]
                    if core_choice[u,u[1],i].x==1:
                        core_choicex[u,u[1]] = i
                        
        core_usagex = {}
        for u in self.traffic_pairs:
            if is_suc[u].x==1:
                chout = core_choicex[u,u[0]]
                chin = core_choicex[u,u[1]]
                core_out = np.where(B[:,chout]==1)[0]
                core_in = np.where(B[:,chin]==1)[0]
                core_usagex[u,u[0]] = core_out
                core_usagex[u,u[1]] = core_in
                        
        is_sucx = {}
        for u in self.traffic_pairs:
            is_sucx[u] = is_suc[u].x
            
        flow_corex = {}
        for i in self.pods:
            for j in self.cores:
                flow_corex[i,j] = flow_core[i,j].x

        cnk_in_core = {} # set of connections using a particular core
        for i in self.pods:
            tmp = list((i, j) for (i, j) in self.traffic_pairs.select(i, '*'))
            tmp0 = list((j, i) for (j, i) in self.traffic_pairs.select('*', i))
            # all the traffics in link i
            tmp.extend(tmp0)
            for j in self.cores:
                cnk_in_core[i,j] = []
                for u in tmp:
                    if sum(core_choice[u,i,k].x*B[j,k]for k in channels_core)==1:
                        cnk_in_core[i,j].append(u)
        
        suclist = []
        for u in self.traffic_pairs:
            if is_sucx[u]==1:
                suclist.append(u)
                
        self.core_choice = core_choicex
        self.core_usagex = core_usagex
        self.is_suc_routing = is_sucx
        self.flow_core = flow_corex
        self.cnk_in_core = cnk_in_core
        self.suclist = suclist
        self.nslot_choice = nslot_choice
        self.n_suc_routing = len(suclist)
        self.model_routing = model_routing
        
        self.connection_ub_ = len(self.suclist)
        self.throughput_ub_ = sum(self.tm[u[0],u[1]] for u in self.suclist)
        self.obj_ub_ = self.alpha*self.connection_ub_+self.beta*self.throughput_ub_

    def create_model_sa(self, **kwargs):
        smallM = self.num_slots
        bigM = 10*smallM
        
        model_sa = Model('model_sa')
        
        spec_order = {}
        for i in self.pods:
            for k in self.cores:
                for c in itertools.combinations(self.cnk_in_core[i,k],2):
                    spec_order[c[0],c[1]] = model_sa.addVar(vtype=GRB.BINARY)

        spec_idx = {}
        for u in self.suclist:
            spec_idx[u] = model_sa.addVar(vtype=GRB.CONTINUOUS)

        isfail = {}
        for u in self.suclist:
            isfail[u] = model_sa.addVar(vtype=GRB.BINARY, obj=self.alpha+self.beta*self.tm[u[0],u[1]])

        model_sa.update()

        for i in self.pods:
            for k in self.cores:
                for c in itertools.combinations(self.cnk_in_core[i,k],2):
                    model_sa.addConstr(
                    spec_idx[c[0]]+self.nslot_choice[c[0]]-spec_idx[c[1]]+
                    bigM*spec_order[c[0],c[1]]<=bigM)
                    model_sa.addConstr(
                    spec_idx[c[1]]+self.nslot_choice[c[1]]-spec_idx[c[0]]+
                    bigM*(1-spec_order[c[0],c[1]])<=bigM)

        for u in self.suclist:
            model_sa.addConstr(
            bigM*isfail[u]>=spec_idx[u]+self.nslot_choice[u]-smallM)
            
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(model_sa.params, key, value)
                
        model_sa.optimize()
        
        self.model_sa = model_sa
          
        tmp = list(self.suclist)
        for u in self.suclist:
            if isfail[u].x==1:
                tmp.remove(u)
        self.suclist_sa = list(tmp)
                
        self.spec_idxx = {}
        for u in self.suclist:
            self.spec_idxx[u] = spec_idx[u].x

        self.connection_lb_ = len(self.suclist_sa)           
        self.throughput_lb_ = sum(self.tm[u[0],u[1]] for u in self.suclist_sa)
        self.obj_lb_ = self.alpha*self.connection_lb_+self.beta*self.throughput_lb_
        
        # construct the resource tensor
        tensor_milp = np.ones((self.num_pods, self.num_cores, self.num_slots))
        for u in self.suclist_sa:
            src = u[0]
            dst = u[1]
            core_src = self.core_usagex[u,src]
            core_dst = self.core_usagex[u,dst]
            spec_idx = int(round(self.spec_idxx[u]))
            spec_bd = int(round(self.nslot_choice[u]))
            res_src = tensor_milp[src,core_src,spec_idx:(spec_idx+spec_bd)]
            res_dst = tensor_milp[dst,core_dst,spec_idx:(spec_idx+spec_bd)]
            if (np.sum(res_src)==spec_bd*core_src.size) and (np.sum(res_dst)==spec_bd*core_dst.size):
                tensor_milp[src,core_src,spec_idx:(spec_idx+spec_bd)] = 0
                tensor_milp[dst,core_dst,spec_idx:(spec_idx+spec_bd)] = 0
        self.tensor_milp = tensor_milp
        self.efficiency_milp = (float(sum(self.tm[i] for i in self.suclist_sa))/
            sum(self.nslot_choice[i]*self.core_usagex[i,i[0]].size*self.slot_capacity 
            for i in self.suclist_sa))
            
            
#    def write_result_csv(self, file_name, suclist):
#        with open(file_name, 'w') as f:
#            writer = csv.writer(f, delimiter=',')
#            writer.writerow(['src', 'dst', 'spec', 'core_src', 
#                             'core_dst', '#core', 'used_slot', 'tfk_slot'])
#            for u in suclist:
#                col_src = [self.B[j,self.core_choice[u,u[0]]] for j in self.cores]
#                core_src = self.one_runs(col_src)[0][0]
#                col_dst = [self.B[j,self.core_choice[u,u[1]]] for j in self.cores]
#                core_dst = self.one_runs(col_dst)[0][0]
#                num_cores = self.one_runs(col_dst)[0][1]
#                used_slot = self.nslot_choice[u]
#                tfk_slot = np.ceil(float(self.tm[u])/self.slot_capacity)
#                writer.writerow([u[0],u[1],
#                                 self.spec_idxx[u],core_src,core_dst,num_cores,
#                                 used_slot,tfk_slot])
            
    def write_result_csv(self, file_name, suclist):
        with open(file_name, 'w') as f:
            f.write('src,dst,spec,slots_used,core_src,core_dst,cores_used,tfk_slot\n')
            for c in suclist:
                wstr = '{},{},{},{},{},{},{},{}\n'.format(c[0], c[1], c[2], 
                    c[3], c[4], c[5], c[6], c[7])
                f.write(wstr)
        
    def sa_heuristic(self, ascending1=False, ascending2=True):
        """
        """
        suclist = list(self.suclist)
        suclist_tm = [self.nslot_choice[u] for u in suclist]
        if ascending1:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
            
        IS_list = {} # independent set
        IS_list[0] = []
        cl_list = {}
        cl_list[0] = set()
        i = 0
        while len(suclist):
            tmplist = list(suclist)
            for u in tmplist:
                src = u[0]
                dst = u[1]
                src_core = list(self.core_usagex[u,src])
                dst_core = list(self.core_usagex[u,dst])
                srct = set(zip([src]*len(src_core),src_core))
                dstt = set(zip([dst]*len(dst_core),dst_core))
                sdset = srct|dstt
                if len(sdset-cl_list[i])==len(sdset):
                    # add connection if it's independent to element in IS_list[i]
                    IS_list[i].append(u)
                    cl_list[i].update(sdset)
                    tmplist.remove(u)
            i += 1
            IS_list[i] = []
            cl_list[i] = set()
            suclist = tmplist
            
        del cl_list[i]
        del IS_list[i]
        
        self.obj_sah_ = 0
        self.obj_sah_connection_ = 0
        self.obj_sah_throughput_ = 0
        suclist = []
        self.cnklist_sah = []
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots))        
        for i in range(len(IS_list)):
            for u in IS_list[i]:
                src = u[0]
                dst = u[1]
                src_core = self.core_usagex[u,src]
                dst_core = self.core_usagex[u,dst]
                tmpsrc = np.prod(restensor[src,src_core,:],axis=0,dtype=bool)
                tmpdst = np.prod(restensor[dst,dst_core,:],axis=0,dtype=bool)
                tmp = tmpsrc*tmpdst
                tmpavail = self.one_runs(tmp)
                tmpidx = np.where(tmpavail[:,1]>=self.nslot_choice[u])[0]
                if tmpidx.size:
                   spec_idx = tmpavail[tmpidx[0],0]
                   restensor[src,src_core,spec_idx:(spec_idx+self.nslot_choice[u])] = False
                   restensor[dst,dst_core,spec_idx:(spec_idx+self.nslot_choice[u])] = False
                   self.obj_sah_ += self.alpha+self.beta*self.tm[src,dst]
                   self.obj_sah_connection_ += 1
                   self.obj_sah_throughput_ += self.tm[src,dst]
                   suclist.append(u)
                   tmp = [src, dst, spec_idx, self.nslot_choice[u], 
                          src_core[0], dst_core[0], len(src_core), self.tm[u]]
                   self.cnklist_sah.append(tmp)

        remain_cnk = [u for u in self.traffic_pairs if u not in suclist]
        remain_tm = [self.tm[u]/float(self.slot_capacity) for u in remain_cnk]
        if ascending2:
            remain_cnk = [x for (y,x) in sorted(zip(remain_tm,remain_cnk))]
        else:
            remain_cnk = [x for (y,x) in sorted(zip(remain_tm,remain_cnk), reverse=False)]
            
        for u in remain_cnk:
            src = u[0]
            dst = u[1]
            tmpsrc = restensor[src,:,:]
            tmpdst = restensor[dst,:,:]
            tmpcmb = np.zeros((self.num_cores**2, self.num_slots))
            k = 0
            avail_slots = {}
            for ksrc in self.cores:
                for kdst in self.cores:
                    tmpcmb[k,:] = tmpsrc[ksrc,:]*tmpdst[kdst,:]
                    tmpavail = self.one_runs(tmpcmb[k,:])
                    tmpidx = np.where(tmpavail[:,1]>=self.tm[u]*self.slot_capacity)[0]
                    if not tmpidx.size:
                        avail_slots[ksrc,kdst] = np.array([-1, self.num_slots+1])
                    else:
                        idxm = np.argmin(tmpavail[tmpidx,1])
                        avail_slots[ksrc,kdst] = np.array(tmpavail[tmpidx[idxm],:])
                    k += 1
            avail_slots = list(sorted(avail_slots.iteritems(), key=lambda (x,y):y[1]))
            # avail_slots[0] has the form of ((core_out,core_in), [spec_idx,available_slots])
            if avail_slots[0][1][1]<=self.num_slots:
                src_core = avail_slots[0][0][0]
                dst_core = avail_slots[0][0][1]
                spec_idx = avail_slots[0][1][0]
                spec_bd = int(np.ceil(self.tm[u]/float(self.slot_capacity))+self.num_guard_slot)
                restensor[src,src_core,spec_idx:(spec_idx+spec_bd)] = 0
                restensor[dst,dst_core,spec_idx:(spec_idx+spec_bd)] = 0
                self.obj_sah_ += self.alpha+self.beta*self.tm[src,dst]
                self.obj_sah_connection_ += 1
                self.obj_sah_throughput_ += self.tm[src,dst]
                tmp = [src, dst, spec_idx, spec_bd, 
                      src_core, dst_core, 1, self.tm[u]]
                self.cnklist_sah.append(tmp)
                
    def sa_heuristic_ff(self, ascending=True):
        """
        """
        suclist = list(self.suclist)
        suclist_tm = [self.nslot_choice[u] for u in suclist]
        if ascending:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
        
        self.obj_sahff_ = 0
        self.obj_sahff_connection_ = 0
        self.obj_sahff_throughput_ = 0
        self.cnklist_sahff = []
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots), dtype=bool)
        for u in suclist:
            src = u[0]
            dst = u[1]
            src_core = self.core_usagex[u,src]
            dst_core = self.core_usagex[u,dst]
            tmpsrc = np.prod(restensor[src,src_core,:],axis=0,dtype=bool)
            tmpdst = np.prod(restensor[dst,dst_core,:],axis=0,dtype=bool)
            tmp = tmpsrc*tmpdst
            tmpavail = self.one_runs(tmp)
            tmpidx = np.where(tmpavail[:,1]>=self.nslot_choice[u])[0]
            if tmpidx.size:
               spec_idx = tmpavail[tmpidx[0],0]
               restensor[src,src_core,spec_idx:(spec_idx+self.nslot_choice[u])] = False
               restensor[dst,dst_core,spec_idx:(spec_idx+self.nslot_choice[u])] = False
               self.obj_sahff_ += self.alpha+self.beta*self.tm[src,dst]
               self.obj_sahff_connection_ += 1
               self.obj_sahff_throughput_ += self.tm[src,dst]
               tmp = [src, dst, spec_idx, self.nslot_choice[u], 
                      src_core[0], dst_core[0], len(src_core), self.tm[u]]
               self.cnklist_sahff.append(tmp)
                
    def ff(self, ascending=True):
        """First fit 
        """
        suclist = list(self.traffic_pairs)
        suclist_tm = [self.tm[u] for u in suclist]
        if ascending:
            suclist = [x for (y,x) in sorted(zip(suclist_tm, suclist))]
        else:
            suclist = [x for (y, x) in sorted(zip(suclist_tm, suclist), reverse=True)]
            
        self.obj_ff_ = 0
        self.obj_ff_connection_ = 0
        self.obj_ff_throughput_ = 0
        self.cnklist_ff = []
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots))
        for u in suclist:
            src = u[0]
            dst = u[1]
            tfk = self.tm[u]
            for ncores in range(1, self.num_cores+1):
                spec_idx, nslots, src_core, dst_core, flag = \
                    self.try_channels(src, dst, tfk, restensor, ncores)
                if flag:
                    break
        
    def try_channels(self, src, dst, tfk, restensor, ncores):
        """Find hole for traffic with ncores
        """
        group_candidates = [(x,y) for x in self.group_core[ncores]
                                  for y in self.group_core[ncores]]
        flag = False
        for src_group, dst_group in group_candidates:
            src_cores = np.where(self.B[:,src_group])[0]
            dst_cores = np.where(self.B[:,dst_group])[0]
            tmp = restensor[src,src_cores[0],:]
            for i in src_cores:
                tmp = tmp*restensor[src,i,:]
            for i in dst_cores:
                tmp = tmp*restensor[dst,i,:]
            tmpavail = self.one_runs(tmp)
            nslots = self.slot_set[src,dst][ncores-1]
            tmpidx = np.where(tmpavail[:,1]>=nslots)[0]
            if tmpidx.size:
                spec_idx = tmpavail[tmpidx[0],0]
                restensor[src,src_cores[0]:(src_cores[0]+ncores),
                          spec_idx:(spec_idx+nslots)] = 0
                restensor[dst,dst_cores[0]:(dst_cores[0]+ncores),
                          spec_idx:(spec_idx+nslots)] = 0
                flag = True
                cnk = [src,dst,spec_idx,nslots,src_cores[0],dst_cores[0],
                       ncores,self.tm[src,dst]]
                self.cnklist_ff.append(cnk)
                self.obj_ff_ += self.alpha+self.beta*self.tm[src,dst]
                self.obj_ff_connection_ += 1
                self.obj_ff_throughput_ += self.tm[src,dst]
                return spec_idx, nslots, src_cores[0], dst_cores[0], flag
        return self.num_slots+1, 0, self.num_cores+1, self.num_cores+1, flag

    def one_runs(self, a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        isone = np.concatenate(([0], np.equal(a, 1).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isone))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        ranges[:,1] = ranges[:,1]-ranges[:,0]
        return ranges  
        
    def save_tensor(self, tensor, filename):
        """Save resource tensor
        save as csv
        """
        tmp = tensor.reshape((-1, self.num_slots))
        np.savetxt(filename, tmp, fmt='%1d',delimiter=',')
        
    def check_cnklist(self, cnklist):
        """Check the connection list
        """
        restensor = np.ones((self.num_pods, self.num_cores, self.num_slots))
        outage = []
        for c in cnklist:
            src = c[0]
            dst = c[1]
            spec_idx = c[2]
            nslots = c[3]
            src_core = c[4]
            dst_core = c[5]
            ncores = c[6]
            tfk = c[7]
            flag = True
            for i in range(src_core, src_core+ncores):
                if np.sum(restensor[src,i,spec_idx:(spec_idx+nslots)])<nslots:
                    flag = False
            for i in range(dst_core, dst_core+ncores):
                if np.sum(restensor[dst,i,spec_idx:(spec_idx+nslots)])<nslots:
                    flag = False
            if flag:
                restensor[src,src_core:(src_core+ncores),spec_idx:(spec_idx+nslots)] = 0
                restensor[dst,dst_core:(dst_core+ncores),spec_idx:(spec_idx+nslots)] = 0
            else:
                outage.append(c)
        print 'Number of conflict allocations: {}'.format(len(outage))
        return restensor
        
    def heuristic(self):
        objbest = 0
        objcnk = 0
        objthp = 0
        cnklist = []
        self.sa_heuristic(ascending1=True, ascending2=True)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.cnklist_sah
        self.sa_heuristic(ascending1=True, ascending2=False)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.cnklist_sah
        self.sa_heuristic(ascending1=False, ascending2=True)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.cnklist_sah
        self.sa_heuristic(ascending1=False, ascending2=False)
        if objbest < self.obj_sah_:
            objbest = self.obj_sah_
            objcnk = self.obj_sah_connection_
            objthp = self.obj_sah_throughput_
            cnklist = self.cnklist_sah
            
        self.sa_heuristic_ff(ascending=True)
        if objbest < self.obj_sahff_:
            objbest = self.obj_sahff_
            objcnk = self.obj_sahff_connection_
            objthp = self.obj_sahff_throughput_
            cnklist = self.cnklist_sahff
        self.sa_heuristic_ff(ascending=False)
        if objbest < self.obj_sahff_:
            objbest = self.obj_sahff_
            objcnk = self.obj_sahff_connection_
            objthp = self.obj_sahff_throughput_
            cnklist = self.cnklist_sahff
            
        self.ff(ascending=True)
        if objbest < self.obj_ff_:
            objbest = self.obj_ff_
            objcnk = self.obj_ff_connection_
            objthp = self.obj_ff_throughput_
            cnklist = self.cnklist_ff
        self.ff(ascending=False)
        if objbest < self.obj_ff_:
            objbest = self.obj_ff_
            objcnk = self.obj_ff_connection_
            objthp = self.obj_ff_throughput_
            cnklist = self.cnklist_ff
        
        self.obj_heuristic_ = objbest
        self.obj_heuristic_connection_ = objcnk
        self.obj_heuristic_throughput_ = objthp
        self.cnklist_heuristic_ = cnklist

        
if __name__=='__main__':
    np.random.seed(2014)
    #%% testing Traffic
#    np.random.seed(2016)
#    # generate a list of traffic matrices
#    traffic_dict = {}
##    traffic_list = []
#    num_pods=100
#    max_pod_connected=20
#    min_pod_connected=1
#    mean_capacity=320
#    variance_capacity=200
#    s = []
#    for i in range(20):
#        t = Traffic(num_pods=num_pods, max_pod_connected=max_pod_connected, 
#                    min_pod_connected=min_pod_connected, 
#                    mean_capacity=mean_capacity, 
#                    variance_capacity=variance_capacity,
#                    capacity_choices=np.arange(10,1210,10))
#        t.generate_traffic()
#        filename='traffic_matrix_new_%d.csv' % i
#        df = pd.DataFrame(t.traffic_matrix)
#        df.to_csv(filename, index=False, header=False)
#        s.append(np.sum(t.traffic_matrix))
#    print np.mean(s)
    
    #%% generate traffic
    num_pods=50
    max_pod_connected=300
    min_pod_connected=150
    mean_capacity=200
    variance_capacity=100
    num_cores=3
    num_slots=30
    t = Traffic(num_pods=num_pods, max_pod_connected=max_pod_connected, 
                min_pod_connected=min_pod_connected, 
                capacity_probs=[0.1875, 0.1875, 0.1875, 0.1876, 0.1250, 0.1250])
    t.generate_traffic()
    tm = t.traffic_matrix
    
    #%% read from file
#        tm = pd.read_csv('simu1_matrix_1.csv',skiprows=12,header=None)
#        tm.dropna(axis=1, how='any', inplace=True)
#        tm = tm.as_matrix()*25

    #%% arch1    
#    m = Arch1_decompose(tm, num_slots=num_slots, num_cores=num_cores, alpha=1, beta=0.0)
#    m.create_model_routing(mipfocus=1,timelimit=100,mipgap=0.1,method=2) # Method=2 or 3
#    m.heuristic()
#    m.write_result_csv('test.csv',m.cnklist_heuristic_)
#    print m.obj_ub_
#    print m.obj_heuristic_

    #%% arch2
#    m = Arch2_decompose(tm, num_slots=num_slots, num_cores=num_cores, alpha=1, beta=0.0)
#    m.create_model_routing(mipfocus=1,timelimit=10,mipgap=0.01,method=2) # Method=2 or 3
#    m.create_model_sa(mipfocus=1,timelimit=10,method=2,SubMIPNodes=2000,
#                      Heuristics=0.8)
#    m.heuristic()
#    m.write_result_csv('test.csv',m.cnklist_heuristic_)
#    print m.obj_ub_
#    print m.obj_heuristic_
    
    #%% arch3
#    m = Arch3_decompose(tm, num_slots=num_slots, num_cores=num_cores, alpha=1, beta=0.0)
#    m.create_model_routing(mipfocus=1, timelimit=20, method=2, mipgap=0.01)
#
#    m.heuristic()
#    a = m.check_cnklist(m.cnklist_heuristic_)
#    m.write_result_csv('test.csv', m.cnklist_heuristic_)
#    
#    print m.obj_heuristic_/m.obj_ub_
#    print m.obj_heuristic_
#    print m.obj_ub_