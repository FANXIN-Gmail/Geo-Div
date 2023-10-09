import numpy as np
import random
import math
import json
import collections
import sklearn.neighbors
import igraph

from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
import scipy


def calculate_user_cate_dis(category):

    poi_cate = collections.defaultdict(list)

    for line in category.readlines():
        pid, cateid = line.split()
        pid = int(pid)
        cateid = int(cateid)
        poi_cate[pid].append(cateid)
    
    return poi_cate

def dist(loc1, loc2):
    lat1, long1 = loc1[0], loc1[1]
    lat2, long2 = loc2[0], loc2[1]
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + math.cos(phi1)*math.cos(phi2))
    arc = math.acos(cos)
    earth_radius = 6371
    return arc * earth_radius


def calculate_candidate(training_user_set,num_):
    user_island= collections.defaultdict(set)
    for key in range(num_):
        neighborset=set()
        userset=training_user_set[key]
        for userindex in range(num_):
            candidateset=training_user_set[userindex]
            # print(candidateset)
            if(len(userset & candidateset)>0):
                # print(userindex)
                # print(userset & candidateset)
                temp=candidateset-userset
                neighborset.update(temp)
        # print(key)
        # print(len(neighborset))
        # neighborset=sorted(neighborset)
        user_island[key]=neighborset
    
    return user_island

def specific_candidate(user_first, user_candidate, cluster):

  limit_user_set = collections.defaultdict()
  
  for k,v in user_candidate.items():
    temp = set()
    for p in v:
      if user_first[k][cluster[p]]:
        temp.add(p)
    user_candidate[k] = temp

  return user_candidate

def specific_candidate_(user_first, user_candidate, cluster):

  limit_user_set = collections.defaultdict()
  
  for k,v in user_candidate.items():
    temp = set()
    for p in v:
      flag = True
      for i in cluster[p]:
        if not user_first[k][i]:
          flag = False
          break
      if flag:
        temp.add(p)
    user_candidate[k] = temp

  return user_candidate

# p 7727
# u 4716

# check_in = np.load("./Philadelphia/trainList.npy", allow_pickle=True).item()

# for k,v in check_in.items():

#   check_in[k] = collections.Counter(v)

# print(len(check_in))

poi = np.load("/data/fan_xin/Philadelphia/location.npy", allow_pickle=True).item()

# print(len(poi))

def poi_cluster(poi, K):
  
  sample = list(poi.values())
  
  # Prepare initial centers using K-Means++ method.
  initial_centers = kmeans_plusplus_initializer(sample, K).initialize()
  
  # create metric that will be used for clustering
  manhattan_metric = distance_metric(type_metric.USER_DEFINED, func=dist)
  
  # create instance of K-Means using specific distance metric:
  kmeans_instance = kmeans(sample, initial_centers, metric=manhattan_metric)
  
  # Run cluster analysis and obtain results.
  kmeans_instance.process()
  clusters = kmeans_instance.get_clusters()
  final_centers = kmeans_instance.get_centers()
  
  # Claculate the cluster
  cluster = collections.defaultdict(int)

  for k,v in poi.items():
    cluster[k] = kmeans_instance.predict([v])[0]
  
  kmeans_visualizer.show_clusters(sample, clusters, final_centers)
  
  return cluster


# cluster = poi_cluster(poi, 1000)

# np.save("./Philadelphia/clusterd.npy", cluster, allow_pickle=True)

# training_user_set = np.load("./Philadelphia/training_user_set.npy", allow_pickle=True).item()
# training_item_set = np.load("./Philadelphia/training_item_set.npy", allow_pickle=True).item()

# cluster = np.load('./Philadelphia/cluster.npy', allow_pickle=True).item()

# user_candidate = np.load('./Philadelphia/user_candidate.npy', allow_pickle=True).item()
# poi_candidate = np.load('./Philadelphia/item_candidate.npy', allow_pickle=True).item()

# Calculate the first layer's distribution
def first_layer_distribution(training_user_set, cluster, cluster_num):
  
  user_first_dis = collections.defaultdict(list)
  poi_first_dis = collections.defaultdict(list)
  
  for k,v in training_user_set.items():
    temp = [0]*cluster_num
    for p in v:
      temp[cluster[p]] += 1
    user_first_dis[k] = temp
  
  return user_first_dis

# Calculate the first layer's distribution
def first_layer_distribution_(training_user_set, cluster, cluster_num):
  
  user_first_dis = collections.defaultdict(list)
  poi_first_dis = collections.defaultdict(list)
  
  for k,v in training_user_set.items():
    temp = [0]*cluster_num
    for p in v:
      for i in cluster[p]:
        temp[i] += 1
    user_first_dis[k] = temp
  
  return user_first_dis

# Calculate the first layer's distribution
def third_layer_distribution(user_candidate, sample_num, cluster, cluster_num):
  
  user_third_dis = collections.defaultdict(list)
  poi_third_dis = collections.defaultdict(list)
  
  user_sample = collections.defaultdict(set)
  poi_sample = collections.defaultdict(set)

  for k,v in user_candidate.items():
    v = list(v)
    np.random.shuffle(v)
    if len(v) >= sample_num:
      v = v[:sample_num]
    user_sample[k] = set(v)
    temp = [0]*cluster_num
    for p in v:
      temp[cluster[p]] += 1
    user_third_dis[k] = temp
  
  return user_third_dis, user_sample

# Calculate the first layer's distribution
def third_layer_distribution_(user_candidate, sample_num, cluster, cluster_num):
  
  user_third_dis = collections.defaultdict(list)
  poi_third_dis = collections.defaultdict(list)
  
  user_sample = collections.defaultdict(set)
  poi_sample = collections.defaultdict(set)

  for k,v in user_candidate.items():
    v = list(v)
    np.random.shuffle(v)
    if len(v) >= sample_num:
      v = v[:sample_num]
    user_sample[k] = set(v)
    temp = [0]*cluster_num
    for p in v:
      for i in cluster[p]:
        temp[i] += 1
    user_third_dis[k] = temp
  
  return user_third_dis, user_sample

# Calculate the first layer's distribution
def third_layer_distribution_without_sample(user_candidate, cluster, cluster_num):
  
  user_third_dis = collections.defaultdict(list)

  for k,v in user_candidate.items():
    v = list(v)
    temp = [0]*cluster_num
    for p in v:
      for i in cluster[p]:
        temp[i] += 1
    user_third_dis[k] = temp
  
  return user_third_dis

def calculate_offset(user_first, user_third, user_num):

  collection = list()

  for i in range(user_num):
    collection.append(JS(user_first[i], user_third[i]))

  return sum(collection)/user_num


def JS(x,y):
  
  JS = 0.0
  px = np.array(x) / np.sum(x)
  py = np.array(y) / np.sum(y)
  M = (px+py) / 2

  return 1 - (0.5*scipy.stats.entropy(px,M)+0.5*scipy.stats.entropy(py, M))


