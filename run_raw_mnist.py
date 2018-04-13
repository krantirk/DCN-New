#-*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:56:33 2016

Perform experiment on Raw-MNIST data

@author: bo
"""

'''
5 epochs pretraining
-------------------------
Pre-training layer 0, epoch 0, cost  328.971387982
Pre-training layer 0, epoch 1, cost  194.048250862
Pre-training layer 0, epoch 2, cost  153.651779398
Pre-training layer 0, epoch 3, cost  234.896763756
Pre-training layer 0, epoch 4, cost  690.266988789
Pre-training layer 1, epoch 0, cost  50.5831995185
Pre-training layer 1, epoch 1, cost  15.7516139313
Pre-training layer 1, epoch 2, cost  10.1559799862
Pre-training layer 1, epoch 3, cost  7.34967242405
Pre-training layer 1, epoch 4, cost  3.44617866858
Pre-training layer 2, epoch 0, cost  79.7649487925
Pre-training layer 2, epoch 1, cost  43.9898411398
Pre-training layer 2, epoch 2, cost  34.6041835953
Pre-training layer 2, epoch 3, cost  28.572255711
Pre-training layer 2, epoch 4, cost  17.6474508977
Pre-training layer 3, epoch 0, cost  196.851036212
Pre-training layer 3, epoch 1, cost  159.880397363
Pre-training layer 3, epoch 2, cost  130.111530737
Pre-training layer 3, epoch 3, cost  106.309108441
Pre-training layer 3, epoch 4, cost  90.9784339374

Initial NMI for deep clustering: 0.42
ARI for deep clustering: 0.28
ACC for deep clustering: 0.44

... getting the finetuning functions
... finetunning the model
(array(1503.6624, dtype=float32), array(1390.3352, dtype=float32), array(113.32715, dtype=float32))
(array(1032.3591, dtype=float32), array(792.8752, dtype=float32), array(239.48395, dtype=float32))
(array(1048.8923, dtype=float32), array(831.78485, dtype=float32), array(217.10748, dtype=float32))
(array(1370.9973, dtype=float32), array(1226.062, dtype=float32), array(144.9353, dtype=float32))
(array(1110.9766, dtype=float32), array(996.3607, dtype=float32), array(114.615845, dtype=float32))
(array(951.3969, dtype=float32), array(831.7283, dtype=float32), array(119.66864, dtype=float32))
(array(957.41284, dtype=float32), array(864.34875, dtype=float32), array(93.06409, dtype=float32))
(array(996.3748, dtype=float32), array(898.0534, dtype=float32), array(98.32141, dtype=float32))
(array(826.7935, dtype=float32), array(736.7627, dtype=float32), array(90.03082, dtype=float32))
(array(886.5431, dtype=float32), array(801.99146, dtype=float32), array(84.551636, dtype=float32))
(array(804.96454, dtype=float32), array(719.05774, dtype=float32), array(85.9068, dtype=float32))
(array(849.8109, dtype=float32), array(765.025, dtype=float32), array(84.78589, dtype=float32))
(array(760.9889, dtype=float32), array(675.9032, dtype=float32), array(85.08569, dtype=float32))
(array(797.7848, dtype=float32), array(714.738, dtype=float32), array(83.046814, dtype=float32))
(array(821.29736, dtype=float32), array(741.56946, dtype=float32), array(79.727905, dtype=float32))
(array(764.7567, dtype=float32), array(683.5788, dtype=float32), array(81.17792, dtype=float32))
(array(704.8359, dtype=float32), array(634.7975, dtype=float32), array(70.03839, dtype=float32))
(array(830.27655, dtype=float32), array(753.60095, dtype=float32), array(76.6756, dtype=float32))
(array(761.67816, dtype=float32), array(686.0034, dtype=float32), array(75.67474, dtype=float32))
(array(728.99036, dtype=float32), array(660.29315, dtype=float32), array(68.697205, dtype=float32))
(array(717.1589, dtype=float32), array(648.18256, dtype=float32), array(68.97632, dtype=float32))
(array(736.42773, dtype=float32), array(667.0561, dtype=float32), array(69.37164, dtype=float32))
(array(711.9728, dtype=float32), array(645.6984, dtype=float32), array(66.27435, dtype=float32))
(array(754.14984, dtype=float32), array(680.20435, dtype=float32), array(73.945496, dtype=float32))
(array(724.22766, dtype=float32), array(653.21436, dtype=float32), array(71.013306, dtype=float32))
(array(692.9665, dtype=float32), array(627.4346, dtype=float32), array(65.53192, dtype=float32))
(array(658.2341, dtype=float32), array(598.55023, dtype=float32), array(59.683838, dtype=float32))
(array(712.6196, dtype=float32), array(647.6075, dtype=float32), array(65.012146, dtype=float32))
Fine-tuning epoch 1 ++++ 
Total cost: 862.95837, Reconstruction: 763.13885, Clustering: 99.81955, 
(array(686.9441, dtype=float32), array(622.1543, dtype=float32), array(64.789795, dtype=float32))
(array(706.77496, dtype=float32), array(642.0045, dtype=float32), array(64.77045, dtype=float32))
(array(693.50323, dtype=float32), array(609.9, dtype=float32), array(83.60321, dtype=float32))
(array(710.8529, dtype=float32), array(635.8094, dtype=float32), array(75.04352, dtype=float32))
(array(710.992, dtype=float32), array(645.8208, dtype=float32), array(65.1712, dtype=float32))
(array(692.5997, dtype=float32), array(591.80164, dtype=float32), array(100.798035, dtype=float32))
(array(685.48486, dtype=float32), array(626.5025, dtype=float32), array(58.98236, dtype=float32))
(array(672.39795, dtype=float32), array(612.5775, dtype=float32), array(59.820435, dtype=float32))
(array(648.088, dtype=float32), array(594.4166, dtype=float32), array(53.671387, dtype=float32))
(array(658.2468, dtype=float32), array(599.1042, dtype=float32), array(59.14264, dtype=float32))
(array(641.97327, dtype=float32), array(586.1108, dtype=float32), array(55.862488, dtype=float32))
(array(677.6938, dtype=float32), array(620.61523, dtype=float32), array(57.078552, dtype=float32))
(array(657.0131, dtype=float32), array(600.20184, dtype=float32), array(56.81128, dtype=float32))
(array(737.4339, dtype=float32), array(672.44617, dtype=float32), array(64.98773, dtype=float32))
(array(696.1926, dtype=float32), array(636.3362, dtype=float32), array(59.856445, dtype=float32))
(array(637.7678, dtype=float32), array(586.504, dtype=float32), array(51.263794, dtype=float32))
(array(594.59735, dtype=float32), array(545.06323, dtype=float32), array(49.53412, dtype=float32))
(array(672.4003, dtype=float32), array(616.42065, dtype=float32), array(55.979675, dtype=float32))
(array(676.2606, dtype=float32), array(619.5927, dtype=float32), array(56.667908, dtype=float32))
(array(646.54004, dtype=float32), array(592.2726, dtype=float32), array(54.267456, dtype=float32))
(array(631.7111, dtype=float32), array(582.62854, dtype=float32), array(49.08258, dtype=float32))
(array(606.0561, dtype=float32), array(555.473, dtype=float32), array(50.58307, dtype=float32))
(array(620.72424, dtype=float32), array(571.0371, dtype=float32), array(49.687134, dtype=float32))
(array(626.4394, dtype=float32), array(570.78455, dtype=float32), array(55.654846, dtype=float32))
(array(611.17645, dtype=float32), array(556.5317, dtype=float32), array(54.644775, dtype=float32))
(array(596.3481, dtype=float32), array(548.57996, dtype=float32), array(47.768127, dtype=float32))
(array(599.8174, dtype=float32), array(550.92285, dtype=float32), array(48.89453, dtype=float32))
(array(609.1853, dtype=float32), array(559.223, dtype=float32), array(49.96228, dtype=float32))
Fine-tuning epoch 2 ++++ 
Total cost: 663.56396, Reconstruction: 605.15210, Clustering: 58.41190, 
(array(603.39703, dtype=float32), array(552.08167, dtype=float32), array(51.31537, dtype=float32))
(array(640.5243, dtype=float32), array(590.946, dtype=float32), array(49.57831, dtype=float32))
(array(611.43994, dtype=float32), array(562.9405, dtype=float32), array(48.49945, dtype=float32))
(array(592.83875, dtype=float32), array(543.9907, dtype=float32), array(48.848022, dtype=float32))
(array(621.4412, dtype=float32), array(572.9292, dtype=float32), array(48.512024, dtype=float32))
(array(587.2985, dtype=float32), array(539.2148, dtype=float32), array(48.08374, dtype=float32))
(array(602.7759, dtype=float32), array(554.7866, dtype=float32), array(47.989258, dtype=float32))
(array(615.366, dtype=float32), array(564.4246, dtype=float32), array(50.941406, dtype=float32))
(array(595.2772, dtype=float32), array(551.94385, dtype=float32), array(43.333374, dtype=float32))
(array(604.07074, dtype=float32), array(557.74475, dtype=float32), array(46.32599, dtype=float32))
(array(591.7144, dtype=float32), array(546.6046, dtype=float32), array(45.109802, dtype=float32))
(array(616.7004, dtype=float32), array(568.89056, dtype=float32), array(47.809814, dtype=float32))
(array(575.019, dtype=float32), array(528.177, dtype=float32), array(46.84198, dtype=float32))
(array(606.98114, dtype=float32), array(562.2746, dtype=float32), array(44.706543, dtype=float32))
(array(626.097, dtype=float32), array(581.0657, dtype=float32), array(45.03131, dtype=float32))
(array(597.5357, dtype=float32), array(550.0282, dtype=float32), array(47.507507, dtype=float32))
(array(545.58026, dtype=float32), array(502.6407, dtype=float32), array(42.939575, dtype=float32))
(array(587.7683, dtype=float32), array(539.5936, dtype=float32), array(48.174683, dtype=float32))
(array(544.40405, dtype=float32), array(502.8368, dtype=float32), array(41.56726, dtype=float32))
(array(615.3715, dtype=float32), array(564.3585, dtype=float32), array(51.013, dtype=float32))
(array(578.8085, dtype=float32), array(536.48517, dtype=float32), array(42.323303, dtype=float32))
(array(553.2496, dtype=float32), array(509.41125, dtype=float32), array(43.838318, dtype=float32))
(array(626.7458, dtype=float32), array(580.52594, dtype=float32), array(46.21985, dtype=float32))
(array(558., dtype=float32), array(512.81354, dtype=float32), array(45.186462, dtype=float32))
(array(560.5221, dtype=float32), array(519.44214, dtype=float32), array(41.079956, dtype=float32))
(array(564.9126, dtype=float32), array(523.79456, dtype=float32), array(41.118042, dtype=float32))
(array(586.38275, dtype=float32), array(541.0541, dtype=float32), array(45.328674, dtype=float32))
(array(586.73004, dtype=float32), array(543.94604, dtype=float32), array(42.783997, dtype=float32))
Fine-tuning epoch 3 ++++ 
Total cost: 599.75732, Reconstruction: 553.07965, Clustering: 46.67770, 
(array(536.7921, dtype=float32), array(494.4214, dtype=float32), array(42.370728, dtype=float32))
(array(559.672, dtype=float32), array(517.52313, dtype=float32), array(42.148865, dtype=float32))
(array(559.3018, dtype=float32), array(516.88983, dtype=float32), array(42.411987, dtype=float32))
(array(561.61804, dtype=float32), array(518.27954, dtype=float32), array(43.3385, dtype=float32))
(array(577.626, dtype=float32), array(535.4188, dtype=float32), array(42.207153, dtype=float32))
(array(538.85126, dtype=float32), array(498.27866, dtype=float32), array(40.5726, dtype=float32))
(array(588.0805, dtype=float32), array(544.3475, dtype=float32), array(43.733032, dtype=float32))
(array(542.2424, dtype=float32), array(501.2227, dtype=float32), array(41.019684, dtype=float32))
(array(555.8797, dtype=float32), array(517.6712, dtype=float32), array(38.208496, dtype=float32))
(array(557.24854, dtype=float32), array(517.0936, dtype=float32), array(40.154907, dtype=float32))
(array(563.4879, dtype=float32), array(521.6405, dtype=float32), array(41.847412, dtype=float32))
(array(596.97736, dtype=float32), array(554.3419, dtype=float32), array(42.635437, dtype=float32))
(array(535.4089, dtype=float32), array(493.59265, dtype=float32), array(41.816223, dtype=float32))
(array(580.9593, dtype=float32), array(540.34546, dtype=float32), array(40.61383, dtype=float32))
(array(583.5964, dtype=float32), array(545.1045, dtype=float32), array(38.491882, dtype=float32))
(array(561.20184, dtype=float32), array(521.4452, dtype=float32), array(39.756653, dtype=float32))
(array(522.4379, dtype=float32), array(484.3303, dtype=float32), array(38.107635, dtype=float32))
(array(546.7058, dtype=float32), array(506.2769, dtype=float32), array(40.428925, dtype=float32))
(array(529.1922, dtype=float32), array(491.58093, dtype=float32), array(37.611267, dtype=float32))
(array(555.7088, dtype=float32), array(516.01855, dtype=float32), array(39.690247, dtype=float32))
(array(547.0273, dtype=float32), array(508.50653, dtype=float32), array(38.520752, dtype=float32))
(array(538.823, dtype=float32), array(499.00388, dtype=float32), array(39.819122, dtype=float32))
(array(575.75507, dtype=float32), array(535.1897, dtype=float32), array(40.56537, dtype=float32))
(array(504.80548, dtype=float32), array(466.6012, dtype=float32), array(38.204285, dtype=float32))
(array(527.34235, dtype=float32), array(490.10428, dtype=float32), array(37.238068, dtype=float32))
(array(579.7061, dtype=float32), array(537.7218, dtype=float32), array(41.984314, dtype=float32))
(array(568.0111, dtype=float32), array(527.58905, dtype=float32), array(40.42206, dtype=float32))
(array(531.3808, dtype=float32), array(495.15076, dtype=float32), array(36.23004, dtype=float32))
Fine-tuning epoch 4 ++++ 
Total cost: 561.90515, Reconstruction: 521.06641, Clustering: 40.83880, 
(array(515.67017, dtype=float32), array(478.28073, dtype=float32), array(37.389435, dtype=float32))
(array(536.44116, dtype=float32), array(499.1361, dtype=float32), array(37.305054, dtype=float32))
(array(564.83295, dtype=float32), array(522.97064, dtype=float32), array(41.862305, dtype=float32))
(array(558.04486, dtype=float32), array(519.4717, dtype=float32), array(38.57318, dtype=float32))
(array(561.7917, dtype=float32), array(525.244, dtype=float32), array(36.54767, dtype=float32))
(array(516.8488, dtype=float32), array(481.40247, dtype=float32), array(35.44635, dtype=float32))
(array(546.539, dtype=float32), array(508.03268, dtype=float32), array(38.506317, dtype=float32))
(array(520.1164, dtype=float32), array(483.38882, dtype=float32), array(36.72757, dtype=float32))
(array(531.1843, dtype=float32), array(495.96945, dtype=float32), array(35.214874, dtype=float32))
(array(528.8051, dtype=float32), array(492.54865, dtype=float32), array(36.25647, dtype=float32))
(array(522.5777, dtype=float32), array(485.77725, dtype=float32), array(36.800446, dtype=float32))
(array(568.5186, dtype=float32), array(530.6165, dtype=float32), array(37.9021, dtype=float32))
(array(524.0513, dtype=float32), array(487.3862, dtype=float32), array(36.66507, dtype=float32))
(array(611.9945, dtype=float32), array(571.7588, dtype=float32), array(40.235718, dtype=float32))
(array(567.28174, dtype=float32), array(530.7144, dtype=float32), array(36.56732, dtype=float32))
(array(522.2925, dtype=float32), array(485.5928, dtype=float32), array(36.699677, dtype=float32))
(array(517.3168, dtype=float32), array(481.61707, dtype=float32), array(35.699707, dtype=float32))
(array(523.98114, dtype=float32), array(487.26846, dtype=float32), array(36.712677, dtype=float32))
(array(505.99582, dtype=float32), array(470.60852, dtype=float32), array(35.3873, dtype=float32))
(array(524.1086, dtype=float32), array(488.39127, dtype=float32), array(35.717316, dtype=float32))
(array(536.65546, dtype=float32), array(502.5026, dtype=float32), array(34.152863, dtype=float32))
(array(512.112, dtype=float32), array(476.92838, dtype=float32), array(35.183624, dtype=float32))
(array(533.50665, dtype=float32), array(499.04315, dtype=float32), array(34.4635, dtype=float32))
(array(489.53772, dtype=float32), array(455.2015, dtype=float32), array(34.336212, dtype=float32))
(array(511.98962, dtype=float32), array(477.73358, dtype=float32), array(34.256042, dtype=float32))
(array(528.6193, dtype=float32), array(492.76358, dtype=float32), array(35.855743, dtype=float32))
(array(528.62524, dtype=float32), array(491.7844, dtype=float32), array(36.84085, dtype=float32))
(array(500.61172, dtype=float32), array(466.44147, dtype=float32), array(34.170258, dtype=float32))
Fine-tuning epoch 5 ++++ 
Total cost: 538.01385, Reconstruction: 500.99353, Clustering: 37.02036, 
NMI for deep clustering: 0.48
ARI for deep clustering: 0.30
ACC for deep clustering: 0.45
The training code for file multi_layer_km.py ran for 70.23m
[0.45939693 0.24487811 0.41533595 0.         0.         0.
'''


import gzip
import cPickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from multi_layer_km import test_SdC
from cluster_acc import acc
  
K = 10
trials = 1

filename = 'mnist_dcn.pkl.gz'
path = './data/'
dataset = path+filename



# perform DCN

#  need to train with 250 epochs of layerwise, and 250 epochs of end-end SAE
#  to get the initialization file with the following setting, takes a while

# config = {'Init': '',
#           'lbd':  1,  # reconstruction
#           'beta': 0,
#           'output_dir': 'MNIST_results',
#           'save_file': 'mnist_pre.pkl.gz',
#           'pretraining_epochs': 250,
#           'pretrain_lr_base': 0.0001,
#           'mu': 0.9,
#           'finetune_lr': 0.0001,
#           'training_epochs': 250,
#           'dataset': dataset,
#           'batch_size': 256,
#           'nClass': K,
#           'hidden_dim': [500, 500, 2000, 10],
#           'diminishing': False}

config = {'Init': 'deepclus_10_pretrain.pkl.gz', # 'mnist_pre.pkl.gz'
          'lbd':  1,  # reconstruction
          'beta': 1,
          'output_dir': 'MNIST_results',
          'save_file': 'mnist_10.pkl.gz',
          'pretraining_epochs': 5,  # 250
          'pretrain_lr_base': 0.0001,
          'mu': 0.9,
          'finetune_lr': 0.0001,
          'training_epochs': 5,  # 50
          'dataset': dataset,
          'batch_size': 256,
          'nClass': K,
          'hidden_dim': [500, 500, 2000, 10],
          'diminishing': False}

results = []
for i in range(trials):
    res_metrics = test_SdC(**config)
    results.append(res_metrics)

results_SAEKM = np.zeros((trials, 3))
results_DCN = np.zeros((trials, 3))

N = config['training_epochs']/5
for i in range(trials):
    results_SAEKM[i] = results[i][0]
    results_DCN[i] = results[i][N]
SAEKM_mean = np.mean(results_SAEKM, axis=0)
SAEKM_std = np.std(results_SAEKM, axis=0)
DCN_mean = np.mean(results_DCN, axis=0)
DCN_std = np.std(results_DCN, axis=0)

results_all = np.concatenate((DCN_mean, DCN_std, SAEKM_mean, SAEKM_std),
                             axis=0)
print(results_all)
np.savetxt('mnist_results.txt', results_all, fmt='%.3f')

