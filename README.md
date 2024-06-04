CUDA_VISIBLE_DEVICES=1 python train.py --run_num 49 --server_path "../../../mnt/nfs/"
CUDA_VISIBLE_DEVICES=1 python train.py --run_num -1 --server_path "../../../nfs3-p2/"

原来的是训练5*20*1*2，测试1*100*1*2
√ log1,2,3,4,5: 训练5*20*1*2，测试1*100*1*2，但是五个高度分别训练5个模型，1km,256
√ log6,7,8,9,10: 训练5*20*1*2，测试1*100*1*2，但是五个高度分别训练5个模型，2km,512
√ log11: 测试多专家，1km，平均值
√ log12: 测试多专家，1km，随机值

√ 40：log13,14,15,16,17: 训练5*20*1*2，测试1*100*1*2，但是五个高度分别训练5个模型，3km,768
√ log18: 测试多专家，2km，平均值
√ log19: 测试多专家，2km，随机值

√ ~~log20: 不同路径随机采用不同的高度，100-500m，1km~~
√ log21: 路径内不同图片采用不同的高度，100-500m，1km
√ ~~log22: 不同路径随机采用不同的高度，100-500m，2km~~
√ 204 ppo5 log23: 路径内不同图片采用不同的高度，100-500m，2km

√ 204 ppo log24: moe，路径内不同图片采用不同的高度，100-500m，1km, randn, k=1
√ 204 ppo2 log25: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1
慢√ 204 ppo3 log26: moe，路径内不同图片采用不同的高度，100-500m，1km, randn, k=2
√ 204 ppo4 log27: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 large特征计算gate
***√ 204 ppo6 log28: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
√ 40 ppo log29: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=2, mobilenetv3 small特征计算gate, 化简vit
√ 40 ppo2 log30: moe，路径内不同图片采用不同的高度，100-500m，1km, randn, k=2, mobilenetv3 small特征计算gate, 化简vit

√ 204 ppo log31: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
√ 204 ppo2 log32: moe，路径内不同图片采用不同的高度，100-500m，2km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
√ 40 ppo3 log33: moe，路径内不同图片采用不同的高度，100-500m，3km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
~~√ 40 ppo4 log34: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit~~
~~204 ppo4 log35: moe，路径内不同图片采用不同的高度，100-500m，2km多终点, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit~~
~~204 ppo3 log36: moe，路径内不同图片采用不同的高度，100-500m，3km多终点, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit, 加载28权重~~
√ 204 ppo log37: moe，路径内不同图片采用不同的高度，100-500m，2km多终点, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit

比较AVG STEP NUM:
√ 204 ppo2 log38: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
diff reward

***√ 204 ppo3 log39: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
diff+angle reward

√ 204 ppo4 log40: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
diff+step reward

√ 204 ppo5 log41: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
diff+angle+step reward

√ 40 ppo log42 diff
√ 40 ppo2 log43 diff+angle
√ 40 ppo4 log44 diff+step
√ 40 ppo5 log45 diff+angle+step

***√ 204 ppo2 log46: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
diff+angle moeloss
慢√ 204 ppo3 log47: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
diff+angle moeloss moeffn
***√ 204 ppo4 log48: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
diff+angle moeloss moeffn ffnloss

~~204 ppo log49: moe，路径内不同图片采用不同的高度，100-500m，2km, 40+200, zeros, k=1, mobilenetv3 small特征计算gate, 化简vit
diff+angle moeloss moeffn ffnloss~~

40 ppo dim32
40 ppo2 dim64
40 ppo3 depth2
40 ppo4 depth6
40 ppo5 head1
40 ppo6 head4
退回到40 ppo log29: moe，路径内不同图片采用不同的高度，100-500m，1km, zeros, k=2, mobilenetv3 small特征计算gate, 化简vit
