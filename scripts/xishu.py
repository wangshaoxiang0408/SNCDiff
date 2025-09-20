import numpy as np
from sklearn.decomposition import SparseCoder
 
# 假设我们有以下10个样本，每个样本有20个特征
X = np.random.rand(100, 9)
 
# 创建一个大小为(20, 9)的随机字典
dictionary_size = (X.shape[1], 9)  # 字典大小为(特征数, 基向量数)
dictionary = np.random.randn(*dictionary_size)
 
# 创建稀疏编码器对象，指定字典和编码方法（这里使用OMP）
sparse_coder = SparseCoder(dictionary=dictionary, transform_algorithm='omp', transform_n_nonzero_coefs=5)
 
# 对输入数据X进行编码
sparse_codes = sparse_coder.transform(X)
 
print("Original input data (samples x features):")
print(X[:2])  # 打印前两个样本
 
print("\nSparse codes (samples x basis vectors):")
print(sparse_codes[:2])  # 打印前两个样本的稀疏编码