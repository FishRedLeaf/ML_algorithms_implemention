
# 参考：https://github.com/stuntgoat/kmeans
# 在原代码基础上修改为numpy实现

from collections import defaultdict
import numpy as np


def update_centers(dataset, assignments):
    """
    params:
    :dataset: ndarray, dim=(num_samples, num_features)
    :assignments: list, len=num_samples, 表示每个sample对应的簇id
    return:
    :centers: 
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, dataset):
        new_means[assignment].append(point)
        
    for points in new_means.values():
        # 计算points的均值向量作为这些points的center，即簇中心
        centers.append(np.average(points, axis=0))
    return np.array(centers)


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        assignments.append(get_point_cluster(point, centers))
    return assignments


def get_point_cluster(point, centers):
    """
    :point: ndarray, dim=(num_features)
    :centers: ndarray, dim=(k, num_features)
    function:
    返回point所属的簇id
    """
    k = len(centers)
    points = np.tile(point, k).reshape(k, -1)
    L2 = np.linalg.norm(points-centers, axis=1)
    return np.argmax(L2)


def generate_k(dataset, k):
    """
    :dataset: ndarray, num_samples * num_features
    :k: num of clusters

    function:
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    mins = np.min(dataset, axis=0)
    maxs = np.max(dataset, axis=0)
    # print('mins, maxs shape: ', mins.shape, maxs.shape)
    for i in range(len(mins)):
        centers.append(np.random.uniform(mins[i], maxs[i], (1, k)))
    return np.concatenate(centers).transpose(1, 0)


def k_means(dataset, k):
    '''
    :dataset: list, num_samples * num_features
    :k: num of clusters
    '''
    # 从数据集中随机抽取k个样本作为初始均值向量, k*num_features
    k_points = generate_k(dataset, k)
    # print('k_points: ', k_points)
    # assignments表示每个样本对应的簇id, 0 ~ k-1
    # 根据生成的初始均值向量去更新每个样本点所属的簇
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    max_iters = 1000
    while assignments != old_assignments and max_iters != 0:
        # 计算新的表示簇中心的均值向量
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        # 根据新的簇中心去更新每个样本点所属的簇
        assignments = assign_points(dataset, new_centers)
        max_iters -= 1
    return zip(assignments, dataset)


num_samples = 50
num_features = 3
k = 3
dataset = np.random.randint(1, 100, (num_samples, num_features))
print(list(k_means(dataset, k)))
