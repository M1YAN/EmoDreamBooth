import torch
import clip
from PIL import Image
from sklearn.cluster import DBSCAN
import numpy as np
import os
from tqdm import tqdm
import shutil
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pickle

# 加载 CLIP 模型和预处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 准备数据集
class EmotionDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        image = self.preprocess(image)

        return image, img_path

def create_dataloader(image_paths, preprocess, batch_size=64, num_workers=4):
    dataset = EmotionDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader

def extract_clip_features(dataloader, model):
    features = []
    model.eval()

    with torch.no_grad():
        for batch_images, _ in tqdm(dataloader):
            batch_images = batch_images.to("cuda")
            batch_features = model.encode_image(batch_images).cpu().numpy()
            features.append(batch_features)
    
    return np.vstack(features)

# 保存category_features到磁盘
def save_category_features(category_features, save_dir):
    with open(os.path.join(save_dir, 'category_features.pkl'), 'wb') as f:
        pickle.dump(category_features, f)

# 加载已经保存的category_features
def load_category_features(save_dir):
    with open(os.path.join(save_dir, 'category_features.pkl'), 'rb') as f:
        category_features = pickle.load(f)
    return category_features

# 使用 DBSCAN 进行聚类
def dbscan_clustering_cosine(features, eps=0.5, min_samples=5):
    """
    使用 sklearn 的 DBSCAN 进行基于余弦距离的密度聚类
    :param features: (N, D) 形状的特征矩阵
    :param eps: DBSCAN 中的最大距离阈值（相似度阈值），值越小，聚类越细致
    :param min_samples: 定义一个聚类的最小样本数
    :return: 每个样本的聚类标签
    """
    # 使用 DBSCAN 并指定使用余弦距离进行度量
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    
    # 进行聚类
    cluster_labels = db.fit_predict(features)
    
    return cluster_labels

def load_image_paths(data_dir, categories):
    image_paths = {}
    for category in categories:
        category_dir = os.path.join(data_dir,category)
        image_paths[category] = [os.path.join(category_dir, img) for img in os.listdir(category_dir)]
    return image_paths

def cluster_each_category_with_dbscan(category_features, emotion_categories, eps=0.5, min_samples=5):
    category_clusters = {}
    for category in emotion_categories:
        print(f"Processing category: {category}")

        feature = category_features[category]
        
        cluster_labels = dbscan_clustering_cosine(feature, eps=eps, min_samples=min_samples)

        category_clusters[category] = cluster_labels
    
    return category_clusters

def save_clusters_by_labels(image_paths, category_clusters, save_dir):
    """
    根据每个类别的聚类标签保存图像
    :param image_paths: 每个类别的图像路径字典
    :param category_clusters: 每个类别的聚类结果
    :param save_dir: 保存结果的目录
    """
    # 创建目标文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 处理每个情感类别的图像
    for category, cluster_labels in category_clusters.items():
        print(f"Saving clusters for category: {category}")
        clusters = defaultdict(list)
        
        # 将图像路径和聚类标签对应起来
        for img_path, label in zip(image_paths[category], cluster_labels):
            clusters[label].append(img_path)
        
        # 创建每个聚类的文件夹并保存图像
        for cluster_id, img_paths in clusters.items():
            cluster_dir = os.path.join(save_dir, category, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)
            for img_path in img_paths:
                shutil.copy(img_path, cluster_dir)


# 提取特征
data_dir = "/openbayes/input/input0/EmoSet-118K/image"
emotion_categories = ['amusement','anger','awe','contentment','disgust','excitement','fear','sadness']
image_paths = load_image_paths(data_dir, emotion_categories)
save_dir = "/openbayes/input/input0/EmoSet-118K/feature"
    
if os.path.exists(os.path.join(save_dir, 'category_features.pkl')):
    print("Loading saved features...")
    category_features = load_category_features(save_dir)
else:    
    category_features = {}
    for category in emotion_categories:
        print(f"Extracting features for {category}...")
        dataloader = create_dataloader(image_paths[category], preprocess)
        category_features[category] = extract_clip_features(dataloader, model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_category_features(category_features, save_dir)
    print("Features saved.")

# 使用 DBSCAN 进行聚类（余弦距离度量）
category_clusters  = cluster_each_category_with_dbscan(category_features, emotion_categories, eps=0.11, min_samples=5)

# 打印聚类结果
print(category_clusters)

cluster_path = "/openbayes/input/input0/EmoSet-118K/cluster_result"

save_clusters_by_labels(image_paths, category_clusters, cluster_path)