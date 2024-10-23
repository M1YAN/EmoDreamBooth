import os
import shutil
import torch
import torch.nn as nn
from PIL import Image
import clip
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 定义批量处理的数据集类
class EmotionDataset(Dataset):
    def __init__(self, data_root, emotion, preprocess):
        self.preprocess = preprocess
        self.image_paths = []
        self.emotion = emotion

        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(root, file))
        self._length = len(self.image_paths)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image)
        return image_input, image_path

# 情感分类器定义
class emo_classifier(nn.Module):
    def __init__(self):
        super(emo_classifier, self).__init__()
        self.fc = nn.Linear(768, 8)  # 输入为 768 维，输出为 8 个情感类别

    def forward(self, x):
        x = self.fc(x)
        return x
    
# 处理聚类的函数，结合CLIP和情感分类器进行筛选
def process_clusters_in_batches(cluster_dir, emotion, classifier, model, preprocess, dest_dir, batch_size=64, min_images=5, classifier_threshold=0.5):
    dataset = EmotionDataset(cluster_dir, emotion, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    valid_images = []
    total_images = 0

    label2idx = {
      "amusement": 0,
      "awe": 1,
      "contentment": 2,
      "excitement": 3,
      "anger": 4,
      "disgust": 5,
      "fear": 6,
      "sadness": 7
    }

    with torch.no_grad():
        for batch in dataloader:
            images, image_paths = batch
            images = images.to("cuda")

            # 使用CLIP提取图像特征 (768维，适用于ViT-L/14)
            image_features = model.encode_image(images).float().detach().cpu()

            # 使用情感分类器对图像进行分类
            logits = classifier(image_features)
            probs = torch.softmax(logits, dim=1)

            # 判断每张图像是否符合分类器的阈值
            for i in range(images.size(0)):
                image_path = image_paths[i]
                total_images += 1

                # 获取最大类别的概率
                max_prob, predicted_class = torch.max(probs[i], dim=0)

                # 如果概率大于阈值，则保留图像
                if max_prob.item() >= classifier_threshold and predicted_class.item() == label2idx[emotion]:
                    valid_images.append(image_path)

    # 如果有效图像数大于等于 min_images，则复制聚类文件夹到目标目录
    if len(valid_images) >= min_images:
        cluster_name = os.path.basename(cluster_dir)  # 获取聚类文件夹名称
        dest_cluster_dir = os.path.join(dest_dir, emotion, cluster_name)
        os.makedirs(dest_cluster_dir, exist_ok=True)

        # 复制有效的图像到目标目录
        for image_path in valid_images:
            shutil.copy(image_path, dest_cluster_dir)
        print(f"Cluster {cluster_name} in emotion {emotion} has {len(valid_images)} valid images and has been copied.")
    else:
        print(f"Cluster {os.path.basename(cluster_dir)} in emotion {emotion} does not meet the requirements.")

# 主函数：处理所有聚类
def process_all_clusters(root_dir, dest_dir, classifier, batch_size=64, min_images=5, classifier_threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载ViT-L/14 CLIP模型和预处理器
    model, preprocess = clip.load("ViT-L/14", device=device)

    for emotion in os.listdir(root_dir):
        emotion_path = os.path.join(root_dir, emotion)

        if not os.path.isdir(emotion_path):
            continue

        for cluster in os.listdir(emotion_path):
            if cluster == 'cluster_-1':
                continue

            cluster_dir = os.path.join(emotion_path, cluster)

            process_clusters_in_batches(cluster_dir, emotion, classifier, model, preprocess, dest_dir, batch_size=batch_size, min_images=min_images, classifier_threshold=classifier_threshold)

if __name__ == "__main__":
    # 加载训练好的情感分类器
    classifier = emo_classifier()
    state = torch.load("weights/time_2023-11-12_03-29-best.pth", map_location='cpu')
    classifier.load_state_dict(state)
    classifier.eval()

    data_root = "/openbayes/input/input0/EmoSet-118K/cluster_result"  
    dest_root = "/openbayes/input/input0/EmoSet-118K/filtered_cluster_result_classfier0.7"

    process_all_clusters(data_root, dest_root, classifier, batch_size=64, min_images=3, classifier_threshold=0.75)