import os
import shutil
import torch
from PIL import Image
import clip
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 定义批量处理的数据集类
class EmotionDataset(Dataset):
    def __init__(self, data_root, emotion, preprocess):
        self.preprocess = preprocess
        self.image_paths = []
        self.emotion = emotion
        self.text_prompt = f"a photo with {emotion} emotion."
        
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


# 相似度计算函数
def calculate_similarity(image_features, text_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze().item()
    return similarity


# 处理聚类的函数，并将符合要求的聚类复制到新目录
def process_clusters_in_batches(cluster_dir, emotion, model, preprocess, dest_dir, batch_size=64, threshold=0.25, min_images=3):
    dataset = EmotionDataset(cluster_dir, emotion, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    valid_images = []
    total_images = 0

    # 文本输入
    text_input = clip.tokenize(f"a photo with emotion of {emotion}").to("cuda")

    with torch.no_grad():
        text_features = model.encode_text(text_input).detach().cpu()

        for batch in dataloader:
            images, image_paths = batch
            images = images.to("cuda")

            image_features = model.encode_image(images).detach().cpu()

            # 计算相似度
            for i in range(images.size(0)):
                similarity = calculate_similarity(image_features[i], text_features[0])
                image_path = image_paths[i]
                total_images += 1

                if similarity > threshold:
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


# 主函数：处理所有情感类别中的所有聚类
def process_all_clusters(root_dir, dest_dir, batch_size=64, threshold=0.25, min_images=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载CLIP模型和预处理器
    model, preprocess = clip.load("ViT-B/32", device=device)

    for emotion in os.listdir(root_dir):
        emotion_path = os.path.join(root_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue

        for cluster in os.listdir(emotion_path):
            if cluster == 'cluster_-1':
                continue  # 忽略噪声点聚类

            cluster_dir = os.path.join(emotion_path, cluster)
            process_clusters_in_batches(cluster_dir, emotion, model, preprocess, dest_dir, batch_size=batch_size, threshold=threshold, min_images=min_images)


if __name__ == "__main__":
    data_root = "/openbayes/input/input0/EmoSet-118K/cluster_result"  
    dest_root = "/openbayes/input/input0/EmoSet-118K/filtered_cluster_result"
    process_all_clusters(data_root, dest_root, batch_size=64, threshold=0.17, min_images=3)  