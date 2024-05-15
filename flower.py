import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_data = datasets.ImageFolder(root=r'D:\下载\102flowers', transform=transform)
val_data = datasets.ImageFolder(root=r'D:\下载\102segmentations', transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, num_classes=102):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)
        return x

model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            #前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            ##反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ##损失累加
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 评估模型
def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy on validation set: {accuracy:.4f}")
    return accuracy
# 训练模型
train(model, train_loader, criterion, optimizer)

#评估模型并获取验证集准确率
val_accuracy = evaluate(model, val_loader)


# 可视化结果（以随机验证集图像为例）
def visualize_predictions(model, val_loader, num_images=5):
    model.eval()
    images_so_far = 0
    fig, axes = plt.subplots(nrows=min(num_images, 4), ncols=max(1, num_images // 4), figsize=(10, 7), squeeze=False)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_so_far >= num_images:
                    break

                # 确保图像数据在[0, 1]范围内
                img_tensor = images[i].cpu().numpy().transpose((1, 2, 0))
                img_tensor = np.clip(img_tensor, 0, 1)

                row = images_so_far // num_images // 4
                col = images_so_far % (num_images // 4)
                ax = axes[row, col]
                ax.axis('off')
                ax.set_title(f'Predicted: {predicted[i].item()}, Actual: {labels[i].item()}')
                ax.imshow(img_tensor)

                images_so_far += 1

    plt.tight_layout()
    plt.show()

visualize_predictions(model, val_loader)