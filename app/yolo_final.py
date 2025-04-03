import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image

# 1. Custom Dataset Class
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
        # Basic augmentation
        self.augment = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt'))
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        # Resize image
        image = transforms.Resize((self.img_size, self.img_size))(image)
        
        # Apply augmentations
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor and normalize
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        
        # Load labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Convert labels to tensor
        labels = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append([class_id, x_center, y_center, width, height])
        
        return image, torch.tensor(labels)

# 2. YOLOv8 Model Architecture
class YOLOv8(torch.nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv8, self).__init__()
        
        # Backbone
        self.backbone = torch.nn.Sequential(
            # Conv blocks
            self._make_conv_block(3, 64, 3),
            self._make_conv_block(64, 128, 3),
            self._make_conv_block(128, 256, 3),
            self._make_conv_block(256, 512, 3),
            self._make_conv_block(512, 1024, 3),
        )
        
        # Detection head
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, num_classes + 5, 1)  # 5 for bbox (x,y,w,h,conf)
        )
        
    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs

# 3. Loss Function
class YOLOLoss(torch.nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        
    def forward(self, predictions, targets):
        obj_loss = torch.nn.functional.mse_loss(predictions[..., 4], targets[..., 4])
        box_loss = torch.nn.functional.mse_loss(predictions[..., :4], targets[..., :4])
        cls_loss = torch.nn.functional.cross_entropy(predictions[..., 5:], targets[..., 5:].argmax(-1))
        
        return obj_loss + box_loss + cls_loss

# 4. Training Function
def train(model, dataloader, optimizer, criterion, device, epochs=100):
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0

# 5. Main Training Script
if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    img_size = 640
    num_classes = 80  # COCO has 80 classes
    learning_rate = 0.001
    epochs = 100
    
    # Create datasets
    train_dataset = YOLODataset(
        img_dir='path/to/train/images',
        label_dir='path/to/train/labels',
        img_size=img_size
    )
    
    val_dataset = YOLODataset(
        img_dir='path/to/val/images',
        label_dir='path/to/val/labels',
        img_size=img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = YOLOv8(num_classes=num_classes)
    criterion = YOLOLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train(model, train_loader, optimizer, criterion, device, epochs=epochs)
    
    # Save the model
    torch.save(model.state_dict(), 'yolov8_custom.pth')
