import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from model import ResNet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_PATH = "./data/cifar_test_nolabel.pkl"  # Adjust if needed
BATCH_SIZE = 128
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

class CIFARTestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape: (32, 32, 3)
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1) / 255.0  # (3,32,32)
        if self.transform:
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img)
            img = self.transform(img)
        return img

def main():
    # 1. Load model
    model = ResNet18(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()

    # 2. Load test data
    with open(TEST_PATH, 'rb') as f:
        test_dict = pickle.load(f)
    test_images_np = test_dict[b'data']  # shape: (N, 32, 32, 3)
    N_test = test_images_np.shape[0]

    # 3. Create DataLoader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    test_dataset = CIFARTestDataset(test_images_np, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Inference
    all_preds = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Test Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())

    # 5. Create submission.csv
    submission_df = pd.DataFrame({
        "ID": range(N_test),
        "Labels": all_preds
    })
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' created successfully!")

if __name__ == "__main__":
    main()
