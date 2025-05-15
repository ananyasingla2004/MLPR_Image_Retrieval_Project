import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# 1. Custom CNN Backbone (Simple CNN from scratch)
class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=128, image_size=224): # Added image_size
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()

        self._initialize_fc_layers(H=image_size[0], W=image_size[1], embedding_dim=embedding_dim) # Unpack image_size

    def _initialize_fc_layers(self, C=3, H=224, W=224, embedding_dim=128):
        # Perform a dry run with a dummy input to determine the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            flattened_size = x.flatten(start_dim=1).shape[1]
        
        self.fc1 = nn.Linear(flattened_size, 512)
        self.relu4 = nn.ReLU() # Added relu4 here
        self.fc2 = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, backbone):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone # This will be an instance of SimpleCNN

    def forward_one(self, x):
        return self.backbone(x)

    def forward(self, input1, input2, input3=None): # input3 is for triplet
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        if input3 is not None:
            output3 = self.forward_one(input3)
            return output1, output2, output3
        return output1, output2

# 3. Distance Metrics
def manhattan_distance_func(output1, output2):
    return F.pairwise_distance(output1, output2, p=1)

def euclidean_distance_func(output1, output2):
    return F.pairwise_distance(output1, output2, p=2)

def cosine_distance_func(output1, output2): # Cosine distance for loss is 1 - similarity
    return 1 - F.cosine_similarity(output1, output2)

# 4. Loss Functions
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        if self.distance_metric not in ['euclidean', 'manhattan', 'cosine']:
            raise ValueError(f"Unknown distance metric for ContrastiveLoss: {self.distance_metric}")

    def forward(self, output1, output2, label):
        # label: 0 for similar, 1 for dissimilar
        if self.distance_metric == 'euclidean':
            dist = euclidean_distance_func(output1, output2)
        elif self.distance_metric == 'manhattan':
            dist = manhattan_distance_func(output1, output2)
        elif self.distance_metric == 'cosine':
            dist = cosine_distance_func(output1, output2)
        
        loss_contrastive = torch.mean((1 - label) * torch.pow(dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss_contrastive

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, distance_metric='euclidean'): # Margin from hold1.py
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        if self.distance_metric not in ['euclidean', 'manhattan', 'cosine']:
            raise ValueError(f"Unknown distance metric for TripletLoss: {self.distance_metric}")

    def forward(self, anchor, positive, negative):
        if self.distance_metric == 'euclidean':
            pos_dist = euclidean_distance_func(anchor, positive)
            neg_dist = euclidean_distance_func(anchor, negative)
        elif self.distance_metric == 'manhattan':
            pos_dist = manhattan_distance_func(anchor, positive)
            neg_dist = manhattan_distance_func(anchor, negative)
        elif self.distance_metric == 'cosine':
            pos_dist = cosine_distance_func(anchor, positive)
            neg_dist = cosine_distance_func(anchor, negative)
            
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# 5. Dataset
class SiameseDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, mode='triplet', image_size=224):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode # 'triplet', 'pair', or 'inference'
        self.image_size = image_size # To ensure images are loaded correctly

        self.labels = self.df['class'].unique()
        self.label_to_indices = {label: np.where(self.df['class'] == label)[0]
                                 for label in self.labels}
        
        if mode == 'pair':
            # For contrastive loss, generate pairs and labels (0 for same class, 1 for different)
            self.generate_pairs()

    def generate_pairs(self):
        self.pairs = []
        self.pair_labels = []
        n_samples = len(self.df)
        for i in range(n_samples):
            # Add a positive pair
            anchor_row = self.df.iloc[i]
            anchor_label = anchor_row['class']
            positive_idx = i
            while positive_idx == i: # Ensure different image if multiple exist for class
                 if len(self.label_to_indices[anchor_label]) > 1:
                    positive_idx = random.choice(self.label_to_indices[anchor_label])
                 else: # Only one image for this class, pair with itself (less ideal)
                    positive_idx = i 
                    break 
            self.pairs.append((i, positive_idx))
            self.pair_labels.append(0) # 0 for similar

            # Add a negative pair
            negative_label = random.choice(list(set(self.labels) - {anchor_label}))
            negative_idx = random.choice(self.label_to_indices[negative_label])
            self.pairs.append((i, negative_idx))
            self.pair_labels.append(1) # 1 for dissimilar
        print(f"Generated {len(self.pairs)} pairs for contrastive training.")

    def __getitem__(self, index):
        if self.mode == 'triplet':
            # Anchor image
            anchor_row = self.df.iloc[index]
            anchor_label = anchor_row['class']
            anchor_img_path = os.path.join(self.image_dir, anchor_row['name'])
            anchor_img = Image.open(anchor_img_path).convert('RGB')

            # Positive image
            positive_indices = self.label_to_indices[anchor_label]
            positive_idx = index
            # Ensure positive is different from anchor if possible
            if len(positive_indices) > 1:
                while positive_idx == index:
                    positive_idx = random.choice(positive_indices)
            else: # Only one image for this class, positive is same as anchor
                positive_idx = index 
            
            positive_row = self.df.iloc[positive_idx]
            positive_img_path = os.path.join(self.image_dir, positive_row['name'])
            positive_img = Image.open(positive_img_path).convert('RGB')

            # Negative image
            negative_label_list = list(set(self.labels) - {anchor_label})
            if not negative_label_list: # Handle case with only one class (should not happen in good dataset)
                negative_label = anchor_label 
            else:
                negative_label = random.choice(negative_label_list)

            negative_idx = random.choice(self.label_to_indices[negative_label])
            negative_row = self.df.iloc[negative_idx]
            negative_img_path = os.path.join(self.image_dir, negative_row['name'])
            negative_img = Image.open(negative_img_path).convert('RGB')

            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            
            return anchor_img, positive_img, negative_img

        elif self.mode == 'pair':
            idx1, idx2 = self.pairs[index]
            label = self.pair_labels[index]

            img1_row = self.df.iloc[idx1]
            img2_row = self.df.iloc[idx2]

            img1_path = os.path.join(self.image_dir, img1_row['name'])
            img2_path = os.path.join(self.image_dir, img2_row['name'])

            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            return img1, img2, torch.tensor(label, dtype=torch.float32)

        elif self.mode == 'inference': # For gallery/query feature extraction
            row = self.df.iloc[index]
            img_path = os.path.join(self.image_dir, row['name'])
            image = Image.open(img_path).convert('RGB')
            class_id = row['class']
            filename = row['name']
            
            if self.transform:
                image = self.transform(image)
            return image, class_id, filename
        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}")

    def __len__(self):
        if self.mode == 'pair':
            return len(self.pairs)
        return len(self.df)

    def get_balanced_subset_indices(self, max_per_class):
        balanced_indices = []
        if max_per_class <= 0:
            print("Warning: max_per_class in get_balanced_subset_indices must be positive. Returning empty list.")
            return []

        for label_class in self.labels: # Renamed 'label' to 'label_class' to avoid conflict
            class_indices = list(self.label_to_indices[label_class]) # Make a mutable copy
            random.shuffle(class_indices)
            num_to_take = min(len(class_indices), max_per_class)
            balanced_indices.extend(class_indices[:num_to_take])
        
        random.shuffle(balanced_indices)
        if not balanced_indices:
            print(f"Warning: No indices selected for balanced subset with max_per_class={max_per_class}. Original dataset size: {len(self.df)}")
        return balanced_indices

# Data Transforms
def get_data_transforms(image_size=(224, 224), high_res_logic=False): # Renamed param for clarity
    current_resize_size = image_size[0] if isinstance(image_size, tuple) else image_size
    current_crop_size = image_size[0] if isinstance(image_size, tuple) else image_size

    train_transform = transforms.Compose([
        transforms.Resize((current_resize_size, current_resize_size)),
        transforms.RandomCrop(current_crop_size), 
        transforms.RandomHorizontalFlip(),      
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Softer jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([ # Also used for inference
        transforms.Resize((current_resize_size, current_resize_size)),
        transforms.CenterCrop(current_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'val': val_transform}

# 6. Training Function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, scheduler=None, loss_type='triplet'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    has_val_loader = 'val' in dataloaders and dataloaders['val'] is not None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'val' and not has_val_loader:
                continue # Skip val phase if no val loader

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch_data in enumerate(progress_bar):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if loss_type == 'triplet':
                        anchor, positive, negative = batch_data
                        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                        emb_a, emb_p, emb_n = model(anchor, positive, negative)
                        loss = criterion(emb_a, emb_p, emb_n)
                    elif loss_type == 'contrastive':
                        img1, img2, label = batch_data
                        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                        emb1, emb2 = model(img1, img2)
                        loss = criterion(emb1, emb2, label)
                    else:
                        raise ValueError(f"Unsupported loss type: {loss_type}")

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * (anchor.size(0) if loss_type == 'triplet' else img1.size(0))
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val': # This implies has_val_loader is true
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"New best validation loss: {best_loss:.4f}")
            elif phase == 'train' and not has_val_loader:
                # If no validation phase, the model from the current training epoch is the best so far for this phase.
                # This means the model from the *last* training epoch will be used.
                best_model_wts = copy.deepcopy(model.state_dict())
        
        if scheduler:
            scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    if has_val_loader:
        print(f'Best val Loss: {best_loss:.4f}')
    else:
        print('Training phase finished. Using model from the last epoch as best model for this phase.')

    model.load_state_dict(best_model_wts)
    return model

# 7. Evaluation and Retrieval
def extract_all_features(model, dataloader, device, image_size):
    model.eval()
    all_features_list = []
    all_classes_list = []
    all_filenames_list = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting features"):
            images, classes, filenames = batch_data # Assumes dataset in 'inference' mode
            images = images.to(device)
            
            features = model.forward_one(images) # Use forward_one to get embeddings
            
            all_features_list.append(features.cpu().numpy())
            all_classes_list.extend(classes.numpy() if isinstance(classes, torch.Tensor) else classes)
            all_filenames_list.extend(filenames)
            
    all_features = np.concatenate(all_features_list, axis=0)
    all_classes = np.array(all_classes_list)
    
    return all_features, all_classes, all_filenames_list

def compute_similarity_scores(query_features_torch, gallery_features_torch, metric_type='cosine', batch_size=100):
    """
    Compute similarity scores between query and gallery features in batches to avoid memory issues.
    
    Args:
        query_features_torch: Query feature embeddings (torch.Tensor)
        gallery_features_torch: Gallery feature embeddings (torch.Tensor)
        metric_type: Distance metric to use ('cosine', 'euclidean', or 'manhattan')
        batch_size: Size of batches to process at once to save memory
        
    Returns:
        numpy.ndarray: Similarity matrix where higher values mean more similar
    """
    num_queries = query_features_torch.shape[0]
    num_gallery = gallery_features_torch.shape[0]
    
    # Initialize the output matrix (on CPU to save device memory)
    similarity_matrix = np.zeros((num_queries, num_gallery), dtype=np.float32)
    
    # Pre-normalize gallery features for cosine similarity to avoid redundant computation
    if metric_type == 'cosine':
        gallery_features_norm = F.normalize(gallery_features_torch, p=2, dim=1)
    
    # Process in batches
    with torch.no_grad():  # No need for gradients during inference
        for q_start in range(0, num_queries, batch_size):
            q_end = min(q_start + batch_size, num_queries)
            q_batch = query_features_torch[q_start:q_end]
            
            if metric_type == 'cosine':
                q_batch_norm = F.normalize(q_batch, p=2, dim=1)
            
            for g_start in range(0, num_gallery, batch_size):
                g_end = min(g_start + batch_size, num_gallery)
                g_batch = gallery_features_torch[g_start:g_end]
                
                if metric_type == 'cosine':
                    g_batch_norm = gallery_features_norm[g_start:g_end]
                    batch_similarity = torch.mm(q_batch_norm, g_batch_norm.T)
                elif metric_type == 'euclidean':
                    batch_similarity = -torch.cdist(q_batch, g_batch, p=2.0)
                elif metric_type == 'manhattan':
                    batch_similarity = -torch.cdist(q_batch, g_batch, p=1.0)
                else:
                    raise ValueError(f"Unknown similarity metric_type: {metric_type}")
                
                # Store the result in our matrix
                similarity_matrix[q_start:q_end, g_start:g_end] = batch_similarity.cpu().numpy()
    
    return similarity_matrix

def calculate_precision_at_k(query_features_torch, query_classes_np, gallery_features_torch, gallery_classes_np, k, metric_type, current_device, batch_size=100):
    """
    Calculate average precision@k using batched processing to save memory
    """
    num_queries = query_features_torch.shape[0]
    all_precisions_at_k = []
    
    # Process each query independently to save memory
    for i in tqdm(range(num_queries), desc=f"Calculating Precision@{k} ({metric_type})"):
        query_class_val = query_classes_np[i]
        q_feat_single = query_features_torch[i:i+1]
        
        # Compute similarities for this query in batches
        similarities_for_query = np.zeros(len(gallery_features_torch), dtype=np.float32)
        
        with torch.no_grad():
            if metric_type == 'cosine':
                q_norm = F.normalize(q_feat_single, p=2, dim=1)
                
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    g_norm = F.normalize(g_batch, p=2, dim=1)
                    batch_sim = torch.mm(q_norm, g_norm.T).cpu().numpy()[0]
                    similarities_for_query[g_start:g_end] = batch_sim
                    
            elif metric_type == 'euclidean':
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    batch_sim = -torch.cdist(q_feat_single, g_batch, p=2.0).cpu().numpy()[0]
                    similarities_for_query[g_start:g_end] = batch_sim
                    
            elif metric_type == 'manhattan':
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    batch_sim = -torch.cdist(q_feat_single, g_batch, p=1.0).cpu().numpy()[0]
                    similarities_for_query[g_start:g_end] = batch_sim
        
        # Get top k indices
        sorted_indices = np.argsort(similarities_for_query)[::-1]
        top_k_indices = sorted_indices[:k]
        retrieved_classes = gallery_classes_np[top_k_indices]
        
        # Calculate precision
        num_correct = np.sum(retrieved_classes == query_class_val)
        precision_at_k = num_correct / k if k > 0 else 0.0
        all_precisions_at_k.append(precision_at_k)
        
    mean_precision_at_k = np.mean(all_precisions_at_k) if all_precisions_at_k else 0.0
    return mean_precision_at_k

def calculate_map_at_k(query_features_torch, query_classes_np, gallery_features_torch, gallery_classes_np, k, metric_type, current_device, batch_size=100):
    """
    Calculate mean Average Precision@k using batched processing to save memory
    """
    num_queries = query_features_torch.shape[0]
    all_aps_at_k = []

    for i in tqdm(range(num_queries), desc=f"Calculating mAP@{k} ({metric_type})"):
        query_class_val = query_classes_np[i]
        q_feat_single = query_features_torch[i:i+1]
        
        # Compute similarities for this query in batches
        similarities_for_query = np.zeros(len(gallery_features_torch), dtype=np.float32)
        
        with torch.no_grad():
            if metric_type == 'cosine':
                q_norm = F.normalize(q_feat_single, p=2, dim=1)
                
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    g_norm = F.normalize(g_batch, p=2, dim=1)
                    batch_sim = torch.mm(q_norm, g_norm.T).cpu().numpy()[0]
                    similarities_for_query[g_start:g_end] = batch_sim
                    
            elif metric_type == 'euclidean':
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    batch_sim = -torch.cdist(q_feat_single, g_batch, p=2.0).cpu().numpy()[0]
                    similarities_for_query[g_start:g_end] = batch_sim
                    
            elif metric_type == 'manhattan':
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    batch_sim = -torch.cdist(q_feat_single, g_batch, p=1.0).cpu().numpy()[0]
                    similarities_for_query[g_start:g_end] = batch_sim

        sorted_indices = np.argsort(similarities_for_query)[::-1]
        
        actual_k = min(k, len(gallery_classes_np))
        top_k_indices = sorted_indices[:actual_k]

        retrieved_classes_top_k = gallery_classes_np[top_k_indices]
        
        relevant_mask = (retrieved_classes_top_k == query_class_val)
        
        num_relevant_in_top_k = np.sum(relevant_mask)

        if num_relevant_in_top_k == 0:
            ap_at_k = 0.0
        else:
            pk_sum = 0.0
            num_correct_so_far = 0
            for j_idx in range(len(relevant_mask)):
                if relevant_mask[j_idx]:
                    num_correct_so_far += 1
                    pk_sum += num_correct_so_far / (j_idx + 1.0)
            ap_at_k = pk_sum / num_relevant_in_top_k
        all_aps_at_k.append(ap_at_k)
        
    mean_ap_at_k = np.mean(all_aps_at_k) if all_aps_at_k else 0.0
    return mean_ap_at_k

# 8. Visualization Function
def visualize_retrieval_results(query_labels_np, query_filepaths, query_features_torch, 
                                gallery_labels_np, gallery_filepaths, gallery_features_torch, 
                                k, num_queries_to_visualize, 
                                query_image_base_dir, 
                                gallery_image_base_dir, 
                                image_size, metric_type, current_device, batch_size=100):
    
    num_total_queries = query_features_torch.shape[0]
    if num_total_queries == 0:
        print("No queries to visualize.")
        return

    if isinstance(query_labels_np, torch.Tensor):
        query_labels_np = query_labels_np.cpu().numpy()
    if isinstance(gallery_labels_np, torch.Tensor):
        gallery_labels_np = gallery_labels_np.cpu().numpy()
    
    # Randomly select query indices to visualize
    query_indices_to_show = random.sample(range(num_total_queries), min(num_queries_to_visualize, num_total_queries))
    
    # Compute similarity scores only for the queries we'll visualize (memory efficient)
    for i, query_idx in enumerate(query_indices_to_show):
        print(f"Visualizing query {i+1}/{len(query_indices_to_show)}")
        query_image_path = os.path.join(query_image_base_dir, query_filepaths[query_idx])
        query_label = query_labels_np[query_idx]
        
        # Compute similarity for just this one query
        q_feat_single = query_features_torch[query_idx:query_idx+1]
        
        # Compute similarities in batches
        similarity_scores = np.zeros(len(gallery_features_torch), dtype=np.float32)
        
        with torch.no_grad():
            if metric_type == 'cosine':
                q_norm = F.normalize(q_feat_single, p=2, dim=1)
                
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    g_norm = F.normalize(g_batch, p=2, dim=1)
                    batch_sim = torch.mm(q_norm, g_norm.T).cpu().numpy()[0]
                    similarity_scores[g_start:g_end] = batch_sim
                    
            elif metric_type == 'euclidean':
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    batch_sim = -torch.cdist(q_feat_single, g_batch, p=2.0).cpu().numpy()[0]
                    similarity_scores[g_start:g_end] = batch_sim
                    
            elif metric_type == 'manhattan':
                for g_start in range(0, len(gallery_features_torch), batch_size):
                    g_end = min(g_start + batch_size, len(gallery_features_torch))
                    g_batch = gallery_features_torch[g_start:g_end]
                    batch_sim = -torch.cdist(q_feat_single, g_batch, p=1.0).cpu().numpy()[0]
                    similarity_scores[g_start:g_end] = batch_sim
            else:
                raise ValueError(f"Unknown similarity metric for visualization: {metric_type}")
        
        # Sort results
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        actual_k_viz = min(k, len(gallery_filepaths))
        top_k_indices = sorted_indices[:actual_k_viz]
        top_k_scores = similarity_scores[top_k_indices]

        retrieved_gallery_paths = [os.path.join(gallery_image_base_dir, gallery_filepaths[j]) for j in top_k_indices]
        retrieved_gallery_labels = gallery_labels_np[top_k_indices]

        fig, axes = plt.subplots(1, actual_k_viz + 1, figsize=( (actual_k_viz + 1) * 3, 4) )
        fig.suptitle(f"Query {i+1}/{num_queries_to_visualize} (Class: {query_label}) - Top {actual_k_viz} ({metric_type})", fontsize=10)

        try:
            q_img = Image.open(query_image_path).convert("RGB").resize(image_size)
            axes[0].imshow(q_img)
            axes[0].set_title(f"Query\nClass: {query_label}", fontsize=8)
            axes[0].axis('off')
        except FileNotFoundError:
            axes[0].set_title(f"Query Image Not Found\n{query_filepaths[query_idx]}", fontsize=8, color='red')
            axes[0].axis('off')

        for j in range(actual_k_viz):
            if j < len(retrieved_gallery_paths):
                retrieved_path = retrieved_gallery_paths[j]
                retrieved_label = retrieved_gallery_labels[j]
                is_correct = (retrieved_label == query_label)
                border_color = 'green' if is_correct else 'red'
                
                current_score = top_k_scores[j]
                if metric_type in ['euclidean', 'manhattan']:
                    display_score = -current_score 
                else:
                    display_score = current_score

                try:
                    r_img = Image.open(retrieved_path).convert("RGB").resize(image_size)
                    axes[j+1].imshow(r_img)
                    axes[j+1].set_title(f"Retrieved {j+1}\nClass: {retrieved_label}\nScore: {display_score:.2f}", fontsize=8)
                    axes[j+1].axis('off')
                    for spine in axes[j+1].spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(2)
                except FileNotFoundError:
                    axes[j+1].set_title(f"Retrieved Image Not Found\n{os.path.basename(retrieved_path)}", fontsize=8, color='red')
                    axes[j+1].axis('off')
            else:
                axes[j+1].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = f"retrieval_results_query_{query_idx}_{metric_type}.png"
        plt.savefig(plot_filename)
        print(f"Saved visualization to {plot_filename}")
        plt.show()

def perform_retrieval_and_evaluation(
    model,
    query_loader, gallery_loader,
    k_for_metrics,
    metric_types_to_test,
    primary_metric_for_viz,
    device,
    image_base_dir_query, 
    image_base_dir_gallery,
    image_size,
    num_queries_to_visualize,
    config,
    pre_extracted_query_features_np=None,
    pre_extracted_query_labels_np=None,
    pre_extracted_query_filepaths=None,
    pre_extracted_gallery_features_np=None,
    pre_extracted_gallery_labels_np=None,
    pre_extracted_gallery_filepaths=None):

    if pre_extracted_query_features_np is not None:
        print("Using pre-extracted query features.")
        query_features_np = pre_extracted_query_features_np
        query_labels_np = pre_extracted_query_labels_np
        query_filepaths = pre_extracted_query_filepaths
    elif model and query_loader:
        print("Extracting query features...")
        query_features_np, query_labels_np, query_filepaths = extract_all_features(model, query_loader, device, image_size)
    else:
        raise ValueError("Either model and query_loader or pre-extracted query features must be provided.")

    if pre_extracted_gallery_features_np is not None:
        print("Using pre-extracted gallery features.")
        gallery_features_np = pre_extracted_gallery_features_np
        gallery_labels_np = pre_extracted_gallery_labels_np
        gallery_filepaths = pre_extracted_gallery_filepaths
    elif model and gallery_loader: 
        print("Extracting gallery features...")
        gallery_features_np, gallery_labels_np, gallery_filepaths = extract_all_features(model, gallery_loader, device, image_size)
    else: 
        print("No specific gallery features/loader. Using query set as gallery.")
        gallery_features_np = query_features_np
        gallery_labels_np = query_labels_np
        gallery_filepaths = query_filepaths

    query_features_torch = torch.from_numpy(query_features_np).to(device)
    gallery_features_torch = torch.from_numpy(gallery_features_np).to(device)

    print(f"\n--- Starting Visualization for metric: {primary_metric_for_viz} ---")
    visualize_retrieval_results(
        query_labels_np, query_filepaths, query_features_torch,
        gallery_labels_np, gallery_filepaths, gallery_features_torch,
        k=k_for_metrics, 
        num_queries_to_visualize=num_queries_to_visualize,
        query_image_base_dir=image_base_dir_query,
        gallery_image_base_dir=image_base_dir_gallery,
        image_size=image_size,
        metric_type=primary_metric_for_viz,
        current_device=device
    )

    print(f"\n--- Calculating Metrics (Precision@{k_for_metrics} and mAP@{k_for_metrics}) ---")
    results_precision = {}
    results_map = {}

    for metric_type in metric_types_to_test:
        print(f"\n-- Metric Type: {metric_type} --")
        
        avg_precision = calculate_precision_at_k(
            query_features_torch, query_labels_np,
            gallery_features_torch, gallery_labels_np,
            k=k_for_metrics, metric_type=metric_type, current_device=device
        )
        results_precision[metric_type] = avg_precision
        print(f"Average Precision@{k_for_metrics} ({metric_type}): {avg_precision:.4f}")

        map_k_value = calculate_map_at_k(
            query_features_torch, query_labels_np,
            gallery_features_torch, gallery_labels_np,
            k=k_for_metrics, metric_type=metric_type, current_device=device
        )
        results_map[metric_type] = map_k_value

    print(f"\nComparison of distance metrics (Precision@{k_for_metrics} and mAP@{k_for_metrics}):")
    print("=" * 60)
    header = f"{'Distance Metric':<20} {'Precision@' + str(k_for_metrics):<15} {'mAP@' + str(k_for_metrics):<15}"
    print(header)
    print("-" * 60)
    for metric_type in metric_types_to_test:
        precision_val = results_precision.get(metric_type, float('nan'))
        map_val = results_map.get(metric_type, float('nan'))
        print(f"{metric_type:<20} {precision_val:<15.4f} {map_val:<15.4f}")
    print("=" * 60)

    return results_precision, results_map

if __name__ == '__main__':
    CONFIG = {
        "base_dir": ".",
        "train_csv": "visual-product-recognition/train.csv",
        "val_csv": None,
        "query_csv": "visual-product-recognition/test.csv",
        "image_dir_train": "visual-product-recognition/train/train",
        "image_dir_val": "val",
        "image_dir_query": "visual-product-recognition/test/test",
        "model_save_path": "siamese_from_scratch_try3.pth",
        "embedding_dim": 1280,
        "image_size": (224, 224),
        "batch_size": 16,
        "num_epochs": 1,
        "finetune_epochs": 3,
        "learning_rate": 0.001,
        "finetune_learning_rate": 0.0001,
        "optimizer": "adam",
        "loss_type": "contrastive",
        "distance_metric": "euclidean",
        "triplet_margin": 0.2,
        "contrastive_margin": 1.0,
        "num_workers": 4,
        "k_for_map": 5,
        "device": "mps",
        "perform_finetuning": True,
        "num_samples_per_class_finetune": 5,
        "num_queries_to_visualize": 5,
        "metric_types_to_test": ["cosine", "euclidean", "manhattan"],
        "primary_metric_for_viz": "euclidean"
    }

    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    data_transforms = get_data_transforms(CONFIG["image_size"])

    print("Loading data for initial training...")
    train_dataset_path = os.path.join(CONFIG["base_dir"], CONFIG["train_csv"])
    train_image_dir_path = os.path.join(CONFIG["base_dir"], CONFIG["image_dir_train"])
    
    dataset_mode_initial = 'pair' if CONFIG["loss_type"] == 'contrastive' else CONFIG["loss_type"]
    if dataset_mode_initial not in ['triplet', 'pair']:
        raise ValueError(f"Invalid loss_type '{CONFIG['loss_type']}' for dataset mode determination. Must map to 'triplet' or 'pair'.")

    initial_train_dataset = SiameseDataset(
        csv_file=train_dataset_path,
        image_dir=train_image_dir_path,
        transform=data_transforms['train'],
        mode=dataset_mode_initial,
        image_size=CONFIG["image_size"]
    )
    initial_train_loader = DataLoader(initial_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

    val_loader = None
    if CONFIG["val_csv"]:
        val_dataset_path = os.path.join(CONFIG["base_dir"], CONFIG["val_csv"])
        val_image_dir_path = os.path.join(CONFIG["base_dir"], CONFIG["image_dir_val"])
        dataset_mode_val = 'pair' if CONFIG["loss_type"] == 'contrastive' else CONFIG["loss_type"]
        if dataset_mode_val not in ['triplet', 'pair']:
             raise ValueError(f"Invalid loss_type '{CONFIG['loss_type']}' for validation dataset mode. Must map to 'triplet' or 'pair'.")

        val_dataset = SiameseDataset(
            csv_file=val_dataset_path,
            image_dir=val_image_dir_path,
            transform=data_transforms['val'],
            mode=dataset_mode_val, 
            image_size=CONFIG["image_size"]
        )
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    dataloaders_initial = {'train': initial_train_loader, 'val': val_loader}

    print("Initializing model...")
    backbone = SimpleCNN(embedding_dim=CONFIG["embedding_dim"], image_size=CONFIG["image_size"])
    siamese_model = SiameseNetwork(backbone).to(device)

    if CONFIG["loss_type"] == "triplet":
        criterion = TripletLoss(margin=CONFIG["triplet_margin"], distance_metric=CONFIG["distance_metric"])
    elif CONFIG["loss_type"] == "contrastive":
        criterion = ContrastiveLoss(margin=CONFIG["contrastive_margin"], distance_metric=CONFIG["distance_metric"])
    else:
        raise ValueError(f"Unsupported loss type: {CONFIG['loss_type']}")

    if CONFIG["optimizer"].lower() == "adam":
        optimizer_initial = torch.optim.Adam(siamese_model.parameters(), lr=CONFIG["learning_rate"])
    elif CONFIG["optimizer"].lower() == "sgd":
        optimizer_initial = torch.optim.SGD(siamese_model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {CONFIG['optimizer']}")
    
    print("Starting initial training phase...")
    siamese_model = train_model(
        siamese_model, dataloaders_initial, criterion, optimizer_initial, 
        num_epochs=CONFIG["num_epochs"], loss_type=CONFIG["loss_type"]
    )
    print("Initial training finished.")

    if CONFIG["perform_finetuning"]:
        print("Preparing for finetuning phase...")
        balanced_indices = initial_train_dataset.get_balanced_subset_indices(CONFIG["num_samples_per_class_finetune"])
        
        dataset_mode_finetune = 'pair' if CONFIG["loss_type"] == 'contrastive' else CONFIG["loss_type"]
        if dataset_mode_finetune not in ['triplet', 'pair']:
            raise ValueError(f"Invalid loss_type '{CONFIG['loss_type']}' for finetuning dataset mode. Must map to 'triplet' or 'pair'.")

        finetune_dataset_for_indices = SiameseDataset(
            csv_file=train_dataset_path,
            image_dir=train_image_dir_path,
            transform=data_transforms['train'],
            mode=dataset_mode_finetune,
            image_size=CONFIG["image_size"]
        )
        finetune_subset = Subset(finetune_dataset_for_indices, balanced_indices)

        if not finetune_subset:
            print("Finetuning subset is empty. Skipping finetuning.")
        else:
            print(f"Finetuning with {len(finetune_subset)} samples.")
            finetune_loader = DataLoader(finetune_subset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
            
            dataloaders_finetune = {'train': finetune_loader, 'val': None}

            if CONFIG["optimizer"].lower() == "adam":
                optimizer_finetune = torch.optim.Adam(siamese_model.parameters(), lr=CONFIG["finetune_learning_rate"])
            elif CONFIG["optimizer"].lower() == "sgd":
                optimizer_finetune = torch.optim.SGD(siamese_model.parameters(), lr=CONFIG["finetune_learning_rate"], momentum=0.9)
            else:
                raise ValueError(f"Unsupported optimizer: {CONFIG['optimizer']}")

            print("Starting finetuning phase...")
            siamese_model = train_model(
                siamese_model, dataloaders_finetune, criterion, optimizer_finetune, 
                num_epochs=CONFIG["finetune_epochs"], loss_type=CONFIG["loss_type"]
            )
            print("Finetuning finished.")

    print(f"Saving final model to {CONFIG['model_save_path']}")
    torch.save(siamese_model.state_dict(), CONFIG['model_save_path'])
    print("Model saved.")

    print("Starting evaluation phase...")
    eval_dataset_path = os.path.join(CONFIG["base_dir"], CONFIG["query_csv"])
    eval_image_dir_path = os.path.join(CONFIG["base_dir"], CONFIG["image_dir_query"])

    eval_dataset = SiameseDataset(
        csv_file=eval_dataset_path,
        image_dir=eval_image_dir_path,
        transform=data_transforms['val'],
        mode='inference',
        image_size=CONFIG["image_size"]
    )
    eval_loader = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print("Extracting features from the evaluation set (query)...")
    query_features_np, query_labels_np, query_filepaths = extract_all_features(siamese_model, eval_loader, device, CONFIG["image_size"])
    
    print("Using evaluation set as gallery for retrieval.")
    gallery_features_np = query_features_np
    gallery_labels_np = query_labels_np
    gallery_filepaths = query_filepaths
    gallery_image_dir_path = eval_image_dir_path 

    perform_retrieval_and_evaluation(
        model=None,
        query_loader=None, gallery_loader=None, 
        k_for_metrics=CONFIG['k_for_map'],
        metric_types_to_test=CONFIG['metric_types_to_test'],
        primary_metric_for_viz=CONFIG.get('primary_metric_for_viz', CONFIG['distance_metric']),
        device=device,
        image_base_dir_query=eval_image_dir_path,
        image_base_dir_gallery=gallery_image_dir_path,
        image_size=CONFIG["image_size"],
        num_queries_to_visualize=CONFIG["num_queries_to_visualize"],
        config=CONFIG,
        pre_extracted_query_features_np=query_features_np,
        pre_extracted_query_labels_np=query_labels_np,
        pre_extracted_query_filepaths=query_filepaths,
        pre_extracted_gallery_features_np=gallery_features_np,
        pre_extracted_gallery_labels_np=gallery_labels_np,
        pre_extracted_gallery_filepaths=gallery_filepaths
    )
    print("Evaluation and visualization complete.")