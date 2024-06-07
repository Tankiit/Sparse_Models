import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from captum.attr import IntegratedGradients
import torch.nn.functional as F

import pandas as pd
import pdb
import seaborn as sns
import numpy as np

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import torchvision.transforms as transforms

# Prepare a transformation to convert the images to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the size expected by CLIP
    transforms.ToTensor(),  # Convert to PyTorch tensor
])

# Prepare sample images and text descriptions
image_paths = [
    "/Users/tanmoymukherjee/research/data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg",
    "/Users/tanmoymukherjee/research/data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.jpg",
    "/Users/tanmoymukherjee/research/data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg",
]
test_images=[Image.open(path) for path in image_paths]
# Load the images
images = [transform(Image.open(path)) for path in image_paths]

# Convert the list of image tensors into a single tensor
images = torch.stack(images)

# Extract image features using CLIP
image_features = model.get_image_features(images)

# text_descriptions = [
#     "A black-footed albatross",
#     "A black-footed albatross in water",
#     "A black-footed albatross flying",
# ]

attributes=pd.read_csv("/Users/tanmoymukherjee/research/data/CUB_200_2011/attributes/attributes.txt", sep=" ", header=None)
text_descriptions = attributes.iloc[:,1].values.tolist()


# Tokenize, truncate, and pad the text descriptions
inputs = processor(text=text_descriptions, truncation=True, padding='max_length', max_length=64, return_tensors="pt")

# Extract text features using CLIP
text_features = model.get_text_features(inputs["input_ids"])# Compute the similarity matrix S
S = torch.matmul(image_features, text_features.T)

# Initialize the learnable matrix W
N = len(image_paths)
W = nn.Parameter(torch.randn(N, N))



# Define the temperature for Gumbel-Softmax
temperature = 0.5

def gumbel_sinkhorn(W, temperature, num_iterations):
     gumbel_noise = -torch.log(-torch.log(torch.rand_like(W)))
     W_noisy = W + gumbel_noise
     W_soft = nn.functional.softmax(W_noisy / temperature, dim=-1)
     for _ in range(num_iterations):
         W_soft = W_soft / W_soft.sum(dim=0, keepdim=True)
         W_soft = W_soft / W_soft.sum(dim=1, keepdim=True)
     return W_soft


# Apply Gumbel-Softmax to the similarity matrix
gumbel_sinkhorn_scores = gumbel_sinkhorn(S, temperature=temperature, num_iterations=10)

# Select the top-k most relevant concepts for each test image
k = 5  # Number of concepts to select
_, concept_indices = torch.topk(gumbel_sinkhorn_scores, k, dim=1)
selected_concepts = text_features[concept_indices]

# Select one test image
test_image = test_images[0]
concept_indices = concept_indices[0]

# Display the test image
plt.figure(figsize=(5, 5))
plt.imshow(test_image)
plt.title("Test Image 1")
plt.axis('off')
plt.show()

# Display the selected concepts
selected_concept_names = [text_descriptions[idx] for idx in concept_indices]
print("Selected Concepts:")
for name in selected_concept_names:
    print(name)

# Display the original concept matrix
original_concept_matrix = S.detach().numpy()
# Select the top 10 concepts from the original concept matrix
top_10_concepts = original_concept_matrix[:10]

# Get the scores for the selected concepts and the rest of the concepts
selected_concept_scores = gumbel_sinkhorn_scores[0, concept_indices].detach().numpy()
other_concept_scores = np.delete(gumbel_sinkhorn_scores[0].detach().numpy(), concept_indices)

# Create labels for the selected concepts and the rest of the concepts
selected_concept_labels = [text_descriptions[idx] for idx in concept_indices]
other_concept_labels = [text_descriptions[i] for i in range(len(text_descriptions)) if i not in concept_indices]

# Combine the scores and labels
scores = np.concatenate([selected_concept_scores, other_concept_scores])
labels = selected_concept_labels + other_concept_labels

# Create a DataFrame from the scores and labels
df = pd.DataFrame({'Concept': labels, 'Score': scores})

# Sort the DataFrame by score
df = df.sort_values('Score', ascending=False).head(10)

# Plot the scores
plt.figure(figsize=(10, 5))
sns.barplot(x='Score', y='Concept', data=df, orient='h')
plt.title('Top 10 Concepts')
plt.xlabel('Score')
plt.ylabel('Concept')
plt.show()
# num_images = len(test_images)
# num_cols = 4
# num_rows = (num_images + num_cols - 1) // num_cols

# fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))


# for i, (test_image, concept_indices) in enumerate(zip(test_images, top_k_indices)):
#     row = i // num_cols
#     col = i % num_cols

#     axes[row, col].imshow(test_image)
#     axes[row, col].set_title(f"Test Image {i+1}")
#     axes[row, col].axis('off')
    
#     selected_concept_names = [concept_names[idx] for idx in concept_indices]
#     axes[row, col].text(0.5, -0.1, "\n".join(selected_concept_names),
#                         transform=axes[row, col].transAxes, ha='center', va='top', fontsize=8)

# plt.tight_layout()
# plt.show()