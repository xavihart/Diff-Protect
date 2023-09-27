import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from PIL import Image

# Load the CLIP model
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)

preprocess = CLIPImageProcessor.from_pretrained(model_ID)

# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)

    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")

    # Return the preprocessed image
    return image


def clip_sim(a, b):
    

    # Load the two images and preprocess them for CLIP
    image_a = load_and_preprocess_image(a)["pixel_values"]
    image_b = load_and_preprocess_image(b)["pixel_values"]

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)

    return similarity_score.item()

if __name__ == '__main__':
    score = clip_sim('/ssdscratch/hxue45/data/phd_2/diff_mist/out_fid/advdm_eps16_steps100_gmode-/anime/0.1/0.png', 
             '/ssdscratch/hxue45/data/phd_2/diff_mist/out_fid/advdm_eps16_steps100_gmode-/anime/0.1/0.png')
    print(score)