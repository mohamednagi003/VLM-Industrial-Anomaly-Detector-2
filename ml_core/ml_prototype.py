import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# --- Configuration ---
# You can change the model, but 'base-patch32' is a good starting point.
MODEL_NAME = "openai/clip-vit-base-patch32"
# Name of the test image (replace with your actual file path/name)
TEST_IMAGE_PATH = "industrial_test_image.jpg"

def load_vlm():
    """Loads the pre-trained VLM and its processor."""
    print(f"Loading VLM: {MODEL_NAME}...")
    # Use 'cuda' if GPU is available (Colab with GPU enabled)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model and processor from the Hugging Face model hub
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    return model, processor, device

def get_similarity_scores(model, processor, device, image_path: str, normal_prompt: str, anomaly_prompt: str):
    """
    Calculates the VLM similarity score between the image and the text prompts.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}. Using a mock image.")
        # Fallback to a blank image if file is missing (for basic testing structure)
        image = Image.new('RGB', (256, 256), color = 'red')
    else:
        # Load the image
        image = Image.open(image_path)

    # 1. Tokenize the text prompts
    texts = [normal_prompt, anomaly_prompt]
    
    # 2. Process the image and text together
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # Move tensors to the correct device (GPU/CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 3. Get the model output (logits are the raw similarity scores)
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Extract the image-text similarity scores
    # The output is a matrix of (ImageCount x TextPromptCount). We have 1x2.
    logits_per_image = outputs.logits_per_image  # This is the (1, 2) tensor

    # 5. Convert logits to probabilities (optional, but makes scores easier to read/compare)
    probs = logits_per_image.softmax(dim=1)
    
    score_normal = probs[0, 0].item()
    score_anomaly = probs[0, 1].item()

    return score_normal, score_anomaly

# --- Main Execution for Testing ---
if __name__ == "__main__":
    # Define your industrial conditions (Week 1 requirement)
    NORMAL_CONDITION = "The hydraulic press is operating normally without sound."
    ANOMALY_CONDITION = "There is smoke coming from the machine and a red light is flashing."
    
    try:
        model, processor, device = load_vlm()
        print(f"Model successfully loaded onto {device}.")

        normal_score, anomaly_score = get_similarity_scores(
            model, processor, device,
            TEST_IMAGE_PATH,
            NORMAL_CONDITION,
            ANOMALY_CONDITION
        )
        
        print("\n--- ML Prototype Results (Single Image) ---")
        print(f"Image Path: {TEST_IMAGE_PATH}")
        print(f"Normal Condition: {NORMAL_CONDITION}")
        print(f"Anomaly Condition: {ANOMALY_CONDITION}")
        print(f"\nSimilarity to Normal: {normal_score:.4f}")
        print(f"Similarity to Anomaly: {anomaly_score:.4f}")
        print("\nInterpretation: The higher the score, the more similar the image is to the prompt.")

    except Exception as e:
        print(f"An error occurred during VLM execution: {e}")
        print("Ensure you are running with a GPU runtime in Colab and have installed all dependencies.")
