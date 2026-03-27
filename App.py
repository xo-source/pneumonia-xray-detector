import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torchxrayvision as xrv

#pneumonia Xray Ai detector

# -------------------------------
# 1 Configurations
# -------------------------------
DATA_DIR = "C:\\Users\\promo\\Downloads\\chest_xray"
# ^^^ parent folder containing train/ and val/ for training

TEST_IMAGE = "normtest4.jpeg"
TEST_DIR =r"C:\Users\promo\Downloads\test\maintest"
# ^^^ batch xray folder
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2  # NORMAL / BACTERIAL

# -------------------------------
# 2 Prepare data loaders
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # converts to [0,1]
    transforms.Lambda(lambda x: x * 2048.0 - 1024.0)
    # converts 0->-1024, 1->1024
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------------
# 3️ Load pretrained DenseNet121
# -------------------------------

model = xrv.models.DenseNet(weights="densenet121-res224-chex")

# -------------------------------
# 3️.5 loadstate
# -------------------------------
model.load_state_dict(torch.load("model2\\xray_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -------------------------------
# 4 Training loop (fine-tuning)
# -------------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)
#
# for epoch in range(EPOCHS):
#     print(f"Epoch {epoch} Started...")
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     # Validates if training model is correct or not.
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     acc = correct / total
#     print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss:.2f}, Val Accuracy: {acc * 100:.2f}%")
#
#
# -------------------------------
# 4.5 Save/adjust the trained model per epoch
# -------------------------------
# torch.save(model.state_dict(), "model/xray_model.pth")
# print("Model saved as xray_model.pth")



# -------------------------------
# 5 Preprocess single image
# -------------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0  # scale to 0-1
    img_np = img_np * 2048.0 - 1024.0                 # match train scaling
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img_tensor


# -------------------------------
# 6 Predict function
# -------------------------------
def predict_xray(image_path):
    model.eval()
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        # outputs shape may be (1,14), pick first NUM_CLASSES
        logits = outputs[0][:NUM_CLASSES]
        probs = F.softmax(logits, dim=0)
    classes = train_dataset.classes  # ['BACTERIAL', 'NORMAL']
    results = {cls: float(prob) for cls, prob in zip(classes, probs)}
    return results


# -------------------------------
# 7.1 Run prediction batch folder
# -------------------------------
# results = predict_xray(TEST_IMAGE)
# top_prediction = max(results.items(), key=lambda x: x[1])
#
# print("Prediction:")
# x_label = 'PNEUMONIA'
# y_label= 'NORMAL'
#
# for filename in os.listdir(TEST_DIR):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(TEST_DIR, filename)
#         results = predict_xray(image_path)
#         top_prediction = max(results.items(), key=lambda x: x[1])
#         comp_class = y_label if top_prediction[0] == x_label else x_label
#
#     print("-" *40)
#     print(f"\nImage: {filename}")
#
#     if top_prediction[0] == 'PNEUMONIA':
#         print(f"{top_prediction[0]}: {top_prediction[1]*100:.2f}%")
#         print(f"{y_label}: {100-top_prediction[1]*100:.2f}%")
#
#     else:
#         print(f"{top_prediction[0]}: {top_prediction[1] * 100:.2f}%")
#         print(f"{x_label}: {100 - top_prediction[1] * 100:.2f}%")
#


# -------------------------------
# 7.2 Run prediction App
# -------------------------------
import gradio as gr

# Labels for display
x_label = 'PNEUMONIA'
y_label = 'NORMAL'

# Prediction wrapper for Gradio
def gr_predict(image):
    # image comes in as a PIL image from Gradio
    # save temporarily to pass to your function
    img_path = "temp_image.jpeg"
    image.save(img_path)

    results = predict_xray(img_path)
    top_prediction = max(results.items(), key=lambda x: x[1])
    comp_class = y_label if top_prediction[0] == x_label else x_label

    #formatted string
    output_text = (
        f"{top_prediction[0]}: {top_prediction[1] * 100:.2f}%\n"
        f"{comp_class}: {100 - top_prediction[1] * 100:.2f}%"
    )
    return output_text


# Build Gradio interface
iface = gr.Interface(
    fn=gr_predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="X-ray Pneumonia Detector",
    description="Upload a chest X-ray image and get a prediction for Pneumonia or Normal."
)

iface.launch()






