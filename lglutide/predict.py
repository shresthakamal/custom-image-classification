import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor

from lglutide import config
from lglutide.nn import NNModel


def predict(image):
    # set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = transforms.Resize((config.IMAGE_W, config.IMAGE_H))(image)
    image = image.numpy()
    image = transforms.ToTensor()(image)
    image = image.reshape(-1, config.IMAGE_C, config.IMAGE_H, config.IMAGE_W)

    model = NNModel()

    image = image.to(device)
    model = model.to(device)

    model.load_state_dict(torch.load(config.INFERENCE_MODEL))
    # set the model to evaluation mode
    model.eval()

    # get the predictions
    with torch.no_grad():
        preds = model(image)

    print(preds)

    # convert the predictions to probabilities
    probs = torch.softmax(preds, dim=1)

    # detach the tensor from the graph
    probs = probs.detach().cpu().numpy()[0]

    return probs


def check_if_model_exists():
    try:
        with open(config.INFERENCE_MODEL, "rb") as f:
            print("Model found. Loading model...")
    except FileNotFoundError:
        if config.download_model:
            import gdown

            # download the pretrained model from google drive
            print("Model not found. Downloading model...")
            gdown.download(
                "https://drive.google.com/uc?id=1Y8W5YV0vJZu5wZoJ8Zu5E6jv9e9X7B9O",
                config.INFERENCE_MODEL,
                quiet=False,
            )
        else:
            print(
                "Model not found. Please train your model or download the pretrained model."
            )


# python main block
if __name__ == "__main__":
    check_if_model_exists()

    img_path = input("Enter the image path: ")
    image = read_image(img_path)

    print(image, image.shape, image.dtype)

    predict(image)
