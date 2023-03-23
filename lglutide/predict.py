import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor

from lglutide import config
from lglutide.architectures.densenet import DenseNet121


def predict(image):
    check_if_model_exists()

    # set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = transforms.Resize((config.IMAGE_W, config.IMAGE_H))(image)
    image = image.numpy()
    image = transforms.ToTensor()(image)
    image = image.reshape(-1, config.IMAGE_C, config.IMAGE_H, config.IMAGE_W)

    model = DenseNet121(num_classes=2, grayscale=False)
    model = model.to(device)
    model.load_state_dict(torch.load(config.INFERENCE_MODEL))

    image = image.to(device)

    # set the model to evaluation mode
    model.eval()

    # get the predictions
    with torch.no_grad():
        probas = model(image)
    print(probas)
    # detach the tensor from the graph
    probas = probas.detach().cpu().numpy()[0]

    return probas


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

    # load the model in memory and get the predictions

    img_path = input("Enter the image path: ")
    image = read_image(img_path)

    print(image, image.shape, image.dtype)

    predict(image)
