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
    # convert image to numpy array
    image = image.numpy()

    image = transforms.ToTensor()(image)
    image = image.to(device)
    image = image.reshape(-1, config.IMAGE_C, config.IMAGE_H, config.IMAGE_W)

    model = NNModel()

    model = model.to(device)

    model.load_state_dict(torch.load("lglutide/models/model_4.pth"))

    # set the model to evaluation mode
    model.eval()

    # get the predictions
    with torch.no_grad():
        preds = model(image)

    # get the class with the highest probability
    preds = torch.argmax(preds, dim=1)

    # print the class
    if preds == 0:
        print("U")
    else:
        print("A")


# python main block
if __name__ == "__main__":
    img_path = input("Enter the image path: ")
    # img_path = "data/A/A1.jpeg"
    image = read_image(img_path)

    predict(image)
