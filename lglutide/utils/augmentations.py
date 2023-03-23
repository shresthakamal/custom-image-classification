import glob

import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def augmentations(a_path, u_path):
    # print starting of augmentation
    print("Augmenting images...")

    a_images = glob.glob(a_path + "*.jpeg")
    u_images = glob.glob(u_path + "*.jpeg")

    # print number of images before augmentation
    print(f"Number of images before augmentation: {len(a_images) + len(u_images)}")

    for images in [a_images, u_images]:
        for img_path in tqdm(images):
            orig_img = Image.open(img_path)

            padded_imgs = [
                transforms.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)
            ]

            gray_img = transforms.Grayscale()(orig_img)

            jitter = transforms.ColorJitter(brightness=0.5, hue=0.3)
            jitted_imgs = [jitter(orig_img) for _ in range(4)]

            blurrer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            blurred_imgs = [blurrer(orig_img) for _ in range(4)]

            perspective_transformer = transforms.RandomPerspective(
                distortion_scale=0.6, p=1.0
            )
            perspective_imgs = [perspective_transformer(orig_img) for _ in range(4)]

            rotater = transforms.RandomRotation(degrees=(0, 180))
            rotated_imgs = [rotater(orig_img) for _ in range(4)]

            affine_transfomer = transforms.RandomAffine(
                degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
            )
            affine_imgs = [affine_transfomer(orig_img) for _ in range(4)]

            elastic_transformer = transforms.ElasticTransform(alpha=250.0)
            transformed_imgs = [elastic_transformer(orig_img) for _ in range(2)]

            inverter = transforms.RandomInvert()
            invertered_imgs = [inverter(orig_img) for _ in range(4)]

            posterizer = transforms.RandomPosterize(bits=2)
            posterized_imgs = [posterizer(orig_img) for _ in range(4)]

            solarizer = transforms.RandomSolarize(threshold=192.0)
            solarized_imgs = [solarizer(orig_img) for _ in range(4)]

            sharpness_adjuster = transforms.RandomAdjustSharpness(sharpness_factor=2)
            sharpened_imgs = [sharpness_adjuster(orig_img) for _ in range(4)]

            autocontraster = transforms.RandomAutocontrast()
            autocontrasted_imgs = [autocontraster(orig_img) for _ in range(4)]

            equalizer = transforms.RandomEqualize()
            equalized_imgs = [equalizer(orig_img) for _ in range(4)]

            policies = [
                transforms.AutoAugmentPolicy.CIFAR10,
                transforms.AutoAugmentPolicy.IMAGENET,
                transforms.AutoAugmentPolicy.SVHN,
            ]
            augmenters = [transforms.AutoAugment(policy) for policy in policies]
            policy_imgs = [
                [augmenter(orig_img) for _ in range(4)] for augmenter in augmenters
            ]

            augmenter = transforms.RandAugment()
            augment_imgs = [augmenter(orig_img) for _ in range(4)]

            augmenter = transforms.TrivialAugmentWide()
            trivialaugment_imgs = [augmenter(orig_img) for _ in range(4)]

            augmenter = transforms.AugMix()
            augmix_imgs = [augmenter(orig_img) for _ in range(4)]

            hflipper = transforms.RandomHorizontalFlip(p=0.5)
            hflip_imgs = [hflipper(orig_img) for _ in range(4)]

            vflipper = transforms.RandomVerticalFlip(p=0.5)
            vflip_imgs = [vflipper(orig_img) for _ in range(4)]

            # combine all the augmented images into a single list
            augmented_imgs = (
                padded_imgs
                + [gray_img]
                + jitted_imgs
                + blurred_imgs
                + perspective_imgs
                + rotated_imgs
                + affine_imgs
                + transformed_imgs
                + invertered_imgs
                + posterized_imgs
                + solarized_imgs
                + sharpened_imgs
                + autocontrasted_imgs
                + equalized_imgs
                + [img for imgs in policy_imgs for img in imgs]
                + augment_imgs
                + trivialaugment_imgs
                + augmix_imgs
                + hflip_imgs
                + vflip_imgs
            )

            # save padded images to a folder
            for i, augmented_img in enumerate(augmented_imgs):
                augmented_img.save(img_path[:-5] + f"_aug_{i}.jpeg")

    # print number of images after augmentation
    a_images = glob.glob(a_path + "*.jpeg")
    u_images = glob.glob(u_path + "*.jpeg")
    print(f"Number of images after augmentation: {len(a_images) + len(u_images)}")

    # print augmentation complete
    print("Augmentation complete!")
