from model import MainModel

from config import get_config
from dataset import SegmentationDataset, compute_class_weights

import matplotlib.pyplot as plt

from torchvision.utils import save_image, draw_segmentation_masks
from torchvision.transforms import Resize

import torch

def resize_segmenter_output(image, segmenter_output):
    bs, _, h, w = image.size()
    _, num_classes, h_diff, w_diff = segmenter_output.size()

    # Resize segmenter output to match the size of the input image
    resize = Resize((h, w))
    resized_segmenter_output = torch.empty(bs, num_classes, h, w)
    for i in range(bs):
        resized_segmenter_output[i] = resize(segmenter_output[i].unsqueeze(0)).squeeze(0)

    return resized_segmenter_output



if __name__ == '__main__':
    # Add model checkpoint here
    MODEL_CHECKPOINT = ''

    config = get_config()
    dataset = SegmentationDataset(config)

    main_model = MainModel(config)

    val_dataset = dataset.retrieve_val_data()

    model = MainModel.load_from_checkpoint(MODEL_CHECKPOINT, config=config)
    model.eval()

    resize = Resize(size=(572, 572))

    for x, _ in val_dataset:
        resized_segmenter_output = resize_segmenter_output(x, model.model(x.cuda()))
        image_uint8 = (x * 255).byte()

        for i in range(resized_segmenter_output.size(0)):
            resized_segmenter_output_i = resized_segmenter_output[i]
            image_uint8_i = image_uint8[i]

            segmented_output = torch.sigmoid(resized_segmenter_output_i) >= torch.sigmoid(resized_segmenter_output_i).max(dim=0).values - 1e-7

            segmented_img = draw_segmentation_masks(image_uint8_i.repeat(3, 1, 1), segmented_output)
            segmented_img = segmented_img.float()
            segmented_img /= segmented_img.max()

            save_image(segmented_img, 'image.jpg')
            exit()
