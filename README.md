# PAAEST

Pytorch implementation of Progressive Artistic Aesthetic Enhancement For Chinese Ink Painting Style Transfer (ECAI2024) By Chihan Huang.

# Structure

```
project-root/
├── models/             # model component code folder
│   ├── WCT.py          # WCT
│   ├── attention.py    # attention mechanism
│   ├── decoder.py      # decoder structure with additional input
│   └── encoder.py      # encoder structure with attention mechanism
├── train_content/      # train content imgae folder
│   └── reminder        # place to put your training content image
├── train_style/        # train style image folder
│   └── reminder        # place to put your training style image
├── dataset.py          # instruction on how to process the image data
├── evaluate.py         # evaluate the test output images to get the results
├── model.py            # model utilizing the model components
├── split_train_test.py # split the data into train and test folder
├── README.md           # project instructions
├── test.py             # test the trained model to get output images
├── train.py            # train the model to get loss curve and checkpoints
└── utils.py            # utils and how ssim is wrongly calculated
```

# Requirements

- PyTorch
- Torchvision
- Pillow
- Skimage
- Tqdm
- Matplotlib
- Torch-fidelity


# Use

1. You can get the style data [here](https://theme.npm.edu.tw/opendata/) and split them into the folder train_content and test_content at 8:2, and content data [here](https://github.com/koishi70/Landscape-Dataset) and split them into the folder train_style and test_style at 8:2.

2. You can run the following command to train the model, and you can adjust the batch size, learning rate, gpu, etc of your custom model here.

```Python
python train.py --batch_size 8 --epoch 100 --learning_rate 5e-5 --train_content_dir './train_content' --train_style_dir './train_style' --save_dir './result'
```

3. You can run the following command to generate the test images into ./output. The transfer1 folder contains stylized images, and the transfer2 folder contains the regenerated content images using stylized images as content images. The FID and KID are calculated between content images and stylized images, and PSNR and SSIM are calculated between content images and regenerated content images.

```Python
python test.py -cf './test_content' -sf './test_style' -of './output'
```

4. Then run the following command to get the FID, KID, PSNR, SSIM results.

```Python
python evaluate.py
```

# Correct Value

Some individuals have pointed out that there is an issue with the SSIM value in the experimental results of the article. We sincerely acknowledge this error and deeply apologize. In fact, the SSIM metric was manually implemented by us, and on line 26 of the utils, we incorrectly computed the mu1_mu2 value, which caused the SSIM values to exceed the expected range.

The SSIM value is not calculated between the original content image and the stylized content image, because it make no sense as style transfer means to transform the iamge. When content image is transfered to the stylized image, the stylized image further transfers to the original content style, and the SSIM value is calculated between the original content image and the twice transfered image. Whereas, FID is calculated between the content image and the stylized images, so it explains why FID is relatively poor and SSIM is relatively good.

| Model | SSIM |
| ------ | ------ |
| ChipGAN | 0.687 |
| AnimeGAN | 0.703 |
| QS-Attn | 0.710 |
| PAAEST | 0.776 |

# Reference

```
@inproceedings{
      huang2024progressive,
      title={Progressive Artistic Aesthetic Enhancement For Chinese Ink Painting Style Transfer},
      author={Chihan Huang},
      booktitle={European Conference on Artificial Intelligence},
      year={2024},
      doi={10.3233/FAIA240557}
}
```
