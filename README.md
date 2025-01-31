# PAAEST

Pytorch implementation of [Progressive Artistic Aesthetic Enhancement For Chinese Ink Painting Style Transfer]

## Results

Some individuals have pointed out that there is an issue with the SSIM value in the experimental results of the article. We sincerely acknowledge this error and deeply apologize. In fact, the SSIM metric was manually implemented by us, and on line 26 of the utils, we incorrectly computed the mu1_mu2 value, which caused the SSIM values to exceed the expected range.

# Use

You can run the following command to train the model:

```Python
python train.py 
```

You can run the following command to test the model:

```Python
python test.py -c 'path/to/your/content/image.jpg' -s 'path/to your/style/image/jpg' -o 'output/path'
```

# Correct Value

The SSIM value is not calculated between the original content image and the stylized content image, because it make no sense as style transfer means to transform the iamge. When content image is transfered to the stylized image, the stylized image further transfers to the original content style, and the SSIM value is calculated between the original content image and the twice transfered image. Whereas, FID is calculated between the content image and the stylized images, so it explains why FID is relatively poor and SSIM is relatively good.

| Model | SSIM |
| ------ | ------ |
| ChipGAN | 0.891 |
| AnimeGAN | 0.913 |
| QS-Attn | 0.916 |
| PAAEST | 0.928 |

