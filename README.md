# PAAEST

Pytorch implementation of [Progressive Artistic Aesthetic Enhancement For Chinese Ink Painting Style Transfer]

## Results

Some individuals have pointed out that there is an issue with the SSIM value in the experimental results of the article. We sincerely acknowledge this error and deeply apologize. In fact, the SSIM metric was manually implemented by us, and on line 26 of the utils, we incorrectly computed the mu1_mu2 value, which caused the SSIM values to exceed the expected range.


