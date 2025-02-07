import warnings

warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
from model import Model

def main():
    parser = argparse.ArgumentParser(description='PAAEST by Pytorch')
    parser.add_argument('--content_folder', '-cf', type=str, default='./test_content',
                        help='Content images folder path')
    parser.add_argument('--style_folder', '-sf', type=str, default='./test_style',
                        help='Style images folder path')
    parser.add_argument('--output_folder', '-of', type=str, default='./output',
                        help='Output folder path for generated images')
    parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta control the degree of WCT')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_state_path', type=str, default='./result/model_state/100_epoch.pth',
                        help='Path to the model state file')

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'transfer1'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'transfer2'), exist_ok=True)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    model = Model().to(device)
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=device))
    model.eval()

    dataset = PreprocessDataset(args.content_folder, args.style_folder)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (content, style) in enumerate(data_loader):
        content = content.to(device)
        style = style.to(device)

        with torch.no_grad():
            out = model.generate(content, style, args.alpha, args.beta)

        out = denorm(out, device)

        with torch.no_grad():
            out1 = model.generate(out, content, args.alpha, args.beta)

        out1 = denorm(out1, device)

        content_name = os.path.splitext(os.path.basename(dataset.images_pairs[i][0]))[0]
        style_name = os.path.splitext(os.path.basename(dataset.images_pairs[i][1]))[0]
        if content_name == style_name:
            output_name = str(content_name)
        else:
            output_name = f'{content_name}_{style_name}'

        save_image(out, os.path.join(args.output_folder, 'transfer1', f'{output_name}.jpg'), nrow=1)
        save_image(out1, os.path.join(args.output_folder, 'transfer2', f'{output_name}.jpg'), nrow=1)

        print(f'Result saved into file: {output_name}.jpg')


if __name__ == '__main__':
    main()
