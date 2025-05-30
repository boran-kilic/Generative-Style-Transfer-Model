import argparse
from . import art_fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for computing activations.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads used for data loading.')
    parser.add_argument('--content_metric', type=str, default='alexnet', choices=['lpips', 'vgg', 'alexnet'], help='Content distance.')
    parser.add_argument('--mode', type=str, default='art_fid', choices=['art_fid', 'art_fid_inf','art_dbf'], help='Evaluate ArtFID, ArtFID_infinity, or ArtDBF.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--style_images', type=str, required=True, help='Path to style images.')
    parser.add_argument('--content_images', type=str, required=True, help='Path to content images.')
    parser.add_argument('--stylized_images', type=str, required=True, help='Path to stylized images.')
    args = parser.parse_args()

    print(f"DEBUG: Mode selected from arguments: {args.mode}")
    
    art_fid_value = art_fid.compute_art_fid(args.stylized_images,
                                            args.style_images,
                                            args.content_images,
                                            args.batch_size,
                                            args.device,
                                            args.mode,
                                            args.content_metric,
                                            args.num_workers)

    print('ArtDBF value:', art_fid_value[0])
    print('ArtFID Value:', art_fid_value[1])


if __name__ == '__main__':
    main()
