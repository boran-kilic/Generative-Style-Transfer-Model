import argparse
import glob
import numpy as np
import os
from PIL import Image
from scipy import linalg
from sklearn.linear_model import LinearRegression
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

from . import utils  # relative import from same package
from . import inception
from . import image_metrics


ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
CKPT_URL = 'https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth'


def custom_collate(batch):
    try:
        return default_collate(batch)
    except RuntimeError:
        # If tensors are different sizes, resize them to the same size
        max_h = max([img.shape[1] for img in batch])
        max_w = max([img.shape[2] for img in batch])
        resized_batch = []
        for img in batch:
            if img.shape[1] != max_h or img.shape[2] != max_w:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), 
                    size=(max_h, max_w), 
                    mode='bilinear', 
                    align_corners=False
                )[0]
            resized_batch.append(img)
        return torch.stack(resized_batch)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms if transforms is not None else ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transforms(img)
            return img
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            # Return a blank image of the expected size
            return torch.zeros((3, 299, 299))


def get_activations(files, model, batch_size=50, device='cpu', num_workers=1):
    """Computes the activations of for all images.

    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (): Activations of the images, shape [num_images, 2048].
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    transforms = Compose([
        Resize((299, 299), interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImagePathDataset(files, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )

    pred_arr = np.empty((len(files), 2048))
    start_idx = 0

    pbar = tqdm(total=len(files))
    for batch in dataloader:
        batch = batch.to(device)
        
        with torch.no_grad():
            features = model(batch, return_features=True)

        features = features.cpu().numpy()
        pred_arr[start_idx:start_idx + features.shape[0]] = features
        start_idx = start_idx + features.shape[0]

        pbar.update(batch.shape[0])

    pbar.close()
    return pred_arr


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    
    Args:
        mu1 (np.ndarray): Sample mean of activations of stylized images.
        mu2 (np.ndarray): Sample mean of activations of style images.
        sigma1 (np.ndarray): Covariance matrix of activations of stylized images.
        sigma2 (np.ndarray): Covariance matrix of activations of style images.
        eps (float): Epsilon for numerical stability.

    Returns:
        (float) FID value.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if np.abs(np.max(np.imag(covmean))) > 1e-3:
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m} is too large')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_activation_statistics(files, model, batch_size=50, device='cpu', num_workers=1):
    """Computes the activation statistics used by the FID.
    
    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (np.ndarray, np.ndarray): mean of activations, covariance of activations
        
    """
    act = get_activations(files, model, batch_size, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_image_paths(path, sort=False):
    """Returns the paths of the images in the specified directory, filtered by allowed file extensions.

    Args:
        path (str): Path to image directory.
        sort (bool): Sort paths alphanumerically.

    Returns:
        (list): List of image paths with allowed file extensions.

    """
    paths = []
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(path, f'*.{extension}')))
    if sort:
        paths.sort()
    return paths


def compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers=1):
    """Computes the FID for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    print('Computing FID...')
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    mu1, sigma1 = compute_activation_statistics(stylized_image_paths, model, batch_size, device, num_workers)
    mu2, sigma2 = compute_activation_statistics(style_image_paths, model, batch_size, device, num_workers)
    
    fid_value = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

def compute_kid(path_to_stylized, path_to_style, batch_size, device, num_workers=1):
    """Computes the Kernel Inception Distance (KID) for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for computing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) KID value.
    """
    print('Computing KID...')
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    # Get activations for both sets of images
    act1 = get_activations(stylized_image_paths, model, batch_size, device, num_workers)
    act2 = get_activations(style_image_paths, model, batch_size, device, num_workers)
    
    # Compute polynomial kernel
    n = act1.shape[0]
    m = act2.shape[0]
    
    # Center the features
    feat1 = act1 - np.mean(act1, axis=0)
    feat2 = act2 - np.mean(act2, axis=0)
    
    # Compute polynomial kernel (k(x,y) = (x^T y + 1)^3)
    kernel_xx = (np.matmul(feat1, feat1.T) / feat1.shape[1] + 1) ** 3
    kernel_yy = (np.matmul(feat2, feat2.T) / feat2.shape[1] + 1) ** 3
    kernel_xy = (np.matmul(feat1, feat2.T) / feat1.shape[1] + 1) ** 3
    
    # Compute KID
    kid_value = np.mean(kernel_xx) + np.mean(kernel_yy) - 2 * np.mean(kernel_xy)
    
    return float(kid_value)

def compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_points=15, num_workers=1):
    """Computes the FID infinity for the given paths."""
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    activations_stylized = get_activations(stylized_image_paths, model, batch_size, device, num_workers)
    activations_style = get_activations(style_image_paths, model, batch_size, device, num_workers)
    
    # Use the size of the smaller dataset
    min_size = min(len(activations_stylized), len(activations_style))
    activation_idcs = np.arange(min_size)

    fids = []
    
    # Adjust batch sizes based on available data
    start_size = min(50, min_size)  # Start with smaller batches
    stop_size = min_size  # Use all available images
    
    fid_batches = np.linspace(start=start_size, stop=stop_size, num=num_points).astype('int32')
    
    print('Computing FID_infinity...')
    print(f'Using batch sizes from {start_size} to {stop_size} with {num_points} points')

    for fid_batch_size in fid_batches:
        np.random.shuffle(activation_idcs)
        idcs = activation_idcs[:fid_batch_size]
        
        act_style_batch = activations_style[idcs]
        act_stylized_batch = activations_stylized[idcs]

        mu_style, sigma_style = np.mean(act_style_batch, axis=0), np.cov(act_style_batch, rowvar=False)
        mu_stylized, sigma_stylized = np.mean(act_stylized_batch, axis=0), np.cov(act_stylized_batch, rowvar=False)
        
        fid_value = compute_frechet_distance(mu_style, sigma_style, mu_stylized, sigma_stylized)
        fids.append(fid_value)

    fids = np.array(fids).reshape(-1, 1)
    reg = LinearRegression().fit(1 / fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    return fid_infinity


def compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric='lpips', device='cuda', num_workers=1):
    """Computes the distance for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) Content distance value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # Sort paths in order to match up the stylized images with the corresponding content image
    stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    content_image_paths = get_image_paths(path_to_content, sort=True)

    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'

    transforms = Compose([
        Resize((299, 299), interpolation=InterpolationMode.BILINEAR),
        ToTensor()
    ])
    
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=transforms)
    dataloader_stylized = torch.utils.data.DataLoader(
        dataset_stylized,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    dataset_content = ImagePathDataset(content_image_paths, transforms=transforms)
    dataloader_content = torch.utils.data.DataLoader(
        dataset_content,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )
    
    if content_metric == 'vgg' or content_metric == 'alexnet':
        metric = image_metrics.Metric(content_metric).to(device)
    elif content_metric == 'lpips':
        metric = image_metrics.LPIPS().to(device)
    else:
        raise ValueError(f'Invalid content metric: {content_metric}')

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_content.to(device))
            N += batch_dist.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N


def compute_art_fid(path_to_stylized, path_to_style, path_to_content, batch_size, device, mode, content_metric='lpips', num_workers=1):
    """Computes the ArtFID for the given paths.
    """
    print('Computing style metric...')
    if mode == 'art_fid_inf':
        print('MODE: FID_infinity')
        fid_value = compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_workers)
    elif mode == 'art_fid':
        print('MODE: FID')
        fid_value = compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers)
    elif mode == 'art_dbf':
        print('MODE: KID')
        kid_value = compute_kid(path_to_stylized, path_to_style, batch_size, device, num_workers)
        #fid_value = compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    print('Computing content distance...')
    content_dist = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric, device, num_workers)
    
    """ if 'VG' in str(path_to_content):
        print('Van Gogh Results with Tuned Parameters')
    elif 'JP' in str(path_to_content):
        print('Ukiyo-e Results with Tuned Parameters')
    elif 'Engraving' in str(path_to_content):
        print('Engraving Results with Tuned Parameters')
    else:
        print('Style: Unknown') """

    

    if mode == 'art_fid_inf':
        print(f'Content distance: {content_dist}')
        print(f'Style metric: FID_infinity={fid_value}')
    elif mode == 'art_fid':
        print(f'Content distance: {content_dist}')
        print(f'Style metric: FID={fid_value}')
    elif mode == 'art_dbf':
        print(f'Content distance: {content_dist}')
        print(f'Style metric: KID={kid_value}')
    else:
        raise ValueError(f'Invalid mode: {mode}')

    art_dbf_value = (content_dist + 1) * (kid_value + 1)
    #art_fid_value = (content_dist + 1) * (fid_value + 1)
    #return [float(art_dbf_value), float(art_fid_value)]
    return [float(art_dbf_value)]

