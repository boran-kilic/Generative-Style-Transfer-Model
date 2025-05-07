import os
import shutil

def get_image_files(directory):
    allowed_extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    return [f for f in os.listdir(directory) if f.split('.')[-1] in allowed_extensions]

def copy_first_n_images(src_dir, dest_dir, n):
    os.makedirs(dest_dir, exist_ok=True)
    files = get_image_files(src_dir)[:n]
    for file in files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

def main():
    style_dir = 'train/style_train/'
    content_dir = 'train/content_train/'
    stylized_dir = 'generated_images/'

    style_files = get_image_files(style_dir)
    content_files = get_image_files(content_dir)
    stylized_files = get_image_files(stylized_dir)

    n = min(len(style_files), len(content_files), len(stylized_files))

    copy_first_n_images(style_dir, 'style1/', n)
    copy_first_n_images(content_dir, 'content1/', n)
    copy_first_n_images(stylized_dir, 'stylized1/', n)

if __name__ == '__main__':
    main()