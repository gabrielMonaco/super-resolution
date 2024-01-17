import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE


class DIV2K:
    def __init__(self,
                 scale=2,
                 subset='train',
                 #downgrade=None, ### modifiquei de bicubic para None
                 images_dir='C:\\Users\\bielm\\Documents\\Mestrado\\0_SuperResolucao\\Dados_SR\\',
                 caches_dir='C:\\Users\\bielm\\Documents\\Mestrado\\0_SuperResolucao\\super-resolution\\super-resolution\\cache\\'):

        self._ntire_2018 = False ### de True para False

        _scales = [2, 4]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError(f'scale must be in ${_scales}')

        if subset == 'train':
            self.image_ids = range(0, -17) ### intervalo de train_img
        elif subset == 'valid':
            self.image_ids = range(-16, -1) ### intervalo de val_img
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        """
        Creates a dataset by combining the low-resolution dataset (lr_dataset) and the high-resolution dataset (hr_dataset).
        
        Args:
            batch_size (int): The batch size for the dataset. Default is 16.
            repeat_count (int): The number of times to repeat the dataset. If None, the dataset will be repeated indefinitely. Default is None.
            random_transform (bool): Whether to apply random transformations to the image pairs. Default is True.
        
        Returns:
            tf.data.Dataset: The combined dataset with optional random transformations, batched and repeated.
        """
        # Combine the low-resolution dataset and the high-resolution dataset
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        
        # Apply random transformations to the image pairs if random_transform is True
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        
        # Batch the dataset
        ds = ds.batch(batch_size)
        
        # Repeat the dataset for repeat_count times
        ds = ds.repeat(repeat_count)
        
        # Prefetch the dataset for faster processing
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        
        return ds


    def hr_dataset(self):
        """
        Retorna o conjunto de dados de alta resolução.

        Verifica se o diretório de imagens de alta resolução existe. Se não existir,
        faz o download do arquivo compactado contendo as imagens e extrai-o.

        Carrega o conjunto de dados de imagens de alta resolução e armazena em cache.

        Se o índice de cache não existir, popula o cache com as imagens do conjunto de dados.

        Retorna o conjunto de dados de alta resolução.

        Returns:
            tf.data.Dataset: O conjunto de dados de alta resolução.
        """
        if not os.path.exists(self._hr_images_dir()):
            download_archive(self._hr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())

        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self):
            """
            Returns a dataset of low-resolution images.

            If the directory containing the low-resolution images does not exist, it will be downloaded and extracted.
            The dataset is then cached for faster access.

            Returns:
                tf.data.Dataset: A dataset of low-resolution images.
            """
            if not os.path.exists(self._lr_images_dir()):
                download_archive(self._lr_images_archive(), self.images_dir, extract=True)

            ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())

            if not os.path.exists(self._lr_cache_index()):
                self._populate_cache(ds, self._lr_cache_file())

            return ds

    def _hr_cache_file(self):
        """
        Returns the file path for the high-resolution cache file.

        Returns:
            str: The file path for the high-resolution cache file.
        """
        return os.path.join(self.caches_dir, f'{self.subset}_HR.cache')  ### 

    def _lr_cache_file(self):
            """
            Returns the file path for the low-resolution cache file.

            Returns:
                str: The file path for the low-resolution cache file.
            """
            return os.path.join(self.caches_dir, f'{self.subset}_LR.cache')

    def _hr_cache_index(self):
        """
        Returns the file path of the index file for the high-resolution cache.
        The index file is used to store metadata about the cached high-resolution images.
        """
        return f'{self._hr_cache_file()}.index'

    def _lr_cache_index(self):
        return f'{self._lr_cache_file()}.index'

    def _hr_image_files(self):
        """
        Returns a list of high-resolution image file paths.

        Returns:
            List[str]: A list of file paths for high-resolution images.
        """
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, f'{image_id:04}.png') for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, self._lr_image_file(image_id)) for image_id in self.image_ids]

    def _lr_image_file(self, image_id):
            """
            Returns the filename of the low-resolution image corresponding to the given image_id.

            Parameters:
                image_id (int): The ID of the image.

            Returns:
                str: The filename of the low-resolution image.
            """
            if not self._ntire_2018 or self.scale == 8:
                return f'{image_id:04}x{self.scale}.png'
            else:
                return f'{image_id:04}x{self.scale}_LR.png'

    def _hr_images_dir(self):
            """
            Returns the directory path for high-resolution images based on the subset.

            Returns:
                str: The directory path for high-resolution images.
            """
            return os.path.join(self.images_dir, f'PNG_MR\\') ### modifiquei de os.path.join(self.images_dir, f'DIV2K_{self.subset}_HR

    def _lr_images_dir(self):
            """
            Returns the directory path for low-resolution images.

            If self._ntire_2018 is True, it returns the path for 1xLR images.
            Otherwise, it returns the path for DIV2K_LR images with the specified subset, downgrade, and scale.

            Returns:
                str: The directory path for low-resolution images.
            """
            if self._ntire_2018:
                return os.path.join(self.images_dir, f'PNG_LR\\') ### modifiquei de os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}') para 1xLR
            else:
                return os.path.join(self.images_dir, f'PNG_LR\\') ### modifiquei de os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}', f'X{self.scale}') para 1xLR

    def _hr_images_archive(self):
            """
            Returns the filename of the high-resolution images archive.

            Returns:
                str: The filename of the high-resolution images archive.
            """
            return f'2xHR.zip' ### modifiquei de f'DIV2K_{self.subset}_HR.zip' para 2xHR

    def _lr_images_archive(self):
            """
            Returns the name of the low-resolution images archive file.

            If self._ntire_2018 is True, it returns '1xLR.zip'.
            Otherwise, it returns '1xLR.zip'.

            Returns:
                str: The name of the low-resolution images archive file.
            """
            if self._ntire_2018:
                return f'1xLR.zip'
            else:
                return f'1xLR.zip'
            
    def _lr_images_archive(self):
        if self._ntire_2018:
            return f'1xLR.zip' ### modifiquei de f'DIV2K_{self.subset}_LR_{self.downgrade}.zip' para 1xLR
        else:
            return f'1xLR.zip' ### modifiquei de f'DIV2K_{self.subset}_LR_{self.downgrade}_X{self.scale}.zip' para 1xLR

    @staticmethod
    def _images_dataset(image_files):
        """
        Create a TensorFlow dataset from a list of image files.

        Args:
            image_files (list): List of image file paths.

        Returns:
            tf.data.Dataset: TensorFlow dataset containing the images.
        """
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=1), num_parallel_calls=AUTOTUNE) #### 
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        """
        Populates the cache with decoded images from the given dataset.

        Args:
            ds (Dataset): The dataset containing the images.
            cache_file (str): The file path to store the cached images.

        Returns:
            None
        """
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------


def random_crop(lr_img, hr_img, hr_crop_size=256, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


# -----------------------------------------------------------
#  IO
# -----------------------------------------------------------


def download_archive(file, target_dir, extract=True):
    source_url = f'Dados_SR\\{file}'
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))
