import os
import errno
import shutil
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MPEG7(Dataset):
    url = 'http://www.dabi.temple.edu/~shape/MPEG7/MPEG7dataset.zip'
    ds_folder = 'mpeg7'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, dimensions=(32, 32)):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resolution = dimensions

        fts_file, lbl_file = self._make_filenames()
        fts_file_tst, lbl_file_tst = self._make_filenames(is_test=True)
        if download:
            self.download(fts_file, lbl_file, fts_file_tst, lbl_file_tst)

        if not self._check_exists(fts_file, lbl_file, fts_file_tst, lbl_file_tst):
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.x, self.y = None, None
        if self.train:
            self.x = np.load(fts_file)
            self.y = np.load(lbl_file)
        else:
            self.x = np.load(fts_file_tst)
            self.y = np.load(lbl_file_tst)

    def __getitem__(self, index):
        # By default torchvision datasets return PIL image objects
        img = Image.fromarray(self.x[index], mode='L')
        target = self.y[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.x.shape[0]

    def _make_filenames(self, is_test: bool = False) -> Tuple[str, str]:
        extra = '_test' if is_test else ''
        features = f'processed_mpeg7_features_{self.resolution[0]}_{self.resolution[1]}{extra}.npy'
        labels = f'processed_mpeg7_labels_{self.resolution[0]}_{self.resolution[1]}{extra}.npy'

        # Adding the absolute part of the path
        features = os.path.join(self.root, MPEG7.ds_folder, features)
        labels = os.path.join(self.root, MPEG7.ds_folder, labels)

        return features, labels

    def download(self, fts_file, lbl_file, fts_file_tst, lbl_file_tst) -> None:
        from six.moves import urllib
        import glob

        if self._check_exists(fts_file, lbl_file, fts_file_tst, lbl_file_tst):
            return

        try:
            os.makedirs(os.path.join(self.root, MPEG7.ds_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading')
        data = urllib.request.urlopen(MPEG7.url)
        destfile = os.path.join(self.root, MPEG7.ds_folder, 'mpeg7.zip')
        with open(destfile, 'wb') as dfile:
            dfile.write(data.read())

        shutil.unpack_archive(destfile, os.path.join(self.root, MPEG7.ds_folder))

        os.unlink(destfile)
        os.unlink(os.path.join(self.root, MPEG7.ds_folder, 'original', 'confusions.gif'))
        os.unlink(os.path.join(self.root, MPEG7.ds_folder, 'original','shapedata.gif'))

        print('Processing')
        bunch_x_trn = list()
        bunch_y_trn = list()

        bunch_x_tst = list()
        bunch_y_tst = list()

        for img_filepath in glob.glob(os.path.join(self.root, MPEG7.ds_folder, 'original', '*.gif')):
            img = Image.open(img_filepath).resize(self.resolution, Image.BILINEAR)
            img = np.asarray(img, dtype=np.uint8)

            # .../bat-3.gif
            label = img_filepath.rsplit('/', maxsplit=1)[1]
            label, index = label.split('-')
            index = int(index.split('.')[0])

            # For each class, the 4 last samples are test, resulting in 80% train and 20% test
            if index > 16:
                bunch_x_tst.append(img)
                bunch_y_tst.append(label)
            else:
                bunch_x_trn.append(img)
                bunch_y_trn.append(label)

        labels_table = set(bunch_y_trn)
        label2idx = {name: idx for idx, name in enumerate(labels_table)}
        encoded_y_trn = [label2idx[label] for label in bunch_y_trn]
        encoded_y_tst = [label2idx[label] for label in bunch_y_tst]

        all_x_trn = np.asarray(bunch_x_trn)
        all_y_trn = np.asarray(encoded_y_trn, dtype=int)

        np.save(fts_file, all_x_trn)
        np.save(lbl_file, all_y_trn)

        all_x_tst = np.asarray(bunch_x_tst)
        all_y_tst = np.asarray(encoded_y_tst, dtype=int)

        np.save(fts_file_tst, all_x_tst)
        np.save(lbl_file_tst, all_y_tst)

    def _check_exists(self, fts_file: str, lbl_file: str, fts_file_tst: str, lbl_file_tst: str) -> bool:
        return os.path.exists(fts_file) and os.path.exists(lbl_file)\
               and os.path.exists(fts_file_tst) and os.path.exists(lbl_file_tst)


if __name__ == '__main__':
    mpeg7 = MPEG7(root='/Users/lzfelix/Desktop/papers/neo_evolving/data/', download=True, dimensions=(100, 100))