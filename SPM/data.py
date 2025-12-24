import os
import numpy as np 
import cv2
import utils as ut
import shutil


class CustomDataset:
   
    def __init__(self, root, resize_h = 300, resize_w = 300):
        
        
        self.root = ut.test_postfix_dir(root)
        # Filter out .DS_Store and other non-directory files
        self.clsses = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)) and not d.startswith('.')]
        self.clsses.sort()
        print(f"DEBUG: Found classes in {self.root}: {self.clsses}")
        self.clss2idx = dict([(clss, i) for i, clss in enumerate(self.clsses)])

        self.img_names = []
        self.num_clss = len(self.clsses)
        print(f"Total number of classes: {self.num_clss}")
        self.size = 0
        self.resize_h = resize_h
        self.resize_w = resize_w
        # load name of every picture in each class
        self.each_clss_size = np.zeros((self.num_clss,), dtype = int)
        for i, clss in enumerate(self.clsses):
            class_dir = os.path.join(self.root, clss)
            print(class_dir)
            if not os.path.isdir(class_dir):
                 continue
            
            # Filter out .DS_Store from images and ensure valid image extensions
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            pics = [p for p in os.listdir(class_dir) 
                    if not p.startswith('.') and os.path.splitext(p)[1].lower() in valid_extensions]
            pics.sort()
         
            self.img_names.append(pics)
            self.each_clss_size[i] = len(pics)
            self.size += len(pics)
        
        print(f"Total number of images: {self.size}")

    def __len__(self):
        return self.size

    def load_image(self, image_path):
        image = cv2.imread(image_path, flags = cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # image = imread(image_path, as_gray=True)
        return self.preprocess(image)

    def preprocess(self, image):
        image = cv2.resize(image, (self.resize_w, self.resize_h))
        return image

    def xy(
        self,
        using_clss=-1,
        split=(0.6, 0.2, 0.2),
        shuffle_images_in_class=False,
        shuffle_indices=False,
        seed=None,
        save_split_dir=None
    ):
        """
        Stratified per-class split with ratios split=(train,val,test).
        Since each class has many images (>=100), every split will include every class.
        """

        if using_clss <= 0 or using_clss > self.num_clss:
            using_clss = self.num_clss

        tr_r, va_r, te_r = split
        s = tr_r + va_r + te_r
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1. Got sum={s}, split={split}")

        rng = np.random.default_rng(seed)

        H, W = self.resize_h, self.resize_w
        Xtrain, ytrain = [], []
        Xval, yval = [], []
        Xtest, ytest = [], []

        for clss, clss_images in enumerate(self.img_names[:using_clss]):
            clss_images = np.array(list(clss_images), dtype=str)
            total = len(clss_images)
            if total == 0:
                continue

            if shuffle_images_in_class:
                rng.shuffle(clss_images)

            # exact per-class counts (test gets remainder)
            n_train = int(total * tr_r)
            n_val = int(total * va_r)
            n_test = total - n_train - n_val

            # load (train)
            xtr = np.zeros((n_train, H, W), dtype=np.uint8)
            ytr = np.full((n_train,), clss, dtype=np.int64)
            for i in range(n_train):
                path = os.path.join(self.root, self.clsses[clss], clss_images[i])
                xtr[i] = self.load_image(path)
                if save_split_dir:
                    dest = os.path.join(save_split_dir, 'train', self.clsses[clss])
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(path, os.path.join(dest, clss_images[i]))

            # load (val)
            xva = np.zeros((n_val, H, W), dtype=np.uint8)
            yva = np.full((n_val,), clss, dtype=np.int64)
            start = n_train
            for i in range(n_val):
                path = os.path.join(self.root, self.clsses[clss], clss_images[start + i])
                xva[i] = self.load_image(path)
                if save_split_dir:
                    dest = os.path.join(save_split_dir, 'val', self.clsses[clss])
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(path, os.path.join(dest, clss_images[start + i]))

            # load (test)
            xte = np.zeros((n_test, H, W), dtype=np.uint8)
            yte = np.full((n_test,), clss, dtype=np.int64)
            start = n_train + n_val
            for i in range(n_test):
                path = os.path.join(self.root, self.clsses[clss], clss_images[start + i])
                xte[i] = self.load_image(path)
                if save_split_dir:
                    dest = os.path.join(save_split_dir, 'test', self.clsses[clss])
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(path, os.path.join(dest, clss_images[start + i]))

            Xtrain.append(xtr); ytrain.append(ytr)
            Xval.append(xva);   yval.append(yva)
            Xtest.append(xte);  ytest.append(yte)

            print(f'Loaded "{self.clsses[clss]}": total={total}, train={n_train}, val={n_val}, test={n_test}')

        # concatenate
        Xtrain = np.concatenate(Xtrain, axis=0)
        ytrain = np.concatenate(ytrain, axis=0)
        Xval = np.concatenate(Xval, axis=0)
        yval = np.concatenate(yval, axis=0)
        Xtest = np.concatenate(Xtest, axis=0)
        ytest = np.concatenate(ytest, axis=0)

        # shuffle inside each split
        if shuffle_indices:
            perm = rng.permutation(Xtrain.shape[0])
            Xtrain, ytrain = Xtrain[perm], ytrain[perm]

            perm = rng.permutation(Xval.shape[0])
            Xval, yval = Xval[perm], yval[perm]

            perm = rng.permutation(Xtest.shape[0])
            Xtest, ytest = Xtest[perm], ytest[perm]

        return Xtrain, ytrain, Xval, yval, Xtest, ytest

    def idx2name(self, target):
        target = np.array(target)
        labels = np.full_like(target, 0, dtype=str)
        for i, ele in enumerate(target):
            labels[i] = self.clsses[ele]

        return labels

    def name2idx(self, name):
        name = np.array(name)
        labels = np.full_like(name, 0, dtype=int)
        for i, ele in enumerate(name):
            labels[i] = self.clss2idx[ele]
        return labels




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    root = "dataset2"

        
    dataset = CustomDataset(root, resize_h=32, resize_w=32)
    
    save_dir = "dataset2_split"
    print(f"Splitting dataset and saving to {save_dir}...")
    
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.xy(
        using_clss=-1, 
        split=(0.6,0.2,0.2), 
        shuffle_images_in_class=False, 
        shuffle_indices=False, 
        seed=0,
        save_split_dir=save_dir
    )
    print("Total number of images {}".format(dataset.size))
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    
    print(f"Unique classes in Train: {len(np.unique(y_train))}")
    print(f"Unique classes in Val: {len(np.unique(y_val))}")
    print(f"Unique classes in Test: {len(np.unique(y_test))}")
    print(f"Split saved to {os.path.abspath(save_dir)}")



