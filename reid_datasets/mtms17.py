from __future__ import print_function, absolute_import
import os.path as osp
import os


def _pluck(root, list_path):
    ret = []
    labels = set()
    with open(file=list_path, mode="r") as f:
        for line in f.readlines():
            path, label = line.strip().split(" ")
            path_list = path.split("_")
            cam_id = int(path_list[2])
            labels.add(int(label))
            ret.append((osp.join(root, path), int(label), cam_id))

    return ret, len(labels)


def _plucktrainval(root, relabel=False):
    ret = []
    labels = set()
    class_sets = os.listdir(root)

    for index, label in enumerate(class_sets):
        class_path = osp.join(root, label)
        imgs = os.listdir(class_path)
        for img in imgs:
            name_grop = img.split("_")
            cam_id = int(name_grop[2])
            if relabel:
                ret.append((osp.join(class_path, img), index, cam_id))
                labels.add(index)
            else:
                ret.append((osp.join(class_path, img), int(label), cam_id))
                labels.add(int(label))

    return ret, len(labels)


class MTMS17(object):

    def __init__(self, root, split_id=0, train_list="list_train.txt", val_list="list_val.txt",
                 gallery_list="list_gallery.txt", query_list="list_query.txt"):
        super(MTMS17, self).__init__()

        self.root = root
        self.images_dir = root
        self.train_list, self.val_list = train_list, val_list
        self.gallery_list, self.query_list = gallery_list, query_list

        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []

        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
        self.num_query_ids, self.num_gallery_ids = 0, 0

        self.load()

    def load(self, verbose=True):

        self.train, self.num_train_ids = _pluck(root=osp.join(self.root, "train"),
                                                list_path=osp.join(self.root, self.train_list))
        self.val, self.num_val_ids = _pluck(root=osp.join(self.root, "train"),
                                            list_path=osp.join(self.root, self.val_list))
        self.trainval, self.num_trainval_ids = _plucktrainval(root=osp.join(self.root, "train"))

        self.query, self.num_query_ids = _pluck(root=osp.join(self.root, "test"),
                                                list_path=osp.join(self.root, self.query_list))
        self.gallery, self.num_gallery_ids = _pluck(root=osp.join(self.root, "test"),
                                                    list_path=osp.join(self.root, self.gallery_list))

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(self.num_query_ids, len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(self.num_gallery_ids, len(self.gallery)))
