from datasets import datasets


class DatasetSpecs:
    def __init__(self, n_channels: int, img_size: int, n_classes: int, loading_fn: callable):
        self.n_channels = n_channels
        self.img_size = img_size
        self.n_classes = n_classes
        self.loading_fn = loading_fn

    def __repr__(self):
        return str(self.__dict__)

_specs = dict(
    mnist=DatasetSpecs(n_channels=1, img_size=28, n_classes=10, loading_fn=datasets.mnist_loaders),
    cifar10=DatasetSpecs(n_channels=3, img_size=32, n_classes=10, loading_fn=datasets.cifar10_loaders),
    mpeg7=DatasetSpecs(n_channels=1, img_size=32, n_classes=70, loading_fn=datasets.mpeg7_loaders)
)


def get_specs(ds_name: str) -> DatasetSpecs:
    return _specs[ds_name]
