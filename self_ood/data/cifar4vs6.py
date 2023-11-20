from typing import *

from torch.utils.data import Subset

from pl_bolts.datamodules import CIFAR10DataModule

from .transforms import SimCLRViews


class CIFAR4vs6(CIFAR10DataModule):
    ood_classes = [0, 1, 3, 5]  # out-of-distribution classes: airoplane, automobile, cat, dog
    id_classes = sorted(set(range(10)).difference(ood_classes))  # in-distribution classes
    EXTRA_ARGS = {
        'target_transform': lambda label: (
            CIFAR4vs6.id_classes.index(label)
            if label in CIFAR4vs6.id_classes
            else -1
        )
    }  # these kwargs are given to torchvision.datasets.CIFAR10 class

    def __init__(
            self,
            data_dir: str,
            batch_size: int = 256,
            num_workers: int = 8,
            **simclr_views_params: Any
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            val_split=1000,
            num_workers=num_workers,
            normalize=True,
            batch_size=batch_size
        )

        params = dict(
            size=32,
            jitter_strength=0.5,
            blur=False,
        )
        params.update(simclr_views_params)
        self._train_transforms = SimCLRViews(
            **params,
            final_transforms=self.default_transforms()
        )
        self._val_transforms = SimCLRViews(
            **params,
            num_views=10,
            final_transforms=self.default_transforms(),
        )

    @property
    def num_classes(self) -> int:
        return len(self.id_classes)

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)

        if stage == "fit" or stage is None:
            self.dataset_train = Subset(
                dataset=self.dataset_train.dataset,
                indices=[i for i in self.dataset_train.indices
                         if self.dataset_train.dataset.targets[i] in CIFAR4vs6.id_classes]
            )
