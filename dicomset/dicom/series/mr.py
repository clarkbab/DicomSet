from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import Any, Dict, List, TYPE_CHECKING

from ... import config
from ...typing import Box3D, Image3D, Point3D, SeriesID, Size3D, Spacing3D
from ...utils.geometry import fov
from ...utils.python import ensure_loaded, get_private_attr
from ..utils.io import load_dicom
from .series import DicomSeries
if TYPE_CHECKING:
    from ..dataset import DicomDataset
    from ..patient import DicomPatient
    from ..study import DicomStudy

class DicomMrSeries(DicomSeries):
    def __init__(
        self,
        dataset: DicomDataset,
        patient: DicomPatient,
        study: DicomStudy,
        id: SeriesID,
        index: pd.DataFrame,
        index_policy: Dict[str, Any],
        ) -> None:
        super().__init__('mr', dataset, patient, study, id, index=index, index_policy=index_policy)
        dspath = os.path.join(config.directories.datasets, 'dicom', self._dataset.id, 'data', 'patients')
        relpaths = list(index['filepath'])
        abspaths = [os.path.join(dspath, p) for p in relpaths]
        self.__filepaths = abspaths
    
    @property
    @ensure_loaded('__data', '__load_data')
    def data(self) -> Image3D:
        return self.__data

    @property
    @ensure_loaded('__data', '__load_data')
    def dicoms(self) -> List[MrDicom]:
        return self.__dicoms

    @ensure_loaded('__data', '__load_data')
    def fov(
        self,
        **kwargs) -> Box3D:
        return fov(self.__data, origin=self.__origin, spacing=self.__spacing, **kwargs)

    def __load_data(self) -> None:
        # Load dicoms.
        # Sort MRs by z position, smallest first.
        dicoms = [load_dicom(f, force=False) for f in self.__filepaths]
        dicoms = list(sorted(dicoms, key=lambda m: m.ImagePositionPatient[2]))
        self.__dicoms = dicoms

        # Store origin.
        # Indexing checked that all 'ImagePositionPatient' keys were the same for the series.
        raise ValueError("Implement 'from_mr_dicom'")
        self.__data, self.__affine = from_mr_dicom(dicoms)

        origin = mr_dicoms[0].ImagePositionPatient    
        self.__origin = tuple(float(round(o)) for o in origin)

        # Store size.
        # Indexing checked that MR slices had consisent x/y spacing in series.
        self.__size = (
            mr_dicoms[0].pixel_array.shape[1],
            mr_dicoms[0].pixel_array.shape[0],
            len(mr_dicoms)
        )

        # Store spacing.
        # Indexing checked that MR slices were equally spaced in z-dimension.
        self.__spacing = (
            float(mr_dicoms[0].PixelSpacing[0]),
            float(mr_dicoms[0].PixelSpacing[1]),
            float(np.abs(mr_dicoms[1].ImagePositionPatient[2] - mr_dicoms[0].ImagePositionPatient[2]))
        )

        # Store MR data.
        data = np.zeros(shape=self.__size)
        for m in mr_dicoms:
            mr_data = np.transpose(m.pixel_array)      # 'pixel_array' contains row-first image data.

            # Get z index.
            z_origin =  m.ImagePositionPatient[2] - self.__origin[2]
            z_idx = int(round(z_origin / self.__spacing[2]))

            # Add data.
            data[:, :, z_idx] = mr_data

        self.__data = data

    @property
    @ensure_loaded('__data', '__load_data')
    def origin(self) -> Point3D:
        return self.__origin

    @property
    @ensure_loaded('__data', '__load_data')
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded('__data', '__load_data')
    def spacing(self) -> Spacing3D:
        return self.__spacing

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepaths']
for p in props:
    setattr(DicomMrSeries, p, property(lambda self, p=p: get_private_attr(self, f'__{p}')))
