from __future__ import annotations

from collections import Counter
from datetime import datetime
import numpy as np
import os
import pydicom as dcm
from typing import Any, Dict, List, Tuple

from ...typing import AffineMatrix3D, DirPath, FilePath, Image2D, Image3D, PatientID, SeriesID, StudyID
from ...utils.geometry import affine_origin, affine_spacing, create_affine
from ...utils.maths import round

DICOM_DATE_FORMAT = '%Y%m%d'
DICOM_TIME_FORMAT = '%H%M%S'

def from_ct_dicom(
    # DirPath | List[CtDicom] -> (CtVolume, Affine), FilePath -> CtSlice.
    cts: FilePath | DirPath | List[dcm.dataset.FileDataset],
    check_orientation: bool = True,
    check_xy_positions: bool = True,
    check_z_spacing: bool = True,
    ) -> Image2D | Tuple[Image3D, AffineMatrix3D]:
    # Load from filepath/dirpath if present.
    if isinstance(cts, str):
        if os.path.isfile(cts):
            # Load single CT slice.
            ct = dcm.dcmread(cts, force=False) 
        else:
            # Load multiple CT slices.
            cts = [dcm.dcmread(os.path.join(cts, f), force=False) for f in os.listdir(cts) if f.endswith('.dcm')]

    # Check that standard orientation is used.
    # TODO: Handle non-standard orientation.
    if check_orientation:
        for c in cts:
            assert c.PatientPosition == 'HFS', f"CT slice has non-standard 'PatientPosition' value: {c.PatientPosition}. Only 'HFS' is supported."
            if c.ImageOrientationPatient != [1, 0, 0, 0, 1, 0]:
                raise ValueError(f"CT slice has non-standard 'ImageOrientationPatient' value: {c.ImageOrientationPatient}. Only axial slices with orientation [1, 0, 0, 0, 1, 0] are supported.")

    # Make sure x/y positions are the same for all slices.
    if check_xy_positions:
        xy_poses = np.array([c.ImagePositionPatient[:2] for c in cts])
        xy_poses = round(xy_poses, tol=1e-6)
        xy_poses = np.unique(xy_poses, axis=0)
        if xy_poses.shape[0] > 1:
            raise ValueError(f"CT slices have inconsistent 'ImagePositionPatient' x/y values: {xy_poses}.")

    # Get z spacings.
    z_pos = list(sorted([c.ImagePositionPatient[2] for c in cts]))
    z_pos = round(z_pos, tol=1e-6)
    z_diffs = np.diff(z_pos)
    z_freqs = Counter(z_diffs)
    if check_z_spacing and len(z_freqs.keys()) > 1:
        raise ValueError(f"CT slices have inconsistent 'ImagePositionPatient' z spacing frequencies: {z_freqs}.")
    # If we're ignoring multiple diffs, then take the most frequent diff.
    z_diff = sorted(z_freqs.items(), key=lambda i: i[1])[-1][0] 

    # Sort CTs by z position, smallest first.
    cts = list(sorted(cts, key=lambda c: c.ImagePositionPatient[2]))

    # Calculate origin.
    # Indexing checked that all 'ImagePositionPatient' keys were the same for the series.
    origin = cts[0].ImagePositionPatient
    origin = tuple(float(o) for o in origin)

    # Calculate size.
    # Indexing checked that CT slices had consisent x/y spacing in series.
    size = (
        cts[0].pixel_array.shape[1],
        cts[0].pixel_array.shape[0],
        len(cts)
    )

    # Calculate spacing.
    # Indexing checked that CT slices were equally spaced in z-dimension.
    spacing = (
        float(cts[0].PixelSpacing[0]),
        float(cts[0].PixelSpacing[1]),
        z_diff,
    )

    # Create CT data - sorted by z-position.
    data = np.zeros(shape=size)
    for i, c in enumerate(cts):
        # Convert values to HU.
        slice_data = np.transpose(c.pixel_array)      # 'pixel_array' contains row-first image data.
        slice_data = c.RescaleSlope * slice_data + c.RescaleIntercept

        # Add slice data.
        data[:, :, i] = slice_data

    affine = create_affine(spacing, origin)

    return data, affine

def from_rtdose_dicom(
    rtdose: FilePath | dcm.dataset.FileDataset | None = None,
    ) -> Tuple[Image3D, AffineMatrix3D]:
    # Load data.
    if isinstance(rtdose, str):
        rtdose = dcm.dcmread(rtdose)
    data = np.transpose(rtdose.pixel_array)
    data = rtdose.DoseGridScaling * data

    # Create affine.
    spacing_xy = rtdose.PixelSpacing 
    z_diffs = np.diff(rtdose.GridFrameOffsetVector)
    z_diffs = round(z_diffs, tol=1e-6)
    z_diffs = np.unique(z_diffs)
    if len(z_diffs) != 1:
        raise ValueError(f"Slice z spacings for RtDoseDicom not equal: {z_diffs}.")
    spacing_z = z_diffs[0]
    spacing = tuple((float(s) for s in np.append(spacing_xy, spacing_z)))
    origin = tuple(float(o) for o in rtdose.ImagePositionPatient)
    affine = create_affine(spacing, origin)

    return data, affine

def from_rtplan_dicom(
    rtplan: FilePath | dcm.dataset.FileDataset,
    ) -> Dict[str, Any]:
    if isinstance(rtplan, str):
        rtplan = dcm.dcmread(rtplan, force=False)

    # Get info.
    info = {}
    info['isocentre'] = tuple([float(i) for i in rtplan.BeamSequence[0].ControlPointSequence[0].IsocenterPosition])
    # info['couch-shift'] = tuple([
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopLateralPosition),
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopVerticalPosition),
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopLongitudinalPosition),
    # ])

    return info

def to_ct_dicom(
    data: Image3D, 
    affine: AffineMatrix3D,
    patient_id: PatientID,
    study_id: StudyID,
    patient_name: str | None = None,
    series_id: SeriesID | None = None,
    ) -> List[dcm.dataset.FileDataset]:
    patient_name = patient_id if patient_name is None else patient_name
    series_id = f'CT ({study_id})' if series_id is None else series_id

    # Data settings.
    if data.min() < -1024:
        raise ValueError(f"Min CT value {data.min()} is less than -1024. Cannot use unsigned 16-bit values for DICOM.")
    rescale_intercept = -1024
    rescale_slope = 1
    n_bits_alloc = 16
    n_bits_stored = 12
    numpy_type = np.uint16  # Must match 'n_bits_alloc'.
    
    # DICOM data is stored using unsigned int with min=0 and max=(2 ** n_bits_stored) - 1.
    # Don't crop at the bottom, but crop large CT values to be below this threshold.
    ct_max_rescaled = 2 ** (n_bits_stored) - 1
    ct_max = (ct_max_rescaled * rescale_slope) + rescale_intercept
    data = np.minimum(data, ct_max)

    # Perform rescale.
    data_rescaled = (data - rescale_intercept) / rescale_slope
    data_rescaled = data_rescaled.astype(numpy_type)
    scaled_ct_min, scaled_ct_max = data_rescaled.min(), data_rescaled.max()
    if scaled_ct_min < 0 or scaled_ct_max > (2 ** n_bits_stored - 1):
        # This should never happen now that we're thresholding raw HU values.
        raise ValueError(f"Scaled CT data out of bounds: min {scaled_ct_min}, max {scaled_ct_max}. Max allowed: {2 ** n_bits_stored - 1}.")

    # Create study and series fields.
    # StudyID and StudyInstanceUID are different fields.
    # StudyID is a human-readable identifier, while StudyInstanceUID is a unique identifier.
    study_uid = dcm.uid.generate_uid()
    series_uid = dcm.uid.generate_uid()
    frame_of_reference_uid = dcm.uid.generate_uid()
    dt = datetime.now()

    # Create a file for each slice.
    n_slices = data.shape[2]
    ct_dicoms = []
    for i in range(n_slices):
        # Create metadata header.
        file_meta = dcm.dataset.FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 204
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.ImplementationClassUID = dcm.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.MediaStorageSOPClassUID = dcm.uid.CtImageArraytorage
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian

        # Create DICOM dataset.
        ct_dicom = dcm.FileDataset('filename', {}, file_meta=file_meta, preamble=b'\0' * 128)
        ct_dicom.is_little_endian = True
        ct_dicom.is_implicit_VR = True
        ct_dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ct_dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        # Set other required fields.
        ct_dicom.ContentDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.ContentTime = dt.strftime(DICOM_TIME_FORMAT)
        ct_dicom.InstanceCreationDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.InstanceCreationTime = dt.strftime(DICOM_TIME_FORMAT)
        ct_dicom.InstitutionName = 'PMCC'
        ct_dicom.Manufacturer = 'PMCC'
        ct_dicom.Modality = 'CT'
        ct_dicom.SpecificCharacterSet = 'ISO_IR 100'

        # Add patient info.
        ct_dicom.PatientID = patient_id
        ct_dicom.PatientName = patient_name

        # Add study info.
        ct_dicom.StudyDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.StudyDescription = study_id
        ct_dicom.StudyInstanceUID = study_uid
        ct_dicom.StudyID = study_id
        ct_dicom.StudyTime = dt.strftime(DICOM_TIME_FORMAT)

        # Add series info.
        ct_dicom.SeriesDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.SeriesDescription = series_id
        ct_dicom.SeriesInstanceUID = series_uid
        ct_dicom.SeriesNumber = 0
        ct_dicom.SeriesTime = dt.strftime(DICOM_TIME_FORMAT)

        # Add data.
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        ct_dicom.BitsAllocated = n_bits_alloc
        ct_dicom.BitsStored = n_bits_stored
        ct_dicom.FrameOfReferenceUID = frame_of_reference_uid
        ct_dicom.HighBit = 11
        origin_z = origin[2] + i * spacing[2]
        ct_dicom.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ct_dicom.ImagePositionPatient = [origin[0], origin[1], origin_z]
        ct_dicom.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        ct_dicom.InstanceNumber = i + 1
        ct_dicom.PhotometricInterpretation = 'MONOCHROME2'
        ct_dicom.PatientPosition = 'HFS'
        ct_dicom.PixelData = np.transpose(data_rescaled[:, :, i]).tobytes()   # Uses (y, x) spacing.
        ct_dicom.PixelRepresentation = 0
        ct_dicom.PixelSpacing = [spacing[0], spacing[1]]    # Uses (x, y) spacing.
        ct_dicom.RescaleIntercept = rescale_intercept
        ct_dicom.RescaleSlope = rescale_slope
        ct_dicom.Rows, ct_dicom.Columns = data.shape[1], data.shape[0]
        ct_dicom.SamplesPerPixel = 1
        ct_dicom.SliceThickness = float(abs(spacing[2]))

        ct_dicoms.append(ct_dicom)

    return ct_dicoms

def to_rtdose_dicom(
    data: Image3D, 
    affine: AffineMatrix3D,
    grid_scaling: float = 1e-3,
    ref_ct: FilePath | dcm.dataset.FileDataset | None = None,
    rtdose_template: FilePath | dcm.dataset.FileDataset | None = None,
    series_description: str | None = None,
    ) -> dcm.dataset.FileDataset:
    if rtdose_template is not None:
        # Start from the template.
        if isinstance(rtdose_template, str):
            rtdose_template = dcm.dcmread(rtdose_template)
        rtdose_dicom = rtdose_template.copy()

        # Overwrite sop ID.
        file_meta = rtdose_dicom.file_meta.copy()
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        rtdose_dicom.file_meta = file_meta
        rtdose_dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    else:
        # Create rtdose from scratch.
        file_meta = dcm.dataset.Dataset()
        file_meta.FileMetaInformationGroupLength = 204
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.ImplementationClassUID = dcm.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.MediaStorageSOPClassUID = dcm.uid.RTDoseStorage
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian

        rtdose_dicom = dcm.dataset.FileDataset('filename', {}, file_meta=file_meta, preamble=b'\0' * 128)
        rtdose_dicom.BitsAllocated = 32
        rtdose_dicom.BitsStored = 32
        rtdose_dicom.DoseGridScaling = grid_scaling
        rtdose_dicom.DoseSummationType = 'PLAN'
        rtdose_dicom.DoseType = 'PHYSICAL'
        rtdose_dicom.DoseUnits = 'GY'
        rtdose_dicom.HighBit = 31
        rtdose_dicom.Modality = 'RTDOSE'
        rtdose_dicom.PhotometricInterpretation = 'MONOCHROME2'
        rtdose_dicom.PixelRepresentation = 0
        rtdose_dicom.SamplesPerPixel = 1
        rtdose_dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID
        rtdose_dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # Set custom attributes.
    rtdose_dicom.DeviceSerialNumber = ''
    rtdose_dicom.InstitutionAddress = ''
    rtdose_dicom.InstitutionName = 'PMCC'
    rtdose_dicom.InstitutionalDepartmentName = 'PMCC-AI'
    rtdose_dicom.Manufacturer = 'PMCC-AI'
    rtdose_dicom.ManufacturerModelName = 'PMCC-AI'
    rtdose_dicom.SoftwareVersions = ''
    
    # Copy atributes from reference ct/rtdose dicom.
    assert rtdose_template is not None or ref_ct is not None
    ref_dicom = rtdose_template if rtdose_template is not None else ref_ct
    attrs = [
        'AccessionNumber',
        'FrameOfReferenceUID',
        'PatientBirthDate',
        'PatientID',
        'PatientName',
        'PatientSex',
        'StudyDate',
        'StudyDescription',
        'StudyID',
        'StudyInstanceUID',
        'StudyTime'
    ]
    for a in attrs:
        if hasattr(ref_dicom, a):
            setattr(rtdose_dicom, a, getattr(ref_dicom, a))

    # Add series info.
    series_description = rtdose_dicom.StudyID if series_description is None else series_description
    rtdose_dicom.SeriesDescription = f'RTDOSE ({series_description})'
    rtdose_dicom.SeriesInstanceUID = dcm.uid.generate_uid()
    rtdose_dicom.SeriesNumber = 1

    # Remove some attributes that might be set from the template.
    remove_attrs = [
        'OperatorsName',
        'StationName',
    ]
    if rtdose_template is not None:
        for a in remove_attrs:
            if hasattr(rtdose_dicom, a):
                delattr(rtdose_dicom, a)

    # Set image properties.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    rtdose_dicom.Columns = data.shape[0]
    rtdose_dicom.FrameIncrementPointer = dcm.datadict.tag_for_keyword('GridFrameOffsetVector')
    rtdose_dicom.GridFrameOffsetVector = [i * spacing[2] for i in range(data.shape[2])]
    rtdose_dicom.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    rtdose_dicom.ImagePositionPatient = list(origin)
    rtdose_dicom.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
    rtdose_dicom.NumberOfFrames = data.shape[2]
    rtdose_dicom.PixelSpacing = [spacing[0], spacing[1]]    # Uses (x, y) spacing.
    rtdose_dicom.Rows = data.shape[1]
    rtdose_dicom.SliceThickness = spacing[2]

    # Get grid scaling and data type.
    grid_scaling = rtdose_dicom.DoseGridScaling
    n_bits = rtdose_dicom.BitsAllocated
    if n_bits == 16:
        data_type = np.uint16
    elif n_bits == 32:
        data_type = np.uint32
    else:
        raise ValueError(f'Unsupported BitsAllocated value: {n_bits}. Must be 16 or 32.')

    # Add dose data. 
    data = (data / grid_scaling).astype(data_type)
    rtdose_dicom.PixelData = np.transpose(data).tobytes()     # Uses (z, y, x) format.

    # Set timestamps.
    dt = datetime.now()
    rtdose_dicom.ContentDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose_dicom.ContentTime = dt.strftime(DICOM_TIME_FORMAT)
    rtdose_dicom.InstanceCreationDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose_dicom.InstanceCreationTime = dt.strftime(DICOM_TIME_FORMAT)
    rtdose_dicom.SeriesDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose_dicom.SeriesTime = dt.strftime(DICOM_TIME_FORMAT)

    return rtdose_dicom