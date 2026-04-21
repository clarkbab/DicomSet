from __future__ import annotations

from collections import Counter
import cv2 as cv
from datetime import datetime
import numpy as np
import os
import pydicom as dcm
import re
import seaborn as sns
import skimage as ski
from typing import Any, Dict, List, Literal, Tuple

from ...typing import AffineMatrix3D, BatchLabelImage3D, DirPath, DiskRegionID, FilePath, Image2D, Image3D, LabelImage3D, Landmarks, PatientID, Point2D, Points2D, RegionID, RtStructDicom, Size2D, Size3D, Spacing2D, StudyID
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.geometry import affine_origin, affine_spacing, create_affine, to_image_coords
from ...utils.landmarks import points_to_landmarks
from ...utils.logging import logger
from ...utils.maths import round
from ...utils.python import filter_lists
from ..utils.io import load_dicom

CONTOUR_FORMATS = ['POINT', 'CLOSED_PLANAR']
CONTOUR_METHOD = 'SKIMAGE'
DEFAULT_LANDMARK_REGEXP = r'^Marker \d+$'
DICOM_DATE_FORMAT = '%Y%m%d'
DICOM_TIME_FORMAT = '%H%M%S'
EQUALITY_TOL_MM = 1e-3      # Equivalent up to 1 micron.

def __add_slice_contours(
    roi_contour: dcm.dataset.Dataset,
    data: Image2D,
    ref_ct: dcm.dataset.Dataset,
    idx: int,
    ) -> None:
    # Convert types. 
    data = data.astype('uint8')

    # Get contour coordinates.
    if CONTOUR_METHOD == 'OPENCV':
        # contours_coords, _ = cv.findContours(slice_data, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 'CHAIN_APPROX_SIMPLE' tries to replace straight line boundaries with two end points, instead
        # of using many points along the line - however it was producing only two points for some small
        # structures which Velocity doesn't like.
        contours_coords, _ = cv.findContours(data, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Process results.
        # Each slice can have multiple contours - separate foreground regions in the image.
        for i, c in enumerate(contours_coords):
            # OpenCV adds intermediate dimension - for some reason?
            c = c.squeeze(1)

            # Change in v4.11?
            # # OpenCV returns (y, x) points, so flip.
            c = np.flip(c, axis=1)
            contours_coords[i] = c

    elif CONTOUR_METHOD == 'SKIMAGE':
        contours_coords = ski.measure.find_contours(data)
        # Skimage needs no post-processing, as it returns (x, y) along the same
        # axes as the passed 'slice_data'. Also no strange intermediate dimensions.
    else:
        raise ValueError(f"CONTOUR_METHOD={CONTOUR_METHOD} not recognised.")

    contours_coords = list(contours_coords)

    # Velocity has an issue with loading contours that contain less than 3 points.
    for i, c in enumerate(contours_coords):
        if len(c) < 3:
            raise ValueError(f"Contour {i} of slice {idx} contains only 3 points: {contours_coords}. Velocity will not like this.")

    # 'contours_coords' is a list of contour coordinates, i.e. multiple contours are possible per slice,
    # - rtstruct masks can be disjoint in space.
    for coords in contours_coords:
        # Translate to world coordinates.
        origin = ref_ct.ImagePositionPatient
        spacing = ref_ct.PixelSpacing
        coords = np.array(coords) * spacing + origin[:-1]

        # Add z-index.
        z_indices = np.ones((len(coords), 1)) * ref_ct.ImagePositionPatient[2]
        coords = np.concatenate((coords, z_indices), axis=1)
        coords = list(coords.flatten())        

        # Create contour.
        contour = dcm.dataset.Dataset()
        contour.ContourData = coords
        contour.ContourGeometricType = 'CLOSED_PLANAR'
        contour.NumberOfContourPoints = len(coords) // 3

        # Add contour images.
        image = dcm.dataset.Dataset()
        image.ReferencedSOPClassUID = ref_ct.file_meta.MediaStorageSOPClassUID
        image.ReferencedSOPInstanceUID = ref_ct.file_meta.MediaStorageSOPInstanceUID

        # Append to contour.
        contour.ContourImageSequence = dcm.sequence.Sequence()
        contour.ContourImageSequence.append(image)

        # Append contour to ROI contour.
        roi_contour.ContourSequence.append(contour)

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
            cts = [load_dicom(cts, force=False)]
        else:
            # Load multiple CT slices.
            cts = [load_dicom(os.path.join(cts, f), force=False) for f in os.listdir(cts) if f.endswith('.dcm')]

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
        xy_poses = round(xy_poses, tol=EQUALITY_TOL_MM)
        xy_poses = np.unique(xy_poses, axis=0)
        if xy_poses.shape[0] > 1:
            raise ValueError(f"CT slices have inconsistent 'ImagePositionPatient' x/y values: {xy_poses}.")

    # Get z spacings.
    z_pos = list(sorted([c.ImagePositionPatient[2] for c in cts]))
    z_pos = round(z_pos, tol=EQUALITY_TOL_MM)
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
        rtdose = load_dicom(rtdose)
    data = np.transpose(rtdose.pixel_array)
    data = rtdose.DoseGridScaling * data

    # Create affine.
    spacing_xy = rtdose.PixelSpacing 
    z_diffs = np.diff(rtdose.GridFrameOffsetVector)
    z_diffs = round(z_diffs, tol=EQUALITY_TOL_MM)
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
        rtplan = load_dicom(rtplan)

    # Get info.
    info = {}
    info['isocentre'] = tuple([float(i) for i in rtplan.BeamSequence[0].ControlPointSequence[0].IsocenterPosition])
    # info['couch-shift'] = tuple([
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopLateralPosition),
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopVerticalPosition),
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopLongitudinalPosition),
    # ])

    return info

@alias_kwargs(
    ('r', 'region_id'),
    ('rr', 'return_regions'),
)
def from_rtstruct_dicom(
    rtstruct: FilePath | dcm.dataset.FileDataset,
    ct_size: Size3D,
    ct_affine: AffineMatrix3D, 
    landmark_id: DiskLandmarkID | List[DiskLandmarkID] | Literal['all'] | None = 'all',
    landmark_regexp: str | None = None,
    region_id: DiskRegionID | List[DiskRegionID] | Literal['all'] | None = 'all',    
    use_world_coords: bool = True,
    ) -> Tuple[BatchLabelImage3D | None, Landmarks | None]:
    if isinstance(rtstruct, str):
        rtstruct = load_dicom(rtstruct)
    ct_spacing = affine_spacing(ct_affine)
    ct_origin = affine_origin(ct_affine)
    landmark_regexp = landmark_regexp or DEFAULT_LANDMARK_REGEXP

    # Determine the rois to load.
    rois = rtstruct.StructureSetROISequence
    all_ids = [r.ROIName for r in rois]
    all_contours = rtstruct.ROIContourSequence
    if len(all_contours) != len(all_ids):
        raise ValueError(f"Length of 'StructureSetROISequence' and 'ROIContourSequence' must be the same, got '{len(all_ids)}' and '{len(all_contours)}' respectively.")
    # if region_id == 'all':
    #     region_ids = all_ids
    #     region_contours = all_contours
    # else:
    #     region_ids = arg_to_list(region_id, str)
    #     region_contours = [] 
    #     for r in region_ids:
    #         if r not in all_region_ids:
    #             raise ValueError(f"RTSTRUCT doesn't contain region '{r}'.")
    #         idx = all_region_ids.index(r)                   
    #         region_contours.append(all_region_contours[idx])

    # The data hierarchy is:
    # rtstruct -> roi_contours -> roi_contour -> contours -> contour -> contour_data.

    # Filter regions with missing 'ContourSequence' - this happened occasionally and we ended
    # up with blank masks.
    tmp_ids = all_ids
    all_ids, all_contours = filter_lists([all_ids, all_contours], lambda ic: getattr(ic[1], 'ContourSequence', None) is not None)
    n_skipped = len(tmp_ids) - len(all_ids)
    if n_skipped > 0:
        skipped = [i for i in tmp_ids if i not in all_ids]
        logger.warn(f"The following ({n_skipped}) rtstruct regions don't have contour data and will be skipped: {skipped}")

    # Handle landmarks first.
    if landmark_id is not None:
        # Filter for landmark contours.
        landmark_ids, landmark_contours = filter_lists([all_ids, all_contours], lambda ic: re.match(landmark_regexp, ic[0]) is not None)

        # Load data.
        points = []
        for i, c in zip(landmark_ids, landmark_contours):
            # Filter contours without data.
            contours = c.ContourSequence
            contours = list(filter(lambda c: hasattr(c, 'ContourData'), contours))

            if len(contours) != 1:
                raise ValueError(f"Expected contour sequence of length 1 for landmark '{i}', got {len(contours)}.")

            # Load landmark.
            contour = contours[0]
            if contour.ContourGeometricType != 'POINT':
                raise ValueError(f"Expected contour type 'POINT' for landmark data. Got '{contour.ContourGeometricType}'.")
            points.append(contour.ContourData)

        # Convert to dataframe.
        if len(points) > 0:
            points = np.stack(points, axis=0).astype(np.float32)
            if not use_world_coords:
                points = to_image_coords(points, ct_affine)
            landmarks_data = points_to_landmarks(points, landmark_ids)
        else:
            landmarks_data = None
    else:
        landmarks_data = None

    if region_id is not None:
        # Filter for region contours.
        region_ids, region_contours = filter_lists([all_ids, all_contours], lambda ic: re.match(landmark_regexp, ic[0]) is None)

        # Create label placeholder.
        regions_data = np.zeros((len(region_ids), *ct_size), dtype=bool)

        # Add regions data.
        for i, (r, cs) in enumerate(zip(region_ids, region_contours)):
            # Filter contours without data.
            contours = cs.ContourSequence
            contours = list(filter(lambda c: hasattr(c, 'ContourData'), contours))
            contours = list(sorted(contours, key=lambda c: c.ContourData[2]))  # Sort by z position.

            # Convert from boundary point cloud into binary mask. 
            for j, c in enumerate(contours):
                contour_data = np.array(c.ContourData)
                if not c.ContourGeometricType in CONTOUR_FORMATS:
                    raise ValueError(f"Expected one of '{CONTOUR_FORMATS}' ContourGeometricTypes, got '{c.ContourGeometricType}' for contour '{j}', region '{r}'.")

                # Coords are stored in flat array.
                if contour_data.size % 3 != 0:
                    raise ValueError(f"Size of 'contour_data' (array of points in 3D) should be divisible by 3.")
                points = np.array(contour_data).reshape(-1, 3)

                # Convert contour data to voxels.
                points_2D = points[:, :2]
                size_2D, spacing_2D, origin_2D = list(ct_size)[:2], list(ct_spacing)[:2], list(ct_origin)[:2]
                slice_data = __get_slice_mask(points_2D, size_2D, spacing_2D, origin_2D)

                # Get z index of slice.
                z_idx = int((points[0, 2] - ct_origin[2]) / ct_spacing[2])

                # Filter slices that are outside of the CT FOV.
                if z_idx < 0 or z_idx > ct_size[2] - 1: 
                    # Happened with 'PMCC-COMP:PMCC_AI_GYN_011' - Kidney_L...
                    continue

                # Write slice data to label, using XOR.
                regions_data[i, :, :, z_idx][slice_data == True] = np.invert(regions_data[i, :, :, z_idx][slice_data == True])
    else:
        regions_data = None

    return regions_data, landmarks_data

def __get_slice_mask(
    points: Points2D,
    size: Size2D,
    spacing: Spacing2D,
    origin: Point2D,
    ) -> np.ndarray:
    # Convert from physical coordinates to array indices.
    x_indices = (points[:, 0] - origin[0]) / spacing[0]
    y_indices = (points[:, 1] - origin[1]) / spacing[1]
    x_indices = np.around(x_indices)                    # Round to avoid truncation errors.
    y_indices = np.around(y_indices)

    # Convert to 'cv2' format.
    indices = np.stack((y_indices, x_indices), axis=1)  # (y, x) as 'cv.fillPoly' expects rows, then columns.
    indices = indices.astype('int32')                   # 'cv.fillPoly' expects 'int32' input points.
    pts = [np.expand_dims(indices, axis=0)]

    # Get all voxels on the boundary and interior described by the indices.
    slice_data = np.zeros(size, dtype='uint8')   # 'cv.fillPoly' expects to write to 'uint8' mask.shape=size, dtype='uint8')   # 'cv.fillPoly' expects to write to 'uint8' mask.
    cv.fillPoly(color=1, img=slice_data, pts=pts)
    slice_data = slice_data.astype(bool)

    return slice_data

def list_rtstruct_regions(
    rtstruct: FilePath | RtStructDicom,
    ) -> List[DiskRegionID]:
    if isinstance(rtstruct, str):
        rtstruct = load_dicom(rtstruct, force=False)
    all_infos = rtstruct.StructureSetROISequence
    all_contours = rtstruct.ROIContourSequence
    all_ids = list(sorted(i.ROIName for i in all_infos))
    # Filter on the presence of a 'ContourSequence' - sometimes empty.
    region_ids, _ = filter_lists([all_ids, all_contours], lambda ic: getattr(ic[1], 'ContourSequence', None) is not None)
    return region_ids

def to_ct_dicom(
    data: Image3D, 
    affine: AffineMatrix3D,
    patient_id: PatientID = 'pat_0',
    study_desc: str = 'study_0',
    study_id: str = 'study_0',  # Confusingly, this isn't the actualy ID, but a short human-readable ID - UID is the actual ID.
    patient_name: str | None = None,
    series_desc: str | None = None,
    series_number: int = 0,
    study_uid: StudyID | None = None,
    ) -> List[dcm.dataset.FileDataset]:
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
    study_uid = dcm.uid.generate_uid() if study_uid is None else study_uid
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
        file_meta.MediaStorageSOPClassUID = dcm.uid.CTImageStorage
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
        ct_dicom.PatientName = patient_id if patient_name is None else patient_name

        # Add study info.
        ct_dicom.StudyDate = dt.strftime(DICOM_DATE_FORMAT)
        if study_desc is not None:
            ct_dicom.StudyDescription = study_desc
        ct_dicom.StudyInstanceUID = study_uid
        ct_dicom.StudyID = study_id
        ct_dicom.StudyTime = dt.strftime(DICOM_TIME_FORMAT)

        # Add series info.
        ct_dicom.SeriesDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.SeriesDescription = f'CT ({study_id})' if series_desc is None else series_desc
        ct_dicom.SeriesInstanceUID = series_uid
        ct_dicom.SeriesNumber = series_number
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
    series_desc: str | None = None,
    ) -> dcm.dataset.FileDataset:
    if rtdose_template is not None:
        # Start from the template.
        if isinstance(rtdose_template, str):
            rtdose_template = load_dicom(rtdose_template)
        rtdose = rtdose_template.copy()

        # Overwrite sop ID.
        file_meta = rtdose.file_meta.copy()
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        rtdose.file_meta = file_meta
        rtdose.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    else:
        # Create rtdose from scratch.
        file_meta = dcm.dataset.Dataset()
        file_meta.FileMetaInformationGroupLength = 204
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.ImplementationClassUID = dcm.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.MediaStorageSOPClassUID = dcm.uid.RTDoseStorage
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian

        rtdose = dcm.dataset.FileDataset('filename', {}, file_meta=file_meta, preamble=b'\0' * 128)
        rtdose.BitsAllocated = 32
        rtdose.BitsStored = 32
        rtdose.DoseGridScaling = grid_scaling
        rtdose.DoseSummationType = 'PLAN'
        rtdose.DoseType = 'PHYSICAL'
        rtdose.DoseUnits = 'GY'
        rtdose.HighBit = 31
        rtdose.Modality = 'RTDOSE'
        rtdose.PhotometricInterpretation = 'MONOCHROME2'
        rtdose.PixelRepresentation = 0
        rtdose.SamplesPerPixel = 1
        rtdose.SOPClassUID = file_meta.MediaStorageSOPClassUID
        rtdose.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # Set custom attributes.
    rtdose.DeviceSerialNumber = ''
    rtdose.InstitutionAddress = ''
    rtdose.InstitutionName = 'PMCC'
    rtdose.InstitutionalDepartmentName = 'PMCC-AI'
    rtdose.Manufacturer = 'PMCC-AI'
    rtdose.ManufacturerModelName = 'PMCC-AI'
    rtdose.SoftwareVersions = ''
    
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
            setattr(rtdose, a, getattr(ref_dicom, a))

    # Add series info.
    rtdose.SeriesDescription = f'RTDOSE ({rtdose.study_id})' if series_desc is None else series_desc
    rtdose.SeriesInstanceUID = dcm.uid.generate_uid()
    rtdose.SeriesNumber = 1

    # Remove some attributes that might be set from the template.
    remove_attrs = [
        'OperatorsName',
        'StationName',
    ]
    if rtdose_template is not None:
        for a in remove_attrs:
            if hasattr(rtdose, a):
                delattr(rtdose, a)

    # Set image properties.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    rtdose.Columns = data.shape[0]
    rtdose.FrameIncrementPointer = dcm.datadict.tag_for_keyword('GridFrameOffsetVector')
    rtdose.GridFrameOffsetVector = [i * spacing[2] for i in range(data.shape[2])]
    rtdose.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    rtdose.ImagePositionPatient = list(origin)
    rtdose.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
    rtdose.NumberOfFrames = data.shape[2]
    rtdose.PixelSpacing = [spacing[0], spacing[1]]    # Uses (x, y) spacing.
    rtdose.Rows = data.shape[1]
    rtdose.SliceThickness = spacing[2]

    # Get grid scaling and data type.
    grid_scaling = rtdose.DoseGridScaling
    n_bits = rtdose.BitsAllocated
    if n_bits == 16:
        data_type = np.uint16
    elif n_bits == 32:
        data_type = np.uint32
    else:
        raise ValueError(f'Unsupported BitsAllocated value: {n_bits}. Must be 16 or 32.')

    # Add dose data. 
    data = (data / grid_scaling).astype(data_type)
    rtdose.PixelData = np.transpose(data).tobytes()     # Uses (z, y, x) format.

    # Set timestamps.
    dt = datetime.now()
    rtdose.ContentDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose.ContentTime = dt.strftime(DICOM_TIME_FORMAT)
    rtdose.InstanceCreationDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose.InstanceCreationTime = dt.strftime(DICOM_TIME_FORMAT)
    rtdose.SeriesDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose.SeriesTime = dt.strftime(DICOM_TIME_FORMAT)

    return rtdose

def to_rtstruct_dicom(
    data: BatchLabelImage3D | LabelImage3D,
    region_id: RegionID | List[RegionID], 
    ref_cts: DirPath | List[dcm.dataset.FileDataset],
    generation_algorithm: str | None = None,
    institution: str | None = None, 
    label: str = 'RTSTRUCT',
    series_desc: str | None = None,
    series_id: str | None = None,
    series_number: int = 0,
    ) -> dcm.dataset.FileDataset:
    if data.ndim == 3:
        data = np.expand_dims(data, axis=0)    
    region_ids = arg_to_list(region_id, str)
    assert len(data) == len(region_ids), f"Length of 'data' and 'region_id' must be the same, got {len(data)} and {len(region_ids)} respectively."
    if isinstance(ref_cts, str):
        ref_cts = [load_dicom(os.path.join(ref_cts, f), force=False) for f in os.listdir(ref_cts) if f.endswith('.dcm')]

    # Create metadata.
    metadata = dcm.dataset.FileMetaDataset()
    metadata.FileMetaInformationGroupLength = 204
    metadata.FileMetaInformationVersion = b'\x00\x01'
    metadata.MediaStorageSOPClassUID = dcm.uid.RTStructureSetStorage
    metadata.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
    metadata.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian
    metadata.ImplementationClassUID = dcm.uid.PYDICOM_IMPLEMENTATION_UID

    # Create rtstruct.
    rtstruct = dcm.dataset.FileDataset('filename', {}, file_meta=metadata, preamble=b'\0' * 128)
    rtstruct.StructureSetROISequence = dcm.sequence.Sequence()
    rtstruct.ROIContourSequence = dcm.sequence.Sequence()
    rtstruct.RTROIObservationsSequence = dcm.sequence.Sequence()

    # Set transfer syntax.
    rtstruct.is_little_endian = True
    rtstruct.is_implicit_VR = True

    # Set values from metadata.
    rtstruct.SOPClassUID = metadata.MediaStorageSOPClassUID
    rtstruct.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID

    # Set date/time.
    dt = datetime.now()
    date = dt.strftime(DICOM_DATE_FORMAT)
    time = dt.strftime(DICOM_TIME_FORMAT)

    # Set other required fields.
    rtstruct.ApprovalStatus = 'UNAPPROVED'
    rtstruct.ContentDate = date
    rtstruct.ContentTime = time
    rtstruct.InstanceCreationDate = date
    rtstruct.InstanceCreationTime = time
    if institution is not None:
        rtstruct.InstitutionName = institution
    rtstruct.Modality = 'RTSTRUCT'
    rtstruct.SpecificCharacterSet = 'ISO_IR 100'
    rtstruct.StructureSetDate = date
    rtstruct.StructureSetLabel = label
    rtstruct.StructureSetTime = time

    # Add patient info.
    rtstruct.PatientAge = getattr(ref_cts[0], 'PatientAge', '')
    rtstruct.PatientBirthDate = getattr(ref_cts[0], 'PatientBirthDate', '')
    rtstruct.PatientID = getattr(ref_cts[0], 'PatientID', '')
    rtstruct.PatientName = getattr(ref_cts[0], 'PatientName', '')
    rtstruct.PatientSex = getattr(ref_cts[0], 'PatientSex', '')
    rtstruct.PatientSize = getattr(ref_cts[0], 'PatientSize', '')
    rtstruct.PatientWeight = getattr(ref_cts[0], 'PatientWeight', '')

    # Add study info.
    rtstruct.StudyDate = ref_cts[0].StudyDate
    rtstruct.StudyDescription = getattr(ref_cts[0], 'StudyDescription', '')
    rtstruct.StudyInstanceUID = ref_cts[0].StudyInstanceUID
    rtstruct.StudyID = ref_cts[0].StudyID
    rtstruct.StudyTime = ref_cts[0].StudyTime

    # Add series info.
    rtstruct.SeriesDate = date
    rtstruct.SeriesTime = time
    rtstruct.SeriesInstanceUID = dcm.uid.generate_uid() if series_id is None else series_id
    rtstruct.SeriesDescription = f'RTSTRUCT ({rtstruct.StudyID})' if series_desc is None else series_desc
    rtstruct.SeriesNumber = series_number

    # Add frame of reference.
    ref_frame = dcm.dataset.Dataset()
    ref_frame.FrameOfReferenceUID = ref_cts[0].FrameOfReferenceUID

    # Add referenced series.
    series = dcm.dataset.Dataset()
    series.SeriesInstanceUID = ref_cts[0].SeriesInstanceUID

    # Add contour image sequence.
    series.ContourImageSequence = dcm.sequence.Sequence()
    for c in ref_cts:
        contour_image = dcm.dataset.Dataset()
        contour_image.ReferencedSOPClassUID = c.file_meta.MediaStorageSOPClassUID
        contour_image.ReferencedSOPInstanceUID = c.file_meta.MediaStorageSOPInstanceUID
        series.ContourImageSequence.append(contour_image)

    # Add series to the frame of reference.
    ref_frame.RTReferencedSeriesSequence = dcm.sequence.Sequence()
    ref_frame.RTReferencedSeriesSequence.append(series)

    # Add frame of reference to RTSTRUCT.
    rtstruct.ReferencedFrameOfReferenceSequence = dcm.sequence.Sequence()
    rtstruct.ReferencedFrameOfReferenceSequence.append(ref_frame)

    # Add regions data.
    palette = sns.color_palette('colorblind', len(region_ids))
    for i, r in enumerate(region_ids):
        # Create ROI contour.        
        roi_contour = dcm.dataset.Dataset()
        roi_contour.ROIDisplayColor = list(np.array(palette[i]) * 255)   # To 8-bit colours.
        roi_contour.ReferencedROINumber = str(i)

        # Add structure set ROIs.
        structure_set_roi = dcm.dataset.Dataset()
        structure_set_roi.ROINumber = str(i)
        structure_set_roi.ReferencedFrameOfReferenceUID = rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
        structure_set_roi.ROIName = r
        if generation_algorithm is not None:
            structure_set_roi.ROIGenerationAlgorithm = generation_algorithm
        rtstruct.StructureSetROISequence.append(structure_set_roi)

        # Add point cloud contours.
        roi_contour.ContourSequence = dcm.sequence.Sequence()
        for j, c in enumerate(ref_cts):
            slice_data = data[i, :, :, j]
            if slice_data.sum() == 0:
                continue
            __add_slice_contours(roi_contour, slice_data, c, j)
        rtstruct.ROIContourSequence.append(roi_contour)

        # Add RT ROI observations - I don't know what these are.
        observation = dcm.dataset.Dataset()
        observation.ObservationNumber = str(i)
        observation.ReferencedROINumber = str(i)
        observation.RTROIInterpretedType = ''
        observation.ROIInterpreter = ''
        rtstruct.RTROIObservationsSequence.append(observation)

    return rtstruct
    