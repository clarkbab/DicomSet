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

from ...typing import AffineMatrix3D, BatchLabelImage3D, DirPath, DiskLandmarkID, DiskRegionID, FilePath, Image2D, Image3D, LabelImage3D, LandmarkID, Landmarks, PatientID, Point2D, Point3D, Points2D, RegExp, RegionID, RtStructDicom, Size2D, Size3D, Spacing2D, StudyID
from ...utils.args import alias_kwargs, arg_to_list, resolve_filepath
from ...utils.geometry import affine_origin, affine_spacing, create_affine, to_image_coords
from ...utils.landmarks import points_to_landmarks
from ...utils.logging import logger
from ...utils.maths import round
from ...utils.python import filter_lists, sort_lists
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
        # Slicer didn't like this - didn't load the contours any more.
        # 'find_contours' ignores pixels on the border of the image - which results in 
        # weird and split contours. Pad the image, and then remove the padding from the
        # resulting coords.
        # data = np.pad(data, pad_width=1, mode='constant', constant_values=0)
        # contours_coords = ski.measure.find_contours(data, level=0.5)
        # contours_coords = [c - 1 for c in contours_coords]

        # 'find_contours' doesn't handle border pixels, so erode these.
        data[0, :] = 0
        data[-1, :] = 0
        data[:, 0] = 0
        data[:, -1] = 0
        contours_coords = ski.measure.find_contours(data, level=0.5)

        # Skimage needs no post-processing, as it returns (x, y) along the same
        # axes as the passed 'slice_data'. Also no strange intermediate dimensions.
    else:
        raise ValueError(f"CONTOUR_METHOD={CONTOUR_METHOD} not recognised.")

    contours_coords = list(contours_coords)

    # Velocity has an issue with loading contours that contain less than 3 points.
    for i, c in enumerate(contours_coords):
        if len(c) < 3:
            logger.warn(f"Contour {i} of slice {idx} contains only {len(c)} points: {c}. Velocity will not like this.")

    # 'contours_coords' is a list of contour coordinates, i.e. multiple contours are possible per slice,
    # - rtstruct masks can be disjoint in space.
    for coords in contours_coords:
        # Convert to world coordinates.
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
            filepath = resolve_filepath(cts)
            cts = [load_dicom(filepath, force=False)]
        else:
            # Load multiple CT slices.
            dirpath = resolve_filepath(cts)
            cts = [load_dicom(os.path.join(dirpath, f), force=False) for f in os.listdir(dirpath) if f.endswith('.dcm')]

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
    landmarks_use_world_coords: bool = True,
    region_id: DiskRegionID | List[DiskRegionID] | Literal['all'] | None = 'all',    
    return_regions: bool = True,
    ) -> Tuple[List[RegionID] | BatchLabelImage3D] | Tuple[List[LandmarkID] | Landmarks] | Tuple[List[RegionID], BatchLabelImage3D, List[LandmarkID], Landmarks]:
    if isinstance(rtstruct, str):
        filepath = resolve_filepath(rtstruct)
        rtstruct = load_dicom(filepath)
    ct_spacing = affine_spacing(ct_affine)
    ct_origin = affine_origin(ct_affine)
    landmark_regexp = landmark_regexp or DEFAULT_LANDMARK_REGEXP
    assert region_id is not None or landmark_id is not None, "Either 'region/landmark_id' must not be None."

    # Get landmark/region names.
    if region_id is not None:
        region_ids, region_contours = list_rtstruct_regions(rtstruct, landmark_regexp=landmark_regexp, region_id=region_id, return_contours=True)
    if landmark_id is not None:
        landmark_ids, landmark_contours = list_rtstruct_landmarks(rtstruct, landmark_id=landmark_id, landmark_regexp=landmark_regexp, return_contours=True)

    # The data hierarchy is:
    # rtstruct -> roi_contours -> roi_contour -> contours -> contour -> contour_data.

    # Load regions data.
    regions_data = None
    if region_id is not None:
        regions_data = [__get_region_label(cs, ct_size, ct_affine) for cs in region_contours]
        if len(regions_data) != 0:
            regions_data = np.stack(regions_data, axis=0)
        else:
            regions_data = None

    # Process landmarks data.
    landmarks_data = None
    if landmark_id is not None:
        landmarks_points = [__get_landmark_point(cs, ct_affine, use_world_coords=landmarks_use_world_coords) for cs in landmark_contours]
        if len(landmarks_points) != 0:
            landmarks_points = np.stack(landmarks_points, axis=0)
            landmarks_data = points_to_landmarks(landmarks_points, landmark_ids)
        else:
            landmarks_data = None

    result = []
    if region_id is not None:
        result += [region_ids, regions_data] if return_regions else [regions_data]
    if landmark_id is not None:
        result += [landmark_ids, landmarks_data] if return_regions else [landmarks_data]
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)

def __get_landmark_point(
    contours: List['Something dcm'],
    ct_affine: AffineMatrix3D | None,
    use_world_coords: bool = True,
    ) -> Point3D:
    # Filter contours without data.
    contours = list(filter(lambda c: hasattr(c, 'ContourData'), contours.ContourSequence))

    if len(contours) != 1:
        raise ValueError(f"Expected contour sequence of length 1 for landmark '{i}', got {len(contours)}.")

    # Load landmark.
    contour = contours[0]
    if contour.ContourGeometricType != 'POINT':
        raise ValueError(f"Expected contour type 'POINT' for landmark data. Got '{contour.ContourGeometricType}'.")
    point = contour.ContourData

    # Convert to image coords.
    if not use_world_coords and ct_affine is not None:
        point = to_image_coords(point, ct_affine)

    return point

def __get_region_label(
    contours: List['Something dcm'],
    ct_size: Size3D,
    ct_affine: AffineMatrix3D,
    ) -> LabelImage3D:
    # Create placeholder.
    region_data = np.zeros(ct_size, dtype=bool)

    # Filter contours without data.
    contours = list(filter(lambda c: hasattr(c, 'ContourData'), contours.ContourSequence))
    contours = list(sorted(contours, key=lambda c: c.ContourData[2]))  # Sort by z position.

    # Get axial geometry.
    ct_spacing = affine_spacing(ct_affine)
    ct_origin = affine_origin(ct_affine)
    ct_size_2d, ct_spacing_2d, ct_origin_2d = list(ct_size)[:2], list(ct_spacing)[:2], list(ct_origin)[:2]

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
        slice_data = __get_region_slice_label(points_2D, ct_size_2d, ct_spacing_2d, ct_origin_2d)

        # Get z index of slice.
        z_idx = int((points[0, 2] - ct_origin[2]) / ct_spacing[2])

        # Filter slices that are outside of the CT FOV.
        if z_idx < 0 or z_idx > ct_size[2] - 1: 
            # Happened with 'PMCC-COMP:PMCC_AI_GYN_011' - Kidney_L...
            continue

        # Write slice data to label, using XOR.
        region_data[:, :, z_idx][slice_data == True] = np.invert(region_data[:, :, z_idx][slice_data == True])

    return region_data

def __get_region_slice_label(
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

def list_rtstruct_landmarks(
    rtstruct: FilePath | RtStructDicom,
    landmark_id: DiskLandmarkID | List[DiskLandmarkID] | Literal['all'] = 'all',
    landmark_regexp: RegExp | List[RegExp] | None = None,
    return_contours: bool = False,
    ) -> List[DiskLandmarkID] | Tuple[List[DiskLandmarkID], List['Something dcm']]:
    if isinstance(rtstruct, str):
        rtstruct = load_dicom(rtstruct, force=False)
    landmark_regexp = landmark_regexp or DEFAULT_LANDMARK_REGEXP
    landmark_regexps = arg_to_list(landmark_regexp, str)

    # Load all regions and contours.
    all_ids = [i.ROIName for i in rtstruct.StructureSetROISequence]
    all_contours = rtstruct.ROIContourSequence

    # Filter out regions.
    def is_landmark(r: str) -> bool:
        return any(re.match(regex, r) for regex in landmark_regexps)
    landmark_ids, landmark_contours = filter_lists([all_ids, all_contours], lambda ic: is_landmark(ic[0]))

    # Filter on the presence of a 'ContourSequence' - sometimes empty.
    landmark_ids, landmark_contours = filter_lists([landmark_ids, landmark_contours], lambda ic: getattr(ic[1], 'ContourSequence', None) is not None)

    # Filter by 'landmark_id'.
    if landmark_id != 'all':
        req_landmark_ids = arg_to_list(landmark_id, str)
        landmark_ids, landmark_contours = filter_lists([landmark_ids, landmark_contours], lambda ic: ic[0] in req_landmark_ids)

    # Sort results.
    landmark_ids, landmark_contours = sort_lists([landmark_ids, landmark_contours], key=lambda ic: ic[0])

    if return_contours:
        return landmark_ids, landmark_contours
    else:
        return landmark_ids

def list_rtstruct_regions(
    rtstruct: FilePath | RtStructDicom,
    # TODO: Expand this to handle landmark IDs also.
    # Can we just treatment landmarks (e.g. "Marker 1") as regexps
    # and match using the same logic?
    landmark_regexp: RegExp | List[RegExp] | None = None,
    region_id: DiskRegionID | List[DiskRegionID] | Literal['all'] = 'all',
    return_contours: bool = False,
    ) -> List[DiskRegionID] | Tuple[List[DiskRegionID], List['Something dcm']]:
    if isinstance(rtstruct, str):
        rtstruct = load_dicom(rtstruct, force=False)
    landmark_regexp = landmark_regexp or DEFAULT_LANDMARK_REGEXP
    landmark_regexps = arg_to_list(landmark_regexp, str)

    # Load all regions and contours.
    all_ids = [i.ROIName for i in rtstruct.StructureSetROISequence]
    all_contours = rtstruct.ROIContourSequence

    # Filter out landmarks.
    def is_region(r: str) -> bool:
        return not any(re.match(regex, r) for regex in landmark_regexps)
    region_ids, region_contours = filter_lists([all_ids, all_contours], lambda ic: is_region(ic[0]))

    # Filter on the presence of a 'ContourSequence' - sometimes empty.
    region_ids, region_contours = filter_lists([region_ids, region_contours], lambda ic: getattr(ic[1], 'ContourSequence', None) is not None)

    # Filter by 'region_id'.
    if region_id != 'all':
        req_region_ids = arg_to_list(region_id, str)
        region_ids, region_contours = filter_lists([region_ids, region_contours], lambda ic: ic[0] in req_region_ids)

    # Sort results.
    region_ids, region_contours = sort_lists([region_ids, region_contours], key=lambda ic: ic[0])

    if return_contours:
        return region_ids, region_contours
    else:
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
    