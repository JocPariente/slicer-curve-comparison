#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curve Segmentation Comparison Tool for 3D Slicer

This script computes pairwise interobserver segmentation error metrics for
markup curves in 3D Slicer. It was developed for assessing the reproducibility
of manual nerve segmentation on neurographic MRI sequences.

The script calculates:
- Mean error distance between corresponding curve points
- Maximum error distance
- Median error (50th percentile)
- Length difference between segmented curves

Additionally, it can generate:
- Volumetric segmentations from curves (exportable as NIfTI/NRRD)
- 3D surface visualizations of the nerve segmentations
- Tube models for simpler visualization

Requirements:
    - 3D Slicer (version 5.0+)
    - MarkupsToModel extension
    - NumPy
    - Pandas

Usage:
    1. Load your markup curve nodes into 3D Slicer
    2. Configure the parameters in the CONFIGURATION section below
    3. Run this script from the 3D Slicer Python console or interactor

Author: Jose Carlos Pariente
License: MIT
Version: 1.0.0
"""

import numpy as np
import slicer
import pandas as pd
from itertools import combinations
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION - Modify these parameters for your study
# =============================================================================

# Output directory for results (use forward slashes or raw strings)
OUTPUT_DIR = '/path/to/your/output/directory/'

# Curve naming convention
# Set to None to auto-detect all curves in the scene, or specify lists
# Example: If your curves are named "Observer1_Trunk_T1", "Observer2_Trunk_T1", etc.

OBSERVERS = None  # e.g., ['Observer1', 'Observer2', 'Observer3', 'Observer4'] or None for auto-detect
STRUCTURES = None  # e.g., ['Trunk', 'SuperiorDivision', 'InferiorDivision'] or None for auto-detect
CONDITIONS = None  # e.g., ['T1', 'DESS'] or None for auto-detect

# Naming pattern: how are your curve nodes named?
# Options:
#   'observer_structure_condition' -> "Observer1_Trunk_T1"
#   'structure_observer_condition' -> "Trunk_Observer1_T1"
#   'custom' -> Define your own pattern in get_curve_name()
NAMING_PATTERN = 'observer_structure_condition'

# Separator used in curve names (e.g., '_' for "Observer1_Trunk_T1")
NAME_SEPARATOR = '_'

# Resampling interval in mm (curves are resampled for uniform point spacing)
RESAMPLE_INTERVAL_MM = 1.0

# Whether to generate 3D tube models for visualization
GENERATE_TUBE_MODELS = True
TUBE_RADIUS_MM = 1.0
TUBE_INHERIT_CURVE_COLOR = True  # If True, tubes will have the same color as their source curves

# Whether to generate volumetric segmentations (3D surfaces) from curves
# This creates actual segmentation nodes that can be exported as NIfTI/NRRD
GENERATE_SEGMENTATIONS = True
SEGMENTATION_RADIUS_MM = 1.0  # Radius around curve centerline in mm (matches SegmentEditorEffect default)

# How to group segments in segmentation nodes:
# Options: 'by_observer', 'by_structure', 'by_condition', 'all_in_one'
SEGMENTATION_GROUPING = 'by_observer'

# Voxel-based segmentation parameters (creates more natural nerve-like appearance)
SEGMENTATION_VOXEL_SIZE_MM = 0.5  # Voxel size (smaller = higher resolution, slower)
SEGMENTATION_RESAMPLE_INTERVAL_MM = 0.3  # Curve resampling interval (smaller = smoother)

# Keep intermediate data (usually False to clean up)
KEEP_TUBE_MODELS = False

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_curve_name(observer, structure, condition, pattern='observer_structure_condition', sep='_'):
    """
    Generate curve node name based on naming pattern.

    Parameters
    ----------
    observer : str
        Observer/segmenter identifier
    structure : str
        Anatomical structure identifier
    condition : str
        Imaging condition/sequence identifier
    pattern : str
        Naming pattern to use
    sep : str
        Separator character

    Returns
    -------
    str
        Formatted curve node name
    """
    if pattern == 'observer_structure_condition':
        return f'{observer}{sep}{structure}{sep}{condition}'
    elif pattern == 'structure_observer_condition':
        return f'{structure}{sep}{observer}{sep}{condition}'
    elif pattern == 'condition_observer_structure':
        return f'{condition}{sep}{observer}{sep}{structure}'
    else:
        # Default pattern
        return f'{observer}{sep}{structure}{sep}{condition}'


def parse_curve_name(name, pattern='observer_structure_condition', sep='_'):
    """
    Parse curve node name to extract components.

    Parameters
    ----------
    name : str
        Curve node name
    pattern : str
        Expected naming pattern
    sep : str
        Separator character

    Returns
    -------
    dict or None
        Dictionary with 'observer', 'structure', 'condition' keys, or None if parsing fails
    """
    parts = name.split(sep)
    if len(parts) < 3:
        return None

    if pattern == 'observer_structure_condition':
        return {'observer': parts[0], 'structure': parts[1], 'condition': sep.join(parts[2:])}
    elif pattern == 'structure_observer_condition':
        return {'structure': parts[0], 'observer': parts[1], 'condition': sep.join(parts[2:])}
    elif pattern == 'condition_observer_structure':
        return {'condition': parts[0], 'observer': parts[1], 'structure': sep.join(parts[2:])}
    else:
        return {'observer': parts[0], 'structure': parts[1], 'condition': sep.join(parts[2:])}


def auto_detect_curves():
    """
    Automatically detect all markup curve nodes in the scene and extract
    unique observers, structures, and conditions.

    Returns
    -------
    tuple
        (observers, structures, conditions) - lists of unique values
    """
    observers = set()
    structures = set()
    conditions = set()

    curve_nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsCurveNode')

    for i in range(curve_nodes.GetNumberOfItems()):
        node = curve_nodes.GetItemAsObject(i)
        name = node.GetName()
        parsed = parse_curve_name(name, NAMING_PATTERN, NAME_SEPARATOR)

        if parsed:
            observers.add(parsed['observer'])
            structures.add(parsed['structure'])
            conditions.add(parsed['condition'])

    return sorted(list(observers)), sorted(list(structures)), sorted(list(conditions))


def measure_curve_difference(curve1, curve2, resample_interval=1.0):
    """
    Calculate distance-based error metrics between two curve segmentations.

    For each control point on curve1, finds the closest point on curve2 and
    computes the Euclidean distance. This provides a point-wise assessment
    of segmentation agreement.

    Parameters
    ----------
    curve1 : vtkMRMLMarkupsCurveNode
        First curve node (reference)
    curve2 : vtkMRMLMarkupsCurveNode
        Second curve node (comparison)
    resample_interval : float
        Interval in mm for curve resampling (default: 1.0)

    Returns
    -------
    dict or None
        Dictionary containing:
        - 'distances': array of all point-wise distances
        - 'mean_error': mean distance (mm)
        - 'max_error': maximum distance (mm)
        - 'median_error': median distance (mm)
        - 'percentile_75': 75th percentile distance (mm)
        - 'percentile_95': 95th percentile distance (mm)
        Returns None if curves have no control points
    """
    # Resample curves for uniform point spacing
    try:
        curve1.ResampleCurveWorld(resample_interval)
        curve2.ResampleCurveWorld(resample_interval)
    except AttributeError as e:
        print(f"Error resampling curves: {e}")
        return None

    num_points = curve1.GetNumberOfControlPoints()
    if num_points == 0:
        return None

    distances = []

    for i in range(num_points):
        # Get position of control point on curve1
        control_point_position = [0.0, 0.0, 0.0]
        curve1.GetNthControlPointPositionWorld(i, control_point_position)

        # Find closest point on curve2
        closest_point_position = [0.0, 0.0, 0.0]
        curve2.GetClosestPointPositionAlongCurveWorld(control_point_position, closest_point_position)

        # Calculate Euclidean distance
        distance = np.linalg.norm(
            np.array(control_point_position) - np.array(closest_point_position)
        )
        distances.append(distance)

    distances = np.array(distances)
    percentiles = np.percentile(distances, [50, 75, 95])

    return {
        'distances': distances,
        'mean_error': np.mean(distances),
        'max_error': np.max(distances),
        'median_error': percentiles[0],
        'percentile_75': percentiles[1],
        'percentile_95': percentiles[2]
    }


def measure_curve_length(curve_node):
    """
    Get the total length of a curve in world coordinates.

    Parameters
    ----------
    curve_node : vtkMRMLMarkupsCurveNode
        Curve node to measure

    Returns
    -------
    float
        Curve length in mm
    """
    return curve_node.GetCurveLengthWorld()


def create_tube_model(curve_node, radius=1.0, color=None, inherit_color=True):
    """
    Create a 3D tube model from a curve for visualization.

    Parameters
    ----------
    curve_node : vtkMRMLMarkupsCurveNode
        Source curve node
    radius : float
        Tube radius in mm (default: 1.0)
    color : list or None
        RGB color values [0-1] (default: None for automatic)
    inherit_color : bool
        If True and color is None, inherit color from the source curve's
        display node (default: True)

    Returns
    -------
    vtkMRMLModelNode or None
        Created model node, or None if failed
    """
    if curve_node is None:
        return None

    # Create model node
    model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    if not model_node:
        return None

    model_node.SetName(f'{curve_node.GetName()}_tube')

    # Create MarkupsToModel node
    markups_to_model = slicer.mrmlScene.AddNode(slicer.vtkMRMLMarkupsToModelNode())
    markups_to_model.SetAndObserveInputNodeID(curve_node.GetID())
    markups_to_model.SetAndObserveOutputModelNodeID(model_node.GetID())

    # Configure tube parameters
    markups_to_model.SetModelType(1)  # Tube
    markups_to_model.SetTubeRadius(radius)
    markups_to_model.SetTubeSegmentsBetweenControlPoints(5)
    markups_to_model.SetTubeNumberOfSides(8)
    markups_to_model.SetCurveType(2)  # Polynomial

    # Update model
    try:
        slicer.modules.markupstomodel.logic().UpdateOutputModel(markups_to_model)
    except Exception as e:
        print(f"Error creating tube model: {e}")
        return None

    # Get the model's display node
    model_display_node = model_node.GetDisplayNode()
    if model_display_node:
        # If explicit color provided, use it
        if color is not None:
            model_display_node.SetColor(color)
        # Otherwise, inherit color from the source curve if requested
        elif inherit_color:
            curve_display_node = curve_node.GetDisplayNode()
            if curve_display_node:
                # Get the selected color from the markup curve
                curve_color = curve_display_node.GetSelectedColor()
                model_display_node.SetColor(curve_color)

                # Optionally also inherit opacity
                opacity = curve_display_node.GetOpacity()
                model_display_node.SetOpacity(opacity)

    return model_node


def get_or_create_reference_volume(curves_dict, voxel_size=0.5):
    """
    Get an existing volume from the scene or create a minimal reference volume
    that encompasses all curves.

    Parameters
    ----------
    curves_dict : dict
        Dictionary of curves to calculate bounds from
    voxel_size : float
        Voxel size in mm for the created volume (default: 0.5mm)

    Returns
    -------
    vtkMRMLScalarVolumeNode
        Reference volume node
    """
    # First, try to find an existing volume in the scene
    volume_nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLScalarVolumeNode')
    if volume_nodes.GetNumberOfItems() > 0:
        ref_volume = volume_nodes.GetItemAsObject(0)
        print(f"Using existing volume as reference: {ref_volume.GetName()}")
        return ref_volume

    # No volume found, create a minimal one around the curves
    print("No reference volume found. Creating minimal reference volume...")

    # Calculate bounds of all curves
    min_bounds = [float('inf'), float('inf'), float('inf')]
    max_bounds = [float('-inf'), float('-inf'), float('-inf')]

    for curve in curves_dict.values():
        for i in range(curve.GetNumberOfControlPoints()):
            pos = [0.0, 0.0, 0.0]
            curve.GetNthControlPointPositionWorld(i, pos)
            for j in range(3):
                min_bounds[j] = min(min_bounds[j], pos[j])
                max_bounds[j] = max(max_bounds[j], pos[j])

    # Add padding (10mm on each side)
    padding = 10.0
    for j in range(3):
        min_bounds[j] -= padding
        max_bounds[j] += padding

    # Calculate dimensions
    dimensions = [
        int(np.ceil((max_bounds[j] - min_bounds[j]) / voxel_size)) + 1
        for j in range(3)
    ]

    # Create the volume
    import vtk
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dimensions)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Fill with zeros
    for i in range(image_data.GetNumberOfPoints()):
        image_data.GetPointData().GetScalars().SetTuple1(i, 0)

    # Create volume node
    volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    volume_node.SetName("NerveSegmentationReference")
    volume_node.SetAndObserveImageData(image_data)

    # Set the IJK to RAS transform (origin and spacing)
    ijkToRas = vtk.vtkMatrix4x4()
    ijkToRas.Identity()
    ijkToRas.SetElement(0, 0, voxel_size)  # Spacing X
    ijkToRas.SetElement(1, 1, voxel_size)  # Spacing Y
    ijkToRas.SetElement(2, 2, voxel_size)  # Spacing Z
    ijkToRas.SetElement(0, 3, min_bounds[0])  # Origin X
    ijkToRas.SetElement(1, 3, min_bounds[1])  # Origin Y
    ijkToRas.SetElement(2, 3, min_bounds[2])  # Origin Z
    volume_node.SetIJKToRASMatrix(ijkToRas)

    print(f"Created reference volume: dimensions={dimensions}, voxel_size={voxel_size}mm")
    return volume_node


def create_labelmap_from_curve(curve, reference_volume, radius=1.0, resample_interval=0.3):
    """
    Create a labelmap by painting voxels along the curve path.

    This creates a more natural, nerve-like appearance by following
    the actual curve control points rather than creating smooth tubes.

    Parameters
    ----------
    curve : vtkMRMLMarkupsCurveNode
        Source curve node
    reference_volume : vtkMRMLScalarVolumeNode
        Reference volume for voxel grid
    radius : float
        Radius around curve in mm
    resample_interval : float
        Interval for resampling the curve (smaller = smoother)

    Returns
    -------
    vtkMRMLLabelMapVolumeNode
        Labelmap volume with the curve path painted
    """
    import vtk

    # Resample curve for denser point coverage
    curve.ResampleCurveWorld(resample_interval)

    # Get reference volume properties
    ras_to_ijk = vtk.vtkMatrix4x4()
    reference_volume.GetRASToIJKMatrix(ras_to_ijk)

    dims = reference_volume.GetImageData().GetDimensions()
    spacing = reference_volume.GetSpacing()

    # Create labelmap
    labelmap_data = vtk.vtkImageData()
    labelmap_data.SetDimensions(dims)
    labelmap_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Initialize to zeros
    scalars = labelmap_data.GetPointData().GetScalars()
    for i in range(labelmap_data.GetNumberOfPoints()):
        scalars.SetTuple1(i, 0)

    # Calculate radius in voxels for each dimension
    radius_voxels = [int(np.ceil(radius / s)) for s in spacing]

    # For each control point, paint a sphere of voxels
    num_points = curve.GetNumberOfControlPoints()
    for pt_idx in range(num_points):
        # Get RAS position
        ras_pos = [0.0, 0.0, 0.0]
        curve.GetNthControlPointPositionWorld(pt_idx, ras_pos)

        # Convert to IJK
        ras_pos_homogeneous = [ras_pos[0], ras_pos[1], ras_pos[2], 1.0]
        ijk_pos = ras_to_ijk.MultiplyPoint(ras_pos_homogeneous)
        ijk_center = [int(round(ijk_pos[j])) for j in range(3)]

        # Paint voxels within radius
        for di in range(-radius_voxels[0], radius_voxels[0] + 1):
            for dj in range(-radius_voxels[1], radius_voxels[1] + 1):
                for dk in range(-radius_voxels[2], radius_voxels[2] + 1):
                    i = ijk_center[0] + di
                    j = ijk_center[1] + dj
                    k = ijk_center[2] + dk

                    # Check bounds
                    if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
                        # Calculate actual distance in mm
                        dist_mm = np.sqrt(
                            (di * spacing[0]) ** 2 +
                            (dj * spacing[1]) ** 2 +
                            (dk * spacing[2]) ** 2
                        )
                        if dist_mm <= radius:
                            idx = i + j * dims[0] + k * dims[0] * dims[1]
                            scalars.SetTuple1(idx, 1)

    labelmap_data.Modified()

    # Create labelmap node
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmap_node.SetName(f"{curve.GetName()}_labelmap")
    labelmap_node.SetAndObserveImageData(labelmap_data)

    # Copy geometry from reference
    ijkToRas = vtk.vtkMatrix4x4()
    reference_volume.GetIJKToRASMatrix(ijkToRas)
    labelmap_node.SetIJKToRASMatrix(ijkToRas)

    return labelmap_node


def create_segmentation_from_curves(curves_dict, grouping='by_observer', radius=1.0,
                                    inherit_color=True, keep_models=False,
                                    voxel_size=0.5, resample_interval=0.3):
    """
    Create volumetric segmentations from markup curves using voxel-based approach.

    This creates segmentations that follow the actual curve path, giving a more
    natural, nerve-like appearance compared to smooth tube models.

    Parameters
    ----------
    curves_dict : dict
        Dictionary with keys as (observer, structure, condition) tuples
        and values as curve nodes
    grouping : str
        How to group segments: 'by_observer', 'by_structure', 'by_condition', 'all_in_one'
    radius : float
        Radius around curve centerline in mm
    inherit_color : bool
        If True, segments inherit colors from source curves
    keep_models : bool
        Not used in voxel-based approach (kept for API compatibility)
    voxel_size : float
        Voxel size in mm for the reference volume (smaller = higher resolution)
    resample_interval : float
        Interval for curve resampling (smaller = smoother segmentation)

    Returns
    -------
    dict
        Dictionary of created segmentation nodes
    """
    segmentation_nodes = {}
    temp_labelmaps = []

    # Get or create reference volume
    reference_volume = get_or_create_reference_volume(curves_dict, voxel_size)

    # Group curves according to the specified grouping
    grouped_curves = {}
    for (observer, structure, condition), curve in curves_dict.items():
        if grouping == 'by_observer':
            group_key = observer
        elif grouping == 'by_structure':
            group_key = structure
        elif grouping == 'by_condition':
            group_key = condition
        else:  # all_in_one
            group_key = 'AllNerves'

        if group_key not in grouped_curves:
            grouped_curves[group_key] = []
        grouped_curves[group_key].append({
            'curve': curve,
            'observer': observer,
            'structure': structure,
            'condition': condition
        })

    # Create segmentation node for each group
    for group_key, curve_list in grouped_curves.items():
        # Create segmentation node
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentation_node.SetName(f"Segmentation_{group_key}")
        segmentation_node.CreateDefaultDisplayNodes()

        # Set reference geometry from reference volume
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume)

        for curve_info in curve_list:
            curve = curve_info['curve']
            obs = curve_info['observer']
            struct = curve_info['structure']
            cond = curve_info['condition']

            print(f"  Processing: {obs}_{struct}_{cond}")

            # Create labelmap from curve
            labelmap = create_labelmap_from_curve(
                curve, reference_volume, radius, resample_interval
            )

            if labelmap is None:
                print(f"Warning: Could not create labelmap for {curve.GetName()}")
                continue

            temp_labelmaps.append(labelmap)

            # Get color from curve
            segment_color = [0.5, 0.5, 0.5]  # Default gray
            if inherit_color:
                curve_display = curve.GetDisplayNode()
                if curve_display:
                    segment_color = list(curve_display.GetSelectedColor())

            # Track segment count before import
            segmentation = segmentation_node.GetSegmentation()
            num_segments_before = segmentation.GetNumberOfSegments()

            # Import labelmap as segment
            segment_name = f"{obs}_{struct}_{cond}"
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelmap, segmentation_node
            )

            # Get the newly added segment and set its name and color
            num_segments_after = segmentation.GetNumberOfSegments()
            if num_segments_after > num_segments_before:
                new_segment_id = segmentation.GetNthSegmentID(num_segments_after - 1)
                segment = segmentation.GetSegment(new_segment_id)
                if segment:
                    segment.SetName(segment_name)
                    segment.SetColor(segment_color[0], segment_color[1], segment_color[2])

        # Create closed surface representation and enable 3D visibility ("Show 3D")
        segmentation_node.CreateClosedSurfaceRepresentation()

        # Get display node and enable 3D visibility for all segments
        seg_display_node = segmentation_node.GetDisplayNode()
        if seg_display_node:
            seg_display_node.SetVisibility(True)
            seg_display_node.SetVisibility3D(True)
            seg_display_node.SetVisibility2D(True)
            # Set all segments visible in 3D
            segmentation = segmentation_node.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                segment_id = segmentation.GetNthSegmentID(i)
                seg_display_node.SetSegmentVisibility3D(segment_id, True)

        segmentation_nodes[group_key] = segmentation_node
        print(f"Created segmentation: {segmentation_node.GetName()} with "
              f"{segmentation_node.GetSegmentation().GetNumberOfSegments()} segments (3D enabled)")

    # Clean up temporary labelmaps
    for labelmap in temp_labelmaps:
        slicer.mrmlScene.RemoveNode(labelmap)
    print(f"Cleaned up {len(temp_labelmaps)} temporary labelmaps")

    return segmentation_nodes


def compare_all_curves(observers, structures, conditions, output_dir):
    """
    Perform pairwise comparison of all curve segmentations.

    Parameters
    ----------
    observers : list
        List of observer identifiers
    structures : list
        List of structure identifiers
    conditions : list
        List of condition identifiers
    output_dir : str
        Directory for output files

    Returns
    -------
    dict
        Results dictionary with all comparisons
    """
    results = {}
    missing_curves = []

    # Compare all observer pairs for each structure and condition
    for structure in structures:
        results[structure] = {}

        for condition in conditions:
            results[structure][condition] = {}

            # Get all valid curves for this structure/condition
            valid_curves = {}
            for observer in observers:
                curve_name = get_curve_name(observer, structure, condition,
                                           NAMING_PATTERN, NAME_SEPARATOR)
                curve = slicer.mrmlScene.GetFirstNodeByName(curve_name)

                if curve is not None:
                    valid_curves[observer] = curve
                else:
                    missing_curves.append(curve_name)

            # Pairwise comparison of all observers
            for obs1, obs2 in combinations(valid_curves.keys(), 2):
                curve1 = valid_curves[obs1]
                curve2 = valid_curves[obs2]

                # Measure differences
                diff_results = measure_curve_difference(curve1, curve2, RESAMPLE_INTERVAL_MM)

                if diff_results is None:
                    print(f"Warning: Could not compare {obs1} vs {obs2} for {structure}/{condition}")
                    continue

                # Measure lengths
                length1 = measure_curve_length(curve1)
                length2 = measure_curve_length(curve2)

                # Store results
                comparison_key = f'{obs1}_vs_{obs2}'
                results[structure][condition][comparison_key] = {
                    'Mean Error (mm)': round(diff_results['mean_error'], 3),
                    'Max Error (mm)': round(diff_results['max_error'], 3),
                    'Median Error (mm)': round(diff_results['median_error'], 3),
                    'P75 Error (mm)': round(diff_results['percentile_75'], 3),
                    'P95 Error (mm)': round(diff_results['percentile_95'], 3),
                    'Length 1 (mm)': round(length1, 3),
                    'Length 2 (mm)': round(length2, 3),
                    'Length Difference (mm)': round(abs(length1 - length2), 3)
                }

    return results, missing_curves


def compare_conditions(observers, structures, conditions, output_dir):
    """
    Compare segmentations across different conditions (e.g., imaging sequences)
    for the same observer.

    Parameters
    ----------
    observers : list
        List of observer identifiers
    structures : list
        List of structure identifiers
    conditions : list
        List of condition identifiers (must have at least 2)
    output_dir : str
        Directory for output files

    Returns
    -------
    dict
        Results dictionary with condition comparisons
    """
    if len(conditions) < 2:
        print("Need at least 2 conditions for comparison")
        return {}

    results = {}

    for structure in structures:
        for observer in observers:
            # Compare all condition pairs
            for cond1, cond2 in combinations(conditions, 2):
                curve_name1 = get_curve_name(observer, structure, cond1,
                                            NAMING_PATTERN, NAME_SEPARATOR)
                curve_name2 = get_curve_name(observer, structure, cond2,
                                            NAMING_PATTERN, NAME_SEPARATOR)

                curve1 = slicer.mrmlScene.GetFirstNodeByName(curve_name1)
                curve2 = slicer.mrmlScene.GetFirstNodeByName(curve_name2)

                if curve1 is None or curve2 is None:
                    continue

                diff_results = measure_curve_difference(curve1, curve2, RESAMPLE_INTERVAL_MM)

                if diff_results is None:
                    continue

                length1 = measure_curve_length(curve1)
                length2 = measure_curve_length(curve2)

                key = f'{observer}_{structure}_{cond1}_vs_{cond2}'
                results[key] = {
                    'Observer': observer,
                    'Structure': structure,
                    'Condition 1': cond1,
                    'Condition 2': cond2,
                    'Mean Error (mm)': round(diff_results['mean_error'], 3),
                    'Max Error (mm)': round(diff_results['max_error'], 3),
                    'Median Error (mm)': round(diff_results['median_error'], 3),
                    'Length Difference (mm)': round(abs(length1 - length2), 3)
                }

    return results


def save_results(results, missing_curves, condition_results, output_dir):
    """
    Save all results to CSV files.

    Parameters
    ----------
    results : dict
        Interobserver comparison results
    missing_curves : list
        List of missing curve names
    condition_results : dict
        Condition comparison results
    output_dir : str
        Output directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save interobserver results
    if results:
        # Flatten nested dictionary for DataFrame
        flat_results = {}
        for structure in results:
            for condition in results[structure]:
                for comparison in results[structure][condition]:
                    key = (structure, condition, comparison)
                    flat_results[key] = results[structure][condition][comparison]

        if flat_results:
            df = pd.DataFrame.from_dict(flat_results, orient='index')
            df.index = pd.MultiIndex.from_tuples(df.index,
                                                  names=['Structure', 'Condition', 'Comparison'])
            output_file = os.path.join(output_dir, f'interobserver_comparison_{timestamp}.csv')
            df.to_csv(output_file)
            print(f"Saved interobserver results to: {output_file}")

    # Save missing curves
    if missing_curves:
        unique_missing = list(set(missing_curves))
        missing_df = pd.DataFrame(unique_missing, columns=['Missing Curves'])
        output_file = os.path.join(output_dir, f'missing_curves_{timestamp}.csv')
        missing_df.to_csv(output_file, index=False)
        print(f"Saved missing curves list to: {output_file}")

    # Save condition comparison results
    if condition_results:
        df_cond = pd.DataFrame.from_dict(condition_results, orient='index')
        output_file = os.path.join(output_dir, f'condition_comparison_{timestamp}.csv')
        df_cond.to_csv(output_file)
        print(f"Saved condition comparison results to: {output_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_analysis():
    """
    Main function to run the complete analysis.
    """
    print("=" * 60)
    print("Curve Segmentation Comparison Tool")
    print("=" * 60)

    # Determine observers, structures, and conditions
    if OBSERVERS is None or STRUCTURES is None or CONDITIONS is None:
        print("\nAuto-detecting curves in scene...")
        detected_observers, detected_structures, detected_conditions = auto_detect_curves()

        observers = OBSERVERS if OBSERVERS is not None else detected_observers
        structures = STRUCTURES if STRUCTURES is not None else detected_structures
        conditions = CONDITIONS if CONDITIONS is not None else detected_conditions
    else:
        observers = OBSERVERS
        structures = STRUCTURES
        conditions = CONDITIONS

    print(f"\nObservers: {observers}")
    print(f"Structures: {structures}")
    print(f"Conditions: {conditions}")

    if not observers or not structures or not conditions:
        print("\nError: No valid curves detected. Please check naming convention.")
        return

    # Run interobserver comparison
    print("\nRunning interobserver comparison...")
    results, missing_curves = compare_all_curves(observers, structures, conditions, OUTPUT_DIR)

    # Run condition comparison (if multiple conditions)
    print("\nRunning condition comparison...")
    condition_results = compare_conditions(observers, structures, conditions, OUTPUT_DIR)

    # Collect all valid curves for visualization
    all_curves = {}
    for structure in structures:
        for condition in conditions:
            for observer in observers:
                curve_name = get_curve_name(observer, structure, condition,
                                           NAMING_PATTERN, NAME_SEPARATOR)
                curve = slicer.mrmlScene.GetFirstNodeByName(curve_name)
                if curve:
                    all_curves[(observer, structure, condition)] = curve

    # Generate volumetric segmentations if requested (preferred over tube models)
    if GENERATE_SEGMENTATIONS:
        print("\nGenerating volumetric segmentations for 3D visualization...")
        print(f"Grouping: {SEGMENTATION_GROUPING}")
        print(f"Voxel size: {SEGMENTATION_VOXEL_SIZE_MM}mm, Radius: {SEGMENTATION_RADIUS_MM}mm")
        segmentation_nodes = create_segmentation_from_curves(
            all_curves,
            grouping=SEGMENTATION_GROUPING,
            radius=SEGMENTATION_RADIUS_MM,
            inherit_color=TUBE_INHERIT_CURVE_COLOR,
            keep_models=KEEP_TUBE_MODELS,
            voxel_size=SEGMENTATION_VOXEL_SIZE_MM,
            resample_interval=SEGMENTATION_RESAMPLE_INTERVAL_MM
        )
        print(f"Created {len(segmentation_nodes)} segmentation node(s)")

    # Generate tube models if requested (and not already created via segmentation workflow)
    elif GENERATE_TUBE_MODELS:
        print("\nGenerating tube models for visualization...")
        for (observer, structure, condition), curve in all_curves.items():
            create_tube_model(curve, TUBE_RADIUS_MM,
                             inherit_color=TUBE_INHERIT_CURVE_COLOR)

    # Save results
    print("\nSaving results...")
    save_results(results, missing_curves, condition_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return results, condition_results


# Run if executed directly
if __name__ == '__main__':
    run_analysis()
