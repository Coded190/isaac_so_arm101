# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Palm tree dimension randomization."""

import random
import numpy as np
from pxr import Usd, UsdGeom, Gf

from .config import (
    GIRTH_SCALE_RANGE,
    HEIGHT_SCALE_RANGE,
    CROWN_SHAFT_HEIGHT_RANGE,
    CROWN_SHAFT_GIRTH_RANGE,
    CANOPY_MULTIPLIER_RANGE,
    LEAF_VARIANCE_RANGE,
)


class PalmRandomizer:
    """Randomizes palm tree dimensions for variety in training data."""
    
    def __init__(self, stage, debug_verbose: bool = False):
        """Initialize palm randomizer.
        
        Args:
            stage: USD stage
            debug_verbose: Enable verbose console output
        """
        self.stage = stage
        self.debug_verbose = debug_verbose
    
    def randomize_palm_dimensions(self, palm_root_path: str) -> None:
        """Safely randomize height, girth, crown base, and canopy size.
        
        Args:
            palm_root_path: Path to palm tree root prim
        """
        palm_prim = self.stage.GetPrimAtPath(palm_root_path)
        if not palm_prim or not palm_prim.IsValid():
            return
        
        # 1. OVERALL TREE HEIGHT & MAIN TRUNK DIAMETER
        girth_scale = random.uniform(*GIRTH_SCALE_RANGE)
        height_scale = random.uniform(*HEIGHT_SCALE_RANGE)
        
        root_xform = UsdGeom.Xformable(palm_prim)
        scale_op = self._get_or_create_scale_op(root_xform)
        scale_op.Set(Gf.Vec3d(girth_scale, girth_scale, height_scale))
        
        # 2. FIND CROWN
        crown_prim = self._find_crown_prim(palm_prim)
        if not crown_prim:
            return
        
        # 3. TRUNK TOP (CROWN BASE) RANDOMIZATION
        self._randomize_crown_base(crown_prim)
        
        # 4. CANOPY & LEAF RANDOMIZATION
        self._randomize_canopy(crown_prim)
    
    @staticmethod
    def _get_or_create_scale_op(xformable):
        """Get existing scale operation or create new one."""
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                return op
        return xformable.AddScaleOp()
    
    @staticmethod
    def _find_crown_prim(palm_prim):
        """Find the crown child primitive."""
        for child in Usd.PrimRange(palm_prim):
            if child.GetName().lower() == "crown":
                return child
        return None
    
    def _randomize_crown_base(self, crown_prim) -> None:
        """Randomize trunk top (crown base) dimensions."""
        trunk_top_path = f"{crown_prim.GetPath()}/trunk_top"
        trunk_top_prim = self.stage.GetPrimAtPath(trunk_top_path)
        
        if not trunk_top_prim.IsValid():
            return
        
        crown_shaft_height = random.uniform(*CROWN_SHAFT_HEIGHT_RANGE)
        crown_shaft_girth = random.uniform(*CROWN_SHAFT_GIRTH_RANGE)
        
        tt_xform = UsdGeom.Xformable(trunk_top_prim)
        tt_scale_op = self._get_or_create_scale_op(tt_xform)
        tt_scale_op.Set(Gf.Vec3d(crown_shaft_girth, crown_shaft_girth, crown_shaft_height))
    
    def _randomize_canopy(self, crown_prim) -> None:
        """Randomize canopy and leaf dimensions."""
        canopy_multiplier = random.uniform(*CANOPY_MULTIPLIER_RANGE)
        
        for leaf_prim in crown_prim.GetChildren():
            if "leaf" in leaf_prim.GetName().lower():
                individual_leaf_variance = random.uniform(*LEAF_VARIANCE_RANGE)
                final_leaf_scale = canopy_multiplier * individual_leaf_variance
                
                leaf_xform = UsdGeom.Xformable(leaf_prim)
                l_scale_op = self._get_or_create_scale_op(leaf_xform)
                l_scale_op.Set(Gf.Vec3d(final_leaf_scale, final_leaf_scale, final_leaf_scale))
