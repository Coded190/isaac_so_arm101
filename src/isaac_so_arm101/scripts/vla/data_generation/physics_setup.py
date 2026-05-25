# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics configuration for palm trees."""

from pxr import Usd, UsdGeom, UsdPhysics

from .geometry_utils import iter_leaf_prims
from .config import PALM_LEAF_MASS, JOINT_STIFFNESS_OVERRIDE, JOINT_DAMPING_OVERRIDE


class PhysicsSetup:
    """Configures physics properties for palm trees."""
    
    def __init__(self, stage, debug_verbose: bool = False):
        """Initialize physics setup.
        
        Args:
            stage: USD stage
            debug_verbose: Enable verbose console output
        """
        self.stage = stage
        self.debug_verbose = debug_verbose
    
    def disable_palm_physics(self, palm_root_path: str) -> None:
        """Update palm tree leaves to be lightweight and loosen their joints.
        
        Args:
            palm_root_path: Path to palm tree root
        """
        palm = self.stage.GetPrimAtPath(palm_root_path)
        if not palm:
            return
        
        crown = self._get_palm_crown_prim(palm_root_path)
        crown_paths = set()
        if crown:
            for prim in Usd.PrimRange(crown):
                crown_paths.add(prim.GetPath())
        
        # Traverse palm subtree and configure physics
        for child in Usd.PrimRange(palm):
            if child == palm:
                continue
            
            prim_name = child.GetName().lower()
            prim_path = child.GetPath()
            
            # Make leaves lightweight and dynamic
            if crown and prim_path in crown_paths and child.IsA(UsdGeom.Mesh) and (
                prim_name.startswith("leaf_") or prim_name.startswith("leaf_b_")
            ):
                self._configure_leaf_physics(child)
            
            # Keep trunk immovable
            elif prim_name in {"trunk", "trunk_top"}:
                self._make_kinematic(child)
            
            # Neutralize joints
            elif child.IsA(UsdPhysics.Joint):
                self._neutralize_joint(child)
    
    @staticmethod
    def _configure_leaf_physics(prim) -> None:
        """Configure physics for a leaf primitive."""
        # Make dynamic (not kinematic)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = UsdPhysics.RigidBodyAPI(prim)
        else:
            rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        
        kin_attr = rb_api.GetKinematicEnabledAttr()
        if kin_attr:
            kin_attr.Set(False)
        else:
            rb_api.CreateKinematicEnabledAttr(False)
        
        # Set light mass
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
        else:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
        
        mass_attr = mass_api.GetMassAttr()
        if mass_attr:
            mass_attr.Set(PALM_LEAF_MASS)
        else:
            mass_api.CreateMassAttr(PALM_LEAF_MASS)
        
        # Enable collision
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col_api = UsdPhysics.CollisionAPI(prim)
            col_attr = col_api.GetCollisionEnabledAttr()
            if col_attr:
                col_attr.Set(True)
            else:
                col_api.CreateCollisionEnabledAttr(True)
    
    @staticmethod
    def _make_kinematic(prim) -> None:
        """Make a primitive kinematic (immovable)."""
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = UsdPhysics.RigidBodyAPI(prim)
            kin_attr = rb_api.GetKinematicEnabledAttr()
            if kin_attr:
                kin_attr.Set(True)
            else:
                rb_api.CreateKinematicEnabledAttr(True)
    
    @staticmethod
    def _neutralize_joint(joint_prim) -> None:
        """Reduce joint stiffness and damping to allow free bending."""
        for prop in joint_prim.GetAuthoredProperties():
            prop_name = prop.GetName().lower()
            
            # Kill spring forces
            if "stiffness" in prop_name:
                prop.Set(JOINT_STIFFNESS_OVERRIDE)
            
            # Add damping to prevent infinite vibration
            elif "damping" in prop_name:
                prop.Set(JOINT_DAMPING_OVERRIDE)
    
    @staticmethod
    def _get_palm_crown_prim(palm_root_path: str):
        """Get crown primitive from palm root."""
        from .geometry_utils import _get_palm_crown_prim
        stage = Usd.Stage.Open(palm_root_path.split("/")[0])
        return _get_palm_crown_prim(stage, palm_root_path)
