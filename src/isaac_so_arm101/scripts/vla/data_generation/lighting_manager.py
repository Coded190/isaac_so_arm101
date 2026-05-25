# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""HDRI and dome light management for photorealistic rendering."""

import os
import random
from typing import Optional, List
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Sdf, Gf

from .config import (
    HDRI_INTENSITY_RANGE,
    LIGHTING_SPECULAR_MULTIPLIER,
    LIGHTING_DIFFUSE_BOOST,
    AMBIENT_FILL_INTENSITY,
    DEBUG_VERBOSE,
)


class LightingManager:
    """Manages environment lighting with HDRI and dome lights."""
    
    def __init__(self, stage, debug_verbose: bool = False):
        """Initialize lighting manager.
        
        Args:
            stage: USD stage
            debug_verbose: Enable verbose console output
        """
        self.stage = stage
        self.debug_verbose = debug_verbose
    
    def randomize_lighting(self, hdri_folder_path: str, env_ids: Optional[List[int]] = None) -> None:
        """Assign one random HDRI to every env-local DomeLight.
        
        Args:
            hdri_folder_path: Path to folder containing HDRI files (.hdr, .exr)
            env_ids: List of environment IDs to update (None = all)
        """
        # Verify HDRI folder and get available files
        if not os.path.exists(hdri_folder_path):
            if self.debug_verbose:
                print(f"[WARNING] HDRI folder not found: {hdri_folder_path}", flush=True)
            return
        
        valid_exts = (".hdr", ".exr")
        hdri_files = [f for f in os.listdir(hdri_folder_path) if f.endswith(valid_exts)]
        if not hdri_files:
            if self.debug_verbose:
                print(f"[WARNING] No HDRI files (.hdr/.exr) found in {hdri_folder_path}", flush=True)
            return
        
        chosen_hdri = random.choice(hdri_files)
        full_path = os.path.join(hdri_folder_path, chosen_hdri)
        
        if not os.path.exists(full_path):
            if self.debug_verbose:
                print(f"[WARNING] HDRI file not found: {full_path}", flush=True)
            return
        
        target_intensity = random.uniform(*HDRI_INTENSITY_RANGE)
        
        if self.debug_verbose:
            print(f"[randomize_lighting] chosen_hdri={chosen_hdri}", flush=True)
        
        # Auto-detect environment IDs if not provided
        if env_ids is None:
            env_ids = self._detect_env_ids()
        
        # Collect and update valid environment lights
        valid_env_lights = []
        for env_id in env_ids:
            light_path = f"/World/envs/env_{env_id}/Scene/DomeLight"
            prim = self.stage.GetPrimAtPath(light_path)
            if prim and prim.IsA(UsdLux.DomeLight):
                valid_env_lights.append((env_id, prim))
        
        if not valid_env_lights:
            if self.debug_verbose:
                print("[WARNING] No env-local DomeLight prims were found to update.", flush=True)
            return
        
        # Calculate lighting values
        num_envs = len(valid_env_lights)
        fractional_multiplier = 1.0 / num_envs
        specular_value = fractional_multiplier * LIGHTING_SPECULAR_MULTIPLIER
        diffuse_value = fractional_multiplier * LIGHTING_DIFFUSE_BOOST
        
        # Apply HDRI to each environment
        for env_id, prim in valid_env_lights:
            self._apply_hdri_to_light(prim, full_path, target_intensity, specular_value, diffuse_value)
        
        # Create global ambient fill light
        self._create_global_ambient(target_intensity)
    
    def _detect_env_ids(self) -> List[int]:
        """Auto-detect all environment IDs in the stage."""
        env_ids = []
        envs_root = self.stage.GetPrimAtPath("/World/envs")
        if envs_root:
            for env_prim in envs_root.GetChildren():
                name = env_prim.GetName()
                if name.startswith("env_"):
                    suffix = name.split("env_", 1)[1]
                    if suffix.isdigit():
                        env_ids.append(int(suffix))
        return env_ids
    
    def _apply_hdri_to_light(self, prim, hdri_path: str, intensity: float,
                            specular: float, diffuse: float) -> None:
        """Configure a dome light with HDRI texture.
        
        Args:
            prim: USD dome light primitive
            hdri_path: Path to HDRI texture file
            intensity: Light intensity
            specular: Specular component value
            diffuse: Diffuse component value
        """
        light = UsdLux.DomeLight(prim)
        
        # Make visible
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible()
        
        # Set visibility attribute
        try:
            vis_attr = prim.GetAttribute("visibility")
            if not vis_attr:
                vis_attr = prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token)
            vis_attr.Set("inherited")
        except Exception:
            pass
        
        # Apply HDRI texture and intensity
        light.GetTextureFileAttr().Set(hdri_path)
        light.GetIntensityAttr().Set(intensity)
        
        # Set diffuse and specular
        diffuse_attr = light.GetDiffuseAttr()
        if diffuse_attr:
            diffuse_attr.Set(diffuse)
        else:
            light.CreateDiffuseAttr(diffuse)
        
        specular_attr = light.GetSpecularAttr()
        if specular_attr:
            specular_attr.Set(specular)
        else:
            light.CreateSpecularAttr(specular)
        
        # Set texture format
        format_attr = light.GetTextureFormatAttr()
        if format_attr:
            format_attr.Set("latlong")
        else:
            light.CreateTextureFormatAttr("latlong")
        
        # Set exposure
        exposure_attr = light.GetExposureAttr()
        if exposure_attr:
            exposure_attr.Set(0.0)
        else:
            light.CreateExposureAttr(0.0)
        
        # Set color space
        try:
            colorspace_attr = prim.GetAttribute("inputs:texture:colorSpace")
            if not colorspace_attr:
                colorspace_attr = prim.CreateAttribute("inputs:texture:colorSpace", Sdf.ValueTypeNames.String)
            colorspace_attr.Set("sRGB")
        except Exception:
            pass
    
    def _create_global_ambient(self, intensity: float) -> None:
        """Create global ambient fill light outside of clones.
        
        Args:
            intensity: Light intensity
        """
        global_ambient_path = "/World/GlobalAmbientFill"
        ambient_prim = self.stage.GetPrimAtPath(global_ambient_path)
        
        if not ambient_prim:
            ambient_light = UsdLux.DistantLight.Define(self.stage, global_ambient_path)
        else:
            ambient_light = UsdLux.DistantLight(ambient_prim)
        
        ambient_light.GetIntensityAttr().Set(AMBIENT_FILL_INTENSITY)
        ambient_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        
        shadow_attr = ambient_light.GetPrim().GetAttribute("inputs:shadow:enable")
        if not shadow_attr:
            shadow_attr = ambient_light.GetPrim().CreateAttribute("inputs:shadow:enable", Sdf.ValueTypeNames.Bool)
        shadow_attr.Set(False)
