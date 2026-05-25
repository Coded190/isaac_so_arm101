# Data Generation Refactoring - Implementation Guide

## Quick Start

### 📖 Read First
1. **REFACTORING_SUMMARY.md** - High-level overview (5 min read)
2. **REFACTORING_GUIDE.md** - Detailed explanation (10 min read)

### 🚀 Get Started
```bash
# The refactored code is ready to use
# Option 1: Use the refactored version directly
python generate_data_refactored.py --task Isaac-PING-TI-VLA-v0 --num_envs 10 --save_data

# Option 2: Keep original file, import modules individually
from leaf_manager import LeafManager
from robot_controller import RobotController

stage = ...
leaf_manager = LeafManager(stage)
robot_controller = RobotController(stage)
```

## File Organization

### Configuration
```
config.py          # All constants (125 lines)
└─ Centralized configuration for all modules
```

### Utilities
```
math_utils.py      # Quaternion/vector math (110 lines)
geometry_utils.py  # Spatial calculations (95 lines)
noise_filter.py    # Output filtering (45 lines)
└─ Pure functions and utilities
```

### Feature Managers
```
leaf_manager.py      # Leaf operations (95 lines)
├─ set_leaf_prims_active()
├─ remove_top_leaves()
└─ cull_episode_leaves()

palm_randomizer.py   # Tree randomization (105 lines)
├─ randomize_palm_dimensions()
└─ Internal helpers

physics_setup.py     # Physics config (120 lines)
├─ disable_palm_physics()
└─ Internal helpers

lighting_manager.py  # Lighting setup (180 lines)
├─ randomize_lighting()
└─ Internal helpers

camera_controller.py # Camera control (85 lines)
├─ update_recording_camera()
└─ Internal helpers

robot_controller.py  # Robot control (200 lines)
├─ set_rest_pose()
├─ randomize_robot_root_pose()
├─ get_deterministic_target()
└─ prepare_episode_targets()

spray_oracle.py      # FSM logic (140 lines)
├─ compute_action()
└─ _advance()

dataset_manager.py   # Dataset recording (150 lines)
├─ initialize_datasets()
├─ add_frame()
├─ save_episode()
└─ finalize()
```

### Main Entry Point
```
generate_data_refactored.py  # Orchestrator (280 lines)
├─ setup_environment()
└─ main()
```

### Documentation
```
REFACTORING_SUMMARY.md   # High-level overview
REFACTORING_GUIDE.md     # Detailed documentation
README.md                # This file
```

## Before vs After

### Before (Monolithic)
```python
# generate_data.py (2000+ lines)

# Everything mixed together
def randomize_lighting(...): ...          # Line ~400
def disable_palm_physics(...): ...        # Line ~500
def set_leaf_prims_active(...): ...       # Line ~600
def randomize_robot_root_pose(...): ...   # Line ~900
def randomize_palm_dimensions(...): ...   # Line ~1100
class SprayOracle: ...                    # Line ~1300
# ... and 50+ more functions/classes

def main():
    # 800 lines of orchestration logic
    ...
```

### After (Modular)
```python
# generate_data_refactored.py (280 lines)

from leaf_manager import LeafManager
from palm_randomizer import PalmRandomizer
from physics_setup import PhysicsSetup
from lighting_manager import LightingManager
from camera_controller import CameraController
from robot_controller import RobotController
from spray_oracle import SprayOracle
from dataset_manager import DatasetManager

def setup_environment():
    # 50 lines of focused environment setup

def main():
    # 200 lines of clean orchestration
    
    # Initialize managers
    leaf_manager = LeafManager(stage)
    physics_setup = PhysicsSetup(stage)
    robot_controller = RobotController(stage)
    # ... etc
    
    # Use clean interfaces
    leaf_manager.cull_episode_leaves(...)
    physics_setup.disable_palm_physics(...)
    robot_controller.randomize_robot_root_pose(...)
```

## Module API Reference

### LeafManager
```python
manager = LeafManager(stage, debug_verbose=False)

# Methods
manager.set_leaf_prims_active(palm_root_path, active=True)
manager.remove_top_leaves(palm_root_path, crown_z, keep_ratio)
manager.cull_episode_leaves(palm_root_paths, episode_rng, cull_prob, env_ids=None)
```

### PalmRandomizer
```python
randomizer = PalmRandomizer(stage, debug_verbose=False)

# Methods
randomizer.randomize_palm_dimensions(palm_root_path)
```

### PhysicsSetup
```python
physics = PhysicsSetup(stage, debug_verbose=False)

# Methods
physics.disable_palm_physics(palm_root_path)
```

### LightingManager
```python
lighting = LightingManager(stage, debug_verbose=False)

# Methods
lighting.randomize_lighting(hdri_folder_path, env_ids=None)
```

### CameraController
```python
camera = CameraController(debug_verbose=False)

# Methods
camera.update_recording_camera(robot_xy, crown_xy, base_z, episode_rng, ...)
```

### RobotController
```python
controller = RobotController(stage, debug_verbose=False)

# Methods
controller.set_rest_pose(env, rest_pose_tensor, env_ids=None, noise_scale=0.05)
controller.randomize_robot_root_pose(env, palm_root_paths, episode_rng, env_ids=None)
controller.get_deterministic_target(stage, palm_root_path)
controller.prepare_episode_targets(palm_root_paths, robot_xys=None, env_ids=None)
```

### SprayOracle
```python
oracle = SprayOracle()

# Methods
action = oracle.compute_action(ee_pos, ee_quat, hover_target)
oracle.reset()

# Properties
oracle.state        # Current FSM state (0-5)
oracle.completed    # Episode completed successfully
oracle.timed_out    # Episode timed out
```

### DatasetManager
```python
manager = DatasetManager(dataset_root, fps=30, debug_verbose=False)

# Methods
manager.initialize_datasets(num_envs, num_dof, img_shape)
manager.add_frame(env_id, joint_positions, ee_position, ee_quaternion, camera_image, action)
manager.save_episode(env_id, save=True)
manager.finalize()
```

## Usage Examples

### Example 1: Simple randomization
```python
from leaf_manager import LeafManager
from config import LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX

stage = ...
manager = LeafManager(stage)
rng = np.random.default_rng()

# Cull leaves for all environments
manager.cull_episode_leaves(
    palm_root_paths=["/World/envs/env_0/Scene/palm_tree_crown"],
    episode_rng=rng,
    cull_prob=0.5,
)
```

### Example 2: Robot positioning
```python
from robot_controller import RobotController

controller = RobotController(stage)
rng = np.random.default_rng()

# Position robot
controller.randomize_robot_root_pose(
    env=env,
    palm_root_paths=palm_paths,
    episode_rng=rng,
)

# Get hover targets
targets = controller.prepare_episode_targets(
    palm_root_paths=palm_paths,
    robot_xys=robot_positions,
)
```

### Example 3: Complete setup
```python
from leaf_manager import LeafManager
from physics_setup import PhysicsSetup
from lighting_manager import LightingManager
from robot_controller import RobotController

stage = omni.usd.get_context().get_stage()
rng = np.random.default_rng()

# Initialize managers
leaf_mgr = LeafManager(stage)
physics = PhysicsSetup(stage)
lighting = LightingManager(stage)
robot = RobotController(stage)

# Setup scene
for palm_path in palm_root_paths:
    physics.disable_palm_physics(palm_path)

lighting.randomize_lighting("/path/to/hdri")
leaf_mgr.cull_episode_leaves(palm_root_paths, rng, 0.5)
robot.randomize_robot_root_pose(env, palm_root_paths, rng)
```

## Testing

### Run all components
```bash
python generate_data_refactored.py \
    --task Isaac-PING-TI-VLA-v0 \
    --num_envs 10 \
    --enable_cameras \
    --save_data
```

### Test individual modules
```python
import pytest
from leaf_manager import LeafManager

def test_leaf_manager():
    manager = LeafManager(mock_stage)
    manager.set_leaf_prims_active(palm_path, active=False)
    assert all_hidden(mock_stage, palm_path)
```

## Configuration

All configuration is in `config.py`:

```python
# Kinematic constants
ACTION_CLAMP = 0.25
POSITION_GAIN = 0.35
SPRAY_DURATION = 60

# Pose configuration
REST_POSE_VALUES = [...]
SPRAY_POSE_VALUES = [...]

# Randomization ranges
GIRTH_SCALE_RANGE = (0.85, 1.15)
HEIGHT_SCALE_RANGE = (0.85, 1.25)

# Lighting
HDRI_INTENSITY_RANGE = (600.0, 1200.0)
AMBIENT_FILL_INTENSITY = 500.0

# ... and more
```

To customize, edit `config.py` - all modules import from there.

## Troubleshooting

### ImportError: No module named 'leaf_manager'
Ensure you're in the correct directory:
```bash
cd src/isaac_so_arm101/scripts/vla/data_generation
python generate_data_refactored.py ...
```

### Module not found: 'config'
All modules should be in the same directory. Check you have:
- config.py
- leaf_manager.py
- (all other modules)

### Debug output not showing
Enable debug mode:
```python
manager = LeafManager(stage, debug_verbose=True)
```

## Performance Tips

1. **Profile individual managers**: Each can be profiled independently
2. **Lazy initialization**: Initialize only needed managers
3. **Batch operations**: Managers process batches efficiently
4. **No overhead**: Refactoring maintains original performance

## Next Steps

1. ✅ Read REFACTORING_SUMMARY.md (overview)
2. ✅ Read REFACTORING_GUIDE.md (details)
3. ✅ Review generate_data_refactored.py (main entry)
4. ✅ Try running the refactored code
5. ⬜ Adapt to your specific needs
6. ⬜ Create custom managers for extensions

## Contributing

To extend the refactored architecture:

1. **New manager type**: Create new file following the pattern
   ```python
   class MyManager:
       def __init__(self, stage, debug_verbose=False):
           self.stage = stage
           self.debug_verbose = debug_verbose
       
       def my_method(self):
           """Docstring"""
   ```

2. **Import and use** in main orchestrator
   ```python
   from my_manager import MyManager
   
   manager = MyManager(stage)
   manager.my_method()
   ```

## License

Same as original: BSD-3-Clause

---

**Happy refactoring! The modular architecture makes your code more maintainable and extensible. 🚀**
