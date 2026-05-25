# VLA Data Generation Refactoring Guide

## Overview

The `generate_data.py` file has been refactored from a **2000+ line monolithic script** into a **modular architecture** following **SOLID principles**. This improves maintainability, testability, and extensibility.

## Architecture Changes

### Before (Monolithic)
- **Single file**: `generate_data.py` (~2000 lines)
- **Mixed responsibilities**: Physics, randomization, control, recording, lighting, camera
- **Hard to test**: No clear boundaries between concerns
- **Difficult to extend**: Adding new features requires modifying the main file

### After (Modular)

```
data_generation/
├── config.py                    # All constants (SRP)
├── noise_filter.py              # Output stream filtering
├── math_utils.py                # Quaternion, vector operations
├── geometry_utils.py            # Geometric calculations
├── leaf_manager.py              # Leaf management (SRP)
├── palm_randomizer.py           # Palm dimension randomization (SRP)
├── physics_setup.py             # Physics configuration (SRP)
├── lighting_manager.py          # HDRI lighting (SRP)
├── camera_controller.py         # Viewport camera control (SRP)
├── robot_controller.py          # Robot positioning (SRP)
├── spray_oracle.py              # FSM controller (SRP)
├── dataset_manager.py           # Dataset recording (SRP)
├── generate_data_refactored.py  # Main orchestrator
```

## SOLID Principles Applied

### 1. **Single Responsibility Principle (SRP)**

Each module has **one reason to change**:

- `config.py` → Changes only when constants/configuration need updating
- `leaf_manager.py` → Changes only for leaf management logic
- `palm_randomizer.py` → Changes only for randomization logic
- `lighting_manager.py` → Changes only for lighting setup
- `robot_controller.py` → Changes only for robot control
- `spray_oracle.py` → Changes only for FSM logic
- `dataset_manager.py` → Changes only for dataset recording

**Before:**
```python
# Everything mixed in one file
def randomize_lighting(...): ...
def disable_palm_physics(...): ...
def randomize_robot_root_pose(...): ...
def cull_episode_leaves(...): ...
# ... 60+ more functions
```

**After:**
```python
# Each concern isolated
lighting_manager = LightingManager(stage)
physics_setup = PhysicsSetup(stage)
robot_controller = RobotController(stage)
leaf_manager = LeafManager(stage)
```

### 2. **Open/Closed Principle (OCP)**

Modules are **open for extension, closed for modification**:

```python
# Easy to add new randomization strategies
class PalmRandomizer:
    def randomize_palm_dimensions(self, palm_root_path):
        # Core logic here
    
    def _randomize_canopy(self, crown_prim):
        # Can extend this method in subclasses
```

**Adding new features:**
- Need different randomization? Create `PalmRandomizerV2` inheriting from `PalmRandomizer`
- Need new FSM states? Extend `SprayOracle` class
- No need to modify existing code

### 3. **Liskov Substitution Principle (LSP)**

Manager classes share consistent interfaces:

```python
# All managers follow similar patterns
class PhysicsSetup:
    def __init__(self, stage, debug_verbose=False): ...
    def disable_palm_physics(self, palm_root_path): ...

class LightingManager:
    def __init__(self, stage, debug_verbose=False): ...
    def randomize_lighting(self, hdri_folder_path, env_ids=None): ...

class RobotController:
    def __init__(self, stage, debug_verbose=False): ...
    def randomize_robot_root_pose(self, ...): ...

# Can be used polymorphically if needed
managers = [PhysicsSetup(stage), LightingManager(stage), RobotController(stage)]
```

### 4. **Interface Segregation Principle (ISP)**

Each module exposes only necessary interfaces:

```python
# LeafManager doesn't expose physics operations
leaf_manager = LeafManager(stage)
leaf_manager.set_leaf_prims_active(palm_path)  # Only leaf-specific methods
leaf_manager.cull_episode_leaves(...)

# PhysicsSetup doesn't expose lighting operations
physics_setup = PhysicsSetup(stage)
physics_setup.disable_palm_physics(palm_path)  # Only physics-specific methods
```

### 5. **Dependency Inversion Principle (DIP)**

High-level modules depend on abstractions, not implementations:

```python
# Main logic depends on manager interfaces, not internal implementations
def setup_environment():
    lighting_manager = LightingManager(stage)  # Abstraction
    physics_setup = PhysicsSetup(stage)        # Abstraction
    robot_controller = RobotController(stage)  # Abstraction
    
    # Uses public interfaces, doesn't depend on internal details
    lighting_manager.randomize_lighting(hdri_path)
    physics_setup.disable_palm_physics(palm_path)
    robot_controller.randomize_robot_root_pose(env, paths, rng)
```

## Module Descriptions

### Core Modules

| Module | Responsibility | Key Classes |
|--------|---|---|
| `config.py` | Configuration constants | Constants only |
| `math_utils.py` | Quaternion/vector math | Pure functions |
| `geometry_utils.py` | Spatial calculations | Pure functions |

### Feature Modules

| Module | Responsibility | Key Class |
|--------|---|---|
| `leaf_manager.py` | Leaf culling & visibility | `LeafManager` |
| `palm_randomizer.py` | Tree dimension variation | `PalmRandomizer` |
| `physics_setup.py` | Physics configuration | `PhysicsSetup` |
| `lighting_manager.py` | HDRI & dome lights | `LightingManager` |
| `camera_controller.py` | Viewport positioning | `CameraController` |
| `robot_controller.py` | Robot base & arm control | `RobotController` |
| `spray_oracle.py` | FSM state machine | `SprayOracle` |
| `dataset_manager.py` | Dataset recording | `DatasetManager` |

### Main Entry Point

| Module | Responsibility |
|--------|---|
| `generate_data_refactored.py` | Orchestrates modules & main loop |

## Usage Examples

### Before (Monolithic)
```python
# Everything in one massive file
# Hard to understand, test, or modify
```

### After (Modular)
```python
# Setup individual managers
leaf_manager = LeafManager(stage, debug_verbose=True)
physics_setup = PhysicsSetup(stage)
robot_controller = RobotController(stage)

# Use clean, focused interfaces
leaf_manager.cull_episode_leaves(palm_roots, rng, cull_prob=0.5)
physics_setup.disable_palm_physics(palm_path)
robot_controller.randomize_robot_root_pose(env, palm_roots, rng)

# Easy to test each component in isolation
def test_leaf_manager():
    manager = LeafManager(mock_stage)
    manager.set_leaf_prims_active(palm_path, active=False)
    assert all_leaves_hidden(mock_stage, palm_path)
```

## Benefits

### Maintainability
- ✅ Find issues quickly (each module has one concern)
- ✅ Debug in isolation (mock dependencies)
- ✅ Update constants in one place

### Testability
- ✅ Unit test individual managers
- ✅ Mock dependencies easily
- ✅ No need to run full simulation to test one feature

### Extensibility
- ✅ Add new randomization strategies without touching existing code
- ✅ Replace implementations with new versions
- ✅ Share components across other projects

### Performance
- ✅ Lazy import only needed modules
- ✅ Profile individual components
- ✅ Optimize without affecting other parts

## Migration Guide

### Step 1: Use the new modular code
Replace old `generate_data.py` with `generate_data_refactored.py`

### Step 2: Import managers as needed
```python
from leaf_manager import LeafManager
from palm_randomizer import PalmRandomizer

# Create and use
leaf_manager = LeafManager(stage)
leaf_manager.cull_episode_leaves(...)
```

### Step 3: Add custom managers for new features
```python
class CustomRandomizer:
    def __init__(self, stage):
        self.stage = stage
    
    def my_custom_function(self):
        # Your code here
```

## Testing Examples

### Test LeafManager
```python
def test_leaf_visibility():
    manager = LeafManager(mock_stage)
    manager.set_leaf_prims_active(palm_path, active=False)
    assert all(not is_visible(prim) for prim in get_leaves(mock_stage, palm_path))
```

### Test RobotController
```python
def test_robot_positioning():
    controller = RobotController(mock_stage)
    controller.randomize_robot_root_pose(mock_env, palm_paths, rng)
    assert robot_is_in_valid_position(mock_env)
```

### Test SprayOracle
```python
def test_fsm_transitions():
    oracle = SprayOracle()
    action = oracle.compute_action(ee_pos, ee_quat, hover_target)
    assert oracle.state == 0  # Should be in approach state
```

## Configuration Management

All configuration is now centralized in `config.py`:

```python
# Before: scattered throughout file
ACTION_CLAMP = 0.25
POSITION_GAIN = 0.35
SPRAY_DURATION = 60
# ... 50+ lines scattered around

# After: single source of truth
from config import ACTION_CLAMP, POSITION_GAIN, SPRAY_DURATION
```

## Debugging

### Enable debug output
```python
manager = LeafManager(stage, debug_verbose=True)
manager.cull_episode_leaves(...)  # Prints debug info
```

### Trace module initialization
```python
# Each manager logs its setup
leaf_manager = LeafManager(stage, debug_verbose=True)
physics_setup = PhysicsSetup(stage, debug_verbose=True)
# ... startup messages from each manager
```

## Performance Considerations

- **No performance regression**: Refactoring maintains same algorithmic complexity
- **Module instantiation**: Minimal overhead (one-time at startup)
- **Function call overhead**: Negligible compared to simulation cost
- **Memory**: Slightly more due to manager instances, but minimal (~MB)

## Future Improvements

1. **Plugin architecture**: Load managers dynamically
2. **Configuration profiles**: Save/load manager presets
3. **Dataset versioning**: Track which managers generated which datasets
4. **Manager composition**: Combine managers for complex workflows
5. **Async operations**: Background tasks in managers

## Conclusion

This refactoring improves code quality while maintaining performance and functionality. The modular design makes the codebase more maintainable, testable, and extensible for future development.
