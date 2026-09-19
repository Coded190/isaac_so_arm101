"""USD authoring → PhysX tensor root sync for Kit teleop.

Isaac Sim 6.1 / Lab 3.0:

* USD Property panel writes ``xformOp:translate``, ``xformOp:orient`` (Gf WXYZ),
  ``xformOp:scale``.
* Lab tensors / PhysX / Warp use ``(x, y, z, qx, qy, qz, qw)`` (XYZW).
  See Lab 3 migration: WXYZ → XYZW.
* Fabric ``omni:fabric:worldMatrix`` is the viewport cache. PhysX overwrites it
  each step. Do not read it as the panel pose.
* ``write_root_pose_to_sim_index`` is a rigid transform (no scale). Scale is
  reapplied to Fabric after each step so the panel Scale field stays visible.

Write tensors from those attrs, never from ``ComputeLocalToWorldTransform`` /
``GetLocalTransformation`` (those follow Fabric during Play and explode PhysX
on Orient drags). Re-apply the authored pose every step (``reason=hold``) so a
free root stays at the panel pose instead of falling.
"""

from __future__ import annotations

import math
from typing import Any

Pose7 = tuple[float, float, float, float, float, float, float]
Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]
QuatXYZW = tuple[float, float, float, float]

USD_ROOT_POS_EPS = 1.0e-3
USD_ROOT_QUAT_EPS = 1.0e-3
USD_ATTR_POS_EPS = 1.0e-4
USD_ATTR_QUAT_EPS = 1.0e-4
USD_ATTR_SCALE_EPS = 1.0e-4
MAX_ROOT_ABS = 50.0
MIN_SCALE = 1.0e-3
MAX_SCALE = 100.0
IDENTITY_SCALE: Vec3 = (1.0, 1.0, 1.0)


def pose7_pos_delta(a: Pose7, b: Pose7) -> float:
    """Euclidean distance between the xyz of two ``(x,y,z,qx,qy,qz,qw)`` poses."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def pose7_quat_delta(a: Pose7, b: Pose7) -> float:
    """``1 - |dot|`` between xyzw quaternions (0 = same orientation)."""
    return abs(1.0 - abs(a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6]))


def vec3_delta(a: Vec3 | None, b: Vec3 | None) -> float:
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def quat_wxyz_delta(a: QuatWXYZ | None, b: QuatWXYZ | None) -> float:
    if a is None or b is None:
        return float("inf")
    return abs(1.0 - abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]))


def wxyz_to_xyzw(q: QuatWXYZ) -> QuatXYZW:
    """USD / Gf.Quatd ``(w,x,y,z)`` → Lab 3 / PhysX ``(x,y,z,w)``."""
    return (q[1], q[2], q[3], q[0])


def xyzw_to_wxyz(q: QuatXYZW) -> QuatWXYZ:
    return (q[3], q[0], q[1], q[2])


def _xyzw_to_wxyz(pose: Pose7) -> QuatWXYZ:
    return (pose[6], pose[3], pose[4], pose[5])


def normalize_wxyz(q: QuatWXYZ) -> QuatWXYZ:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1.0e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_mul_xyzw(a: QuatXYZW, b: QuatXYZW) -> QuatXYZW:
    """Hamilton product, Lab 3 ``(x,y,z,w)``. ``a`` is applied after ``b`` on a vector."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate_xyzw(q: QuatXYZW, v: Vec3) -> Vec3:
    qx, qy, qz, qw = q
    uv = (qy * v[2] - qz * v[1], qz * v[0] - qx * v[2], qx * v[1] - qy * v[0])
    uuv = (qy * uv[2] - qz * uv[1], qz * uv[0] - qx * uv[2], qx * uv[1] - qy * uv[0])
    return (
        v[0] + 2.0 * qw * uv[0] + 2.0 * uuv[0],
        v[1] + 2.0 * qw * uv[1] + 2.0 * uuv[1],
        v[2] + 2.0 * qw * uv[2] + 2.0 * uuv[2],
    )


def pose7_from_translate_orient_wxyz(
    translate: Vec3,
    orient_wxyz: QuatWXYZ,
    *,
    parent_t: Vec3 = (0.0, 0.0, 0.0),
    parent_q_wxyz: QuatWXYZ = (1.0, 0.0, 0.0, 0.0),
) -> Pose7:
    """World ``(x,y,z,qx,qy,qz,qw)`` from USD translate + Gf orient, optional parent."""
    local_q = wxyz_to_xyzw(normalize_wxyz(orient_wxyz))
    parent_q = wxyz_to_xyzw(normalize_wxyz(parent_q_wxyz))
    world_q = quat_mul_xyzw(parent_q, local_q)
    n = math.sqrt(world_q[0] ** 2 + world_q[1] ** 2 + world_q[2] ** 2 + world_q[3] ** 2)
    world_q = (world_q[0] / n, world_q[1] / n, world_q[2] / n, world_q[3] / n)
    rotated = quat_rotate_xyzw(parent_q, translate)
    world_t = (parent_t[0] + rotated[0], parent_t[1] + rotated[1], parent_t[2] + rotated[2])
    return (world_t[0], world_t[1], world_t[2], world_q[0], world_q[1], world_q[2], world_q[3])


def pose7_is_sane(pose: Pose7, *, max_abs: float = MAX_ROOT_ABS) -> bool:
    if not all(math.isfinite(v) for v in pose):
        return False
    if any(abs(pose[i]) > max_abs for i in range(3)):
        return False
    qn = math.sqrt(pose[3] ** 2 + pose[4] ** 2 + pose[5] ** 2 + pose[6] ** 2)
    return 0.5 < qn < 1.5


def scale_is_sane(scale: Vec3 | None) -> bool:
    if scale is None:
        return True
    if not all(math.isfinite(v) for v in scale):
        return False
    return all(MIN_SCALE <= v <= MAX_SCALE for v in scale)


def next_root_write_reason(
    *,
    has_authored: bool,
    latched: bool,
    usd_changed: bool,
    authored_sane: bool,
    hold: bool = True,
) -> str:
    """Why KitRootSync should (or should not) write tensors this step."""
    if not has_authored:
        return "no_prim"
    if not authored_sane:
        return "rejected_pose"
    if not latched:
        return "latch"
    if usd_changed:
        return "usd_attr"
    if hold:
        return "hold"
    return "idle"


def scale_is_identity(scale: Vec3 | None, *, eps: float = USD_ATTR_SCALE_EPS) -> bool:
    if scale is None:
        return True
    return vec3_delta(scale, IDENTITY_SCALE) < eps


def authored_attrs_changed(
    translate: Vec3 | None,
    orient_wxyz: QuatWXYZ | None,
    prev_translate: Vec3 | None,
    prev_orient_wxyz: QuatWXYZ | None,
    *,
    scale: Vec3 | None = None,
    prev_scale: Vec3 | None = None,
    pos_eps: float = USD_ATTR_POS_EPS,
    quat_eps: float = USD_ATTR_QUAT_EPS,
    scale_eps: float = USD_ATTR_SCALE_EPS,
) -> bool:
    """True when Property-panel USD attrs moved since the last latch."""
    if prev_translate is None and prev_orient_wxyz is None and prev_scale is None:
        return False
    if vec3_delta(translate, prev_translate) >= pos_eps:
        return True
    if quat_wxyz_delta(orient_wxyz, prev_orient_wxyz) >= quat_eps:
        return True
    if scale is not None or prev_scale is not None:
        left = scale if scale is not None else IDENTITY_SCALE
        right = prev_scale if prev_scale is not None else IDENTITY_SCALE
        if vec3_delta(left, right) >= scale_eps:
            return True
    return False


def should_write_root(
    authored: Pose7,
    physx: Pose7,
    *,
    pos_eps: float = USD_ROOT_POS_EPS,
    quat_eps: float = USD_ROOT_QUAT_EPS,
) -> bool:
    """True when the USD-authored world pose differs from the PhysX root.

    Diagnostic only. Live teleop writes on :func:`authored_attrs_changed`.
    """
    return pose7_pos_delta(authored, physx) >= pos_eps or pose7_quat_delta(authored, physx) >= quat_eps


def _fmt_xyz(xyz: tuple[float, ...] | None, digits: int = 5) -> str:
    if xyz is None:
        return "n/a"
    return "(" + ", ".join(f"{xyz[i]:.{digits}f}" for i in range(3)) + ")"


def _fmt_quat_wxyz(quat: QuatWXYZ | None) -> str:
    if quat is None:
        return "n/a"
    return f"(w={quat[0]:.5f}, x={quat[1]:.5f}, y={quat[2]:.5f}, z={quat[3]:.5f})"


def _matrix_to_xyzw(mat) -> Pose7:
    mat.Orthonormalize()
    trans = mat.ExtractTranslation()
    quat = mat.ExtractRotationQuat()
    imag = quat.GetImaginary()
    return (
        float(trans[0]),
        float(trans[1]),
        float(trans[2]),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
        float(quat.GetReal()),
    )


def usd_translate_attr(prim) -> Vec3 | None:
    """Authored ``xformOp:translate`` from USD (Property panel Translate)."""
    if prim is None or not prim.IsValid():
        return None
    attr = prim.GetAttribute("xformOp:translate")
    if not attr:
        return None
    val = attr.Get()
    if val is None:
        return None
    return (float(val[0]), float(val[1]), float(val[2]))


def usd_orient_attr_wxyz(prim) -> QuatWXYZ | None:
    """Authored ``xformOp:orient`` as ``(w, x, y, z)`` (Property panel Orient)."""
    if prim is None or not prim.IsValid():
        return None
    attr = prim.GetAttribute("xformOp:orient")
    if not attr:
        return None
    val = attr.Get()
    if val is None:
        return None
    imag = val.GetImaginary()
    return (float(val.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))


def usd_scale_attr(prim) -> Vec3 | None:
    """Authored ``xformOp:scale`` from USD (Property panel Scale)."""
    if prim is None or not prim.IsValid():
        return None
    attr = prim.GetAttribute("xformOp:scale")
    if not attr:
        return None
    val = attr.Get()
    if val is None:
        return None
    return (float(val[0]), float(val[1]), float(val[2]))


def usd_xform_op_names(prim) -> list[str]:
    if prim is None or not prim.IsValid():
        return []
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    return [op.GetOpName() for op in xformable.GetOrderedXformOps()]


def fabric_world_pose_xyzw(prim) -> Pose7 | None:
    """Composed world matrix. With Fabric this is often the runtime pose."""
    if prim is None or not prim.IsValid():
        return None
    from pxr import Usd, UsdGeom

    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return _matrix_to_xyzw(mat)


def usd_authored_world_pose_xyzw(prim) -> Pose7 | None:
    """World rigid pose from USD translate+orient attrs (Lab 3 xyzw)."""
    translate = usd_translate_attr(prim)
    orient = usd_orient_attr_wxyz(prim)
    if translate is None or orient is None:
        return None
    parent = prim.GetParent() if prim is not None else None
    parent_t = usd_translate_attr(parent) if parent is not None else None
    parent_q = usd_orient_attr_wxyz(parent) if parent is not None else None
    return pose7_from_translate_orient_wxyz(
        translate,
        orient,
        parent_t=parent_t or (0.0, 0.0, 0.0),
        parent_q_wxyz=parent_q or (1.0, 0.0, 0.0, 0.0),
    )


def physx_root_pose_xyzw(robot) -> Pose7:
    current = robot.data.root_pose_w[0].detach()
    return (
        float(current[0]),
        float(current[1]),
        float(current[2]),
        float(current[3]),
        float(current[4]),
        float(current[5]),
        float(current[6]),
    )


def _usdrt_matrix4d_from_pxr(pxr_mat):
    """Convert a pxr ``Gf.Matrix4d`` to usdrt (Fabric) ``Gf.Matrix4d``."""
    from usdrt import Gf as GfRt

    vals = tuple(float(pxr_mat[i][j]) for i in range(4) for j in range(4))
    try:
        return GfRt.Matrix4d(*vals)
    except TypeError:
        mat = GfRt.Matrix4d(1.0)
        for i in range(4):
            for j in range(4):
                mat[i, j] = vals[i * 4 + j]
        return mat


def write_fabric_world_trs(prim, pose: Pose7, scale: Vec3) -> str | None:
    """Write viewport TRS into Fabric. PhysX tensors stay rigid (no scale).

    Official Sim 6.1 / Lab 3 path: USDRT ``Rt.Xformable`` worldMatrix, then
    ``IFabricHierarchy.update_world_xforms``. See Omni PhysX simulation
    control (Fabric ``omni:fabric:worldMatrix``) and Lab ``FabricFrameView``.

    Returns None on success, or an error string.
    """
    try:
        from pxr import Gf
        import omni.usd
        from usdrt import Gf as GfRt
        from usdrt import Rt as UsdRtGeom
        from usdrt import Usd as UsdRt
        from usdrt.hierarchy import IFabricHierarchy
    except Exception as exc:  # noqa: BLE001
        return f"usdrt import: {exc}"

    try:
        stage_id = omni.usd.get_context().get_stage_id()
        rt_stage = UsdRt.Stage.Attach(stage_id)
        rt_prim = rt_stage.GetPrimAtPath(str(prim.GetPath()))
        xformable = UsdRtGeom.Xformable(rt_prim)
        if hasattr(xformable, "HasFabricHierarchyWorldMatrixAttr") and not xformable.HasFabricHierarchyWorldMatrixAttr():
            xformable.CreateFabricHierarchyWorldMatrixAttr()
        # Push authored USD TRS (translate, orient WXYZ, scale) into Fabric.
        if hasattr(xformable, "SetWorldXformFromUsd"):
            xformable.SetWorldXformFromUsd()
        orient_wxyz = _xyzw_to_wxyz(pose)
        xf = Gf.Transform()
        xf.SetScale(Gf.Vec3d(*scale))
        xf.SetRotation(Gf.Rotation(Gf.Quatd(orient_wxyz[0], Gf.Vec3d(*orient_wxyz[1:]))))
        xf.SetTranslation(Gf.Vec3d(pose[0], pose[1], pose[2]))
        attr = xformable.GetFabricHierarchyWorldMatrixAttr()
        attr.Set(_usdrt_matrix4d_from_pxr(xf.GetMatrix()))
        scale_attr = rt_prim.GetAttribute("xformOp:scale")
        if scale_attr:
            scale_attr.Set(GfRt.Vec3d(*scale))
        fabric = IFabricHierarchy().get_fabric_hierarchy(rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId())
        fabric.update_world_xforms()
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None


class KitRootSync:
    """Copy Property-panel USD xformOps into PhysX tensors.

    Panel edits write immediately. Every later step re-applies the same authored
    pose (``hold``) so gravity cannot pull a free-floating root down.
    """

    def __init__(self) -> None:
        self._prev_translate: Vec3 | None = None
        self._prev_orient: QuatWXYZ | None = None
        self._prev_scale: Vec3 | None = None
        self._last_scale: Vec3 = IDENTITY_SCALE
        self._last_written: Pose7 | None = None
        self._fabric_scale_error: str | None = None

    def _commit_root_pose(self, robot, authored: Pose7, *, zero_joint_vel: bool) -> None:
        import torch

        current = robot.data.root_pose_w[0].detach()
        with torch.inference_mode(False):
            pose_t = torch.tensor([list(authored)], device=current.device, dtype=current.dtype)
            env_ids = torch.tensor([0], device=current.device, dtype=torch.long)
            robot.write_root_pose_to_sim_index(root_pose=pose_t, env_ids=env_ids)
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=current.device, dtype=current.dtype),
                env_ids=env_ids,
            )
            if zero_joint_vel:
                n_joints = int(robot.num_joints)
                robot.write_joint_velocity_to_sim_index(
                    velocity=torch.zeros((1, n_joints), device=current.device, dtype=current.dtype),
                    env_ids=env_ids,
                )

    def apply(self, robot, prim) -> dict[str, Any]:
        translate = usd_translate_attr(prim)
        orient = usd_orient_attr_wxyz(prim)
        scale = usd_scale_attr(prim)
        authored = usd_authored_world_pose_xyzw(prim)
        fabric = fabric_world_pose_xyzw(prim)
        physx = physx_root_pose_xyzw(robot)
        usd_changed = authored_attrs_changed(
            translate,
            orient,
            self._prev_translate,
            self._prev_orient,
            scale=scale,
            prev_scale=self._prev_scale,
        )
        physx_mismatch = authored is not None and should_write_root(authored, physx)
        latched = self._prev_translate is not None and self._prev_orient is not None
        authored_sane = (
            authored is not None and pose7_is_sane(authored) and scale_is_sane(scale)
        )
        reason = next_root_write_reason(
            has_authored=authored is not None and translate is not None and orient is not None,
            latched=latched,
            usd_changed=usd_changed,
            authored_sane=authored_sane,
        )
        info: dict[str, Any] = {
            "applied": False,
            "reason": reason,
            "usd_changed": usd_changed,
            "physx_mismatch": physx_mismatch,
            "usd_translate": translate,
            "usd_orient_wxyz": orient,
            "usd_scale": scale,
            "usd_world": authored[:3] if authored is not None else None,
            "usd_world_q_wxyz": _xyzw_to_wxyz(authored) if authored is not None else None,
            "written_q_wxyz": None,
            "fabric_world": fabric[:3] if fabric is not None else None,
            "physx_root": physx[:3],
            "physx_q_wxyz": _xyzw_to_wxyz(physx),
            "pos_delta": pose7_pos_delta(authored, physx) if authored is not None else None,
            "quat_delta": pose7_quat_delta(authored, physx) if authored is not None else None,
        }
        if reason in {"no_prim", "idle"}:
            return info
        if reason == "rejected_pose":
            print(
                f"[teleop] rejected USD pose t={translate} q={orient} scale={scale} "
                f"pose7={authored}",
                flush=True,
            )
            self._prev_translate = translate
            self._prev_orient = orient
            self._prev_scale = scale
            return info

        assert authored is not None
        self._commit_root_pose(robot, authored, zero_joint_vel=(reason == "usd_attr"))
        self._prev_translate = translate
        self._prev_orient = orient
        self._prev_scale = scale
        self._last_scale = scale or IDENTITY_SCALE
        self._last_written = authored
        info["applied"] = True
        info["written_q_wxyz"] = _xyzw_to_wxyz(authored)
        return info

    def sync_visual_scale(self, prim, robot) -> None:
        """Re-apply USD scale onto Fabric after PhysX stomps worldMatrix."""
        scale = usd_scale_attr(prim) or self._last_scale
        if scale_is_identity(scale):
            return
        physx = physx_root_pose_xyzw(robot)
        err = write_fabric_world_trs(prim, physx, scale)
        if err and err != self._fabric_scale_error:
            self._fabric_scale_error = err
            print(f"[teleop] fabric scale write failed: {err}", flush=True)


def apply_kit_root_transform(robot, prim, sync: KitRootSync | None = None) -> dict[str, Any]:
    """Backward-compatible wrapper. Prefer :class:`KitRootSync` in the loop."""
    if sync is None:
        sync = KitRootSync()
    return sync.apply(robot, prim)


def format_root_diag(info: dict[str, Any], *, step: int | None = None) -> str:
    prefix = "[teleop]" if step is None else f"[teleop] step={step}"
    pos_delta = info.get("pos_delta")
    quat_delta = info.get("quat_delta")
    delta_s = f"{pos_delta:.5f}" if isinstance(pos_delta, float) else "n/a"
    qdelta_s = f"{quat_delta:.5f}" if isinstance(quat_delta, float) else "n/a"
    return (
        f"{prefix} reason={info.get('reason')} usd_changed={info.get('usd_changed')} "
        f"applied={info.get('applied')} "
        f"usd_t={_fmt_xyz(info.get('usd_translate'))} "
        f"usd_orient={_fmt_quat_wxyz(info.get('usd_orient_wxyz'))} "
        f"usd_scale={_fmt_xyz(info.get('usd_scale'))} "
        f"written_q={_fmt_quat_wxyz(info.get('written_q_wxyz'))} "
        f"physx_t={_fmt_xyz(info.get('physx_root'))} "
        f"physx_q={_fmt_quat_wxyz(info.get('physx_q_wxyz'))} "
        f"pos_delta={delta_s} quat_delta={qdelta_s} "
        f"physx_mismatch={info.get('physx_mismatch')}"
    )
