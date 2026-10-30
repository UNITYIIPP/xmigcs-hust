"""
Joint mapping utilities for xmigcs.

Provides MuJoCo XML joint name lists (loaded from configuration) and a
:class:`JointMapping` that computes the index mapping between *lab order*
(the joint sequence stored in ONNX metadata) and *mj order* (the full
MuJoCo XML joint order).

Typical usage::

    from xmigcs.common.joint_mapping import JointMapping, MJ_JOINTS_31

    mapping = JointMapping(joint_seq=onnx_joint_names, joint_xml=MJ_JOINTS_31)

    # Scatter a lab-order action into a full mj-order array
    mj_array = mapping.scatter_to_mj(lab_action)

    # Gather only the controlled joints back from mj-order to lab-order
    lab_obs = mapping.gather_from_mj(mj_state)
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

# Default config path (relative to this file)
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "config", "dex_config.yaml"
)


def load_mj_joints(
    config_path: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Load MuJoCo joint name lists from YAML configuration.

    Reads the ``mj_joints`` section from the config file, which should
    contain ``base`` (29-DOF body joints) and ``head`` (2 head joints).
    The 31-DOF list is derived as ``base + head``.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.  Defaults to
        ``src/xmigcs/config/dex_config.yaml``.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(MJ_JOINTS_31, MJ_JOINTS_29)``
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    mj_joints_cfg = cfg.get("mj_joints", {})
    base_joints = list(mj_joints_cfg.get("base", []))
    head_joints = list(mj_joints_cfg.get("head", []))

    if not base_joints:
        raise ValueError(
            f"Config file {config_path} must contain 'mj_joints.base' "
            f"with the body joint name list."
        )

    mj_joints_29 = base_joints
    mj_joints_31 = base_joints + head_joints

    return mj_joints_31, mj_joints_29


# ---------------------------------------------------------------------------
# Module-level constants (loaded from config on import)
# ---------------------------------------------------------------------------

_MJ_JOINTS_31, _MJ_JOINTS_29 = load_mj_joints()

MJ_JOINTS_31: List[str] = _MJ_JOINTS_31
"""31-DOF MuJoCo XML joint order (including head_pitch and head_yaw)."""

MJ_JOINTS_29: List[str] = _MJ_JOINTS_29
"""29-DOF MuJoCo XML joint order (no head joints)."""


# ---------------------------------------------------------------------------
# JointMapping
# ---------------------------------------------------------------------------


class JointMapping:
    """Bidirectional index mapping between *lab order* and *mj order*.

    **Lab order** is the joint name sequence stored as ONNX metadata
    (``joint_names``), typically a subset of the full MuJoCo XML joint list.

    **Mj order** is the complete joint list defined by the MuJoCo XML model
    (e.g. :data:`MJ_JOINTS_31` or :data:`MJ_JOINTS_29`).

    Parameters
    ----------
    joint_seq:
        Joint names in lab order (from ONNX metadata).
    joint_xml:
        Joint names in mj / MuJoCo XML order.  Defaults to
        :data:`MJ_JOINTS_31`.

    Raises
    ------
    ValueError
        If any joint in *joint_seq* is not present in *joint_xml*.
    """

    def __init__(
        self,
        joint_seq: Sequence[str],
        joint_xml: Sequence[str] | None = None,
    ) -> None:
        if joint_xml is None:
            joint_xml = MJ_JOINTS_31

        self._joint_xml: List[str] = list(joint_xml)
        self._joint_seq: List[str] = list(joint_seq)

        # Validate that every lab joint exists in the xml list
        xml_set = set(self._joint_xml)
        missing = [n for n in self._joint_seq if n not in xml_set]
        if missing:
            raise ValueError(
                f"The following joints from ONNX metadata are not present in "
                f"joint_xml: {missing}"
            )

        # lab2mj: for each lab-order index, the corresponding mj-order index
        self._lab2mj: np.ndarray = np.array(
            [self._joint_xml.index(name) for name in self._joint_seq],
            dtype=int,
        )

        # mj2lab: inverse mapping (mj-order index → lab-order index).
        # Joints that are NOT in the lab set receive -1.
        self._mj2lab: np.ndarray = np.full(len(self._joint_xml), -1, dtype=int)
        for lab_idx, mj_idx in enumerate(self._lab2mj):
            self._mj2lab[mj_idx] = lab_idx

    # -- Properties ----------------------------------------------------------

    @property
    def lab2mj(self) -> np.ndarray:
        """Mapping array from lab-order indices to mj-order indices.

        Shape: ``(num_lab_joints,)``, dtype ``int``.
        ``lab2mj[i]`` is the mj-order index of the *i*-th lab joint.
        """
        return self._lab2mj

    @property
    def mj2lab(self) -> np.ndarray:
        """Mapping array from mj-order indices to lab-order indices.

        Shape: ``(num_mj_joints,)``, dtype ``int``.
        ``mj2lab[j]`` is the lab-order index of the *j*-th mj joint, or ``-1``
        if that joint is not part of the lab set.
        """
        return self._mj2lab

    @property
    def num_mj_joints(self) -> int:
        """Total number of joints in the MuJoCo XML order."""
        return len(self._joint_xml)

    @property
    def num_lab_joints(self) -> int:
        """Number of controlled (lab) joints."""
        return len(self._joint_seq)

    @property
    def joint_xml(self) -> List[str]:
        """Joint names in MuJoCo XML order."""
        return self._joint_xml

    # -- Scatter / Gather ----------------------------------------------------

    def scatter_to_mj(self, lab_array: np.ndarray) -> np.ndarray:
        """Scatter a lab-order array into a full mj-order array.

        Positions not covered by the lab joints are filled with zeros.

        Parameters
        ----------
        lab_array:
            1-D array in lab order with shape ``(num_lab_joints,)``.

        Returns
        -------
        np.ndarray
            1-D array in mj order with shape ``(num_mj_joints,)``.
        """
        mj_array = np.zeros(self.num_mj_joints, dtype=lab_array.dtype)
        mj_array[self._lab2mj] = lab_array
        return mj_array

    def gather_from_mj(self, mj_array: np.ndarray) -> np.ndarray:
        """Gather only the controlled joints from mj-order to lab-order.

        Parameters
        ----------
        mj_array:
            1-D array in mj order with shape ``(num_mj_joints,)``.

        Returns
        -------
        np.ndarray
            1-D array in lab order with shape ``(num_lab_joints,)``.
        """
        return mj_array[self._lab2mj]
