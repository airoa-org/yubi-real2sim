# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""KB4 projection, inversion and ray tables shared by wrist consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class FisheyeIntrinsics:
    """KB4 model of one image, with the parameters of ``cv2.fisheye`` and pixel centres at integers.

    A point at the angle ``theta = atan2(|xy|, z)`` from the optical axis (OpenCV frame: +x right,
    +y down, +z forward) lands ``theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)``
    from ``(cx, cy)``, in units of ``(fx, fy)``. The wrist images reach 100 degrees, but
    ``cv2.fisheye`` takes ``atan(|xy| / z)``, so it agrees only for ``z > 0``; use ``project_points``.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float

    def resized(self, width: int, height: int) -> "FisheyeIntrinsics":
        """This model after a bilinear or area ``cv2.resize`` to ``width`` x ``height``."""

        sx = width / self.width
        sy = height / self.height
        return replace(
            self,
            width=int(width),
            height=int(height),
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
        )

    def as_dict(self) -> dict:
        return {"model": "kb4", **asdict(self)}

    def distort(self, theta):
        """Distorted angle in focal-length units."""

        theta2 = theta * theta  # Expanded terms keep the rounding that recorded images were rendered with.
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        return theta * (1.0 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8)

    def distort_slope(self, theta):
        """Derivative of the distorted angle with respect to the ray angle."""

        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        return 1.0 + 3.0 * self.k1 * theta2 + 5.0 * self.k2 * theta4 + 7.0 * self.k3 * theta6 + 9.0 * self.k4 * theta8

    def min_distort_slope(self, theta_max: float) -> float:
        """Minimum of ``distort_slope`` on ``[0, theta_max]``, less a bound on its rounding error.

        The minimum lies at an end or where the slope's derivative in ``u = theta^2``, a cubic, is zero. The bound is
        16 eps times the sum of the terms' magnitudes, because ``distort_slope`` rounds by several eps of that sum
        and ``np.roots`` is inexact.
        """

        u_max = float(theta_max) ** 2
        cubic = [36.0 * self.k4, 21.0 * self.k3, 10.0 * self.k2, 3.0 * self.k1]
        critical = np.roots(cubic).real  # rounding can turn two close real roots complex
        u = np.clip(np.r_[0.0, u_max, critical], 0.0, u_max)
        terms = np.polyval(np.abs([9.0 * self.k4, 7.0 * self.k3, 5.0 * self.k2, 3.0 * self.k1, 1.0]), u)
        return float((self.distort_slope(np.sqrt(u)) - 16.0 * np.finfo(float).eps * terms).min())

    def undistort(self, theta_d, theta_limit):
        """Invert radii, clamping rays outside the field to ``theta_limit`` [rad]."""

        theta_limit = float(theta_limit)
        if self.min_distort_slope(theta_limit) <= 0.0:
            raise ValueError(f"the fisheye model is not monotonic up to {np.rad2deg(theta_limit):g} degrees")
        theta_d = np.asarray(theta_d, dtype=np.float64)
        target = np.minimum(theta_d, self.distort(theta_limit))
        lower, upper = np.zeros_like(target), np.full_like(target, theta_limit)
        theta = np.clip(target, 0.0, theta_limit)
        for _ in range(100):  # A Newton step outside the bracket bisects it instead of cycling.
            residual = self.distort(theta) - target
            lower = np.where(residual <= 0.0, theta, lower)
            upper = np.where(residual >= 0.0, theta, upper)
            step = theta - residual / self.distort_slope(theta)
            step = np.where((lower < step) & (step < upper), step, 0.5 * (lower + upper))
            if np.array_equal(step, theta):
                break
            theta = step
        if np.abs(self.distort(theta) - target).max(initial=0.0) > 1.0e-9 * max(1.0, self.distort(theta_limit)):
            raise ValueError("the fisheye model did not invert to 1e-9")
        return theta

    def project_points(self, points):
        """OpenCV-frame points to image pixels, including rays beyond 90 degrees."""

        x, y, z = np.moveaxis(np.asarray(points, dtype=np.float64), -1, 0)
        rho = np.hypot(x, y)
        theta_d = self.distort(np.arctan2(rho, z))
        scale = np.divide(theta_d, rho, out=np.zeros_like(rho), where=rho > 0.0)
        return np.stack([self.cx + self.fx * scale * x, self.cy + self.fy * scale * y], axis=-1)

    def unproject_points(self, pixels, theta_limit):
        """OpenCV unit rays and validity for pixels; invalid rays stop at ``theta_limit``."""

        x, y = np.moveaxis(np.asarray(pixels, dtype=np.float64), -1, 0)
        dx = (x - self.cx) / self.fx
        dy = (y - self.cy) / self.fy
        theta_d = np.sqrt(dx * dx + dy * dy)
        valid = theta_d <= self.distort(theta_limit)
        theta = self.undistort(theta_d, theta_limit)
        scale = np.divide(np.sin(theta), theta_d, out=np.ones_like(theta_d), where=theta_d > 1.0e-12)
        rays = np.stack([dx * scale, dy * scale, np.cos(theta)], axis=-1)
        return rays, valid

    def camera_rays(self, theta_limit):
        """Float32 raster rays at the pixel centres."""

        y, x = np.mgrid[0:self.height, 0:self.width]
        rays, valid = self.unproject_points(np.stack([x, y], axis=-1), theta_limit)
        return rays.astype(np.float32), valid


def project_points(camera: FisheyeIntrinsics, points) -> np.ndarray:
    """Pixels ``(..., 2)`` of OpenCV-frame points ``(..., 3)``, beyond 90 degrees too."""

    return camera.project_points(points)


def camera_rays(camera: FisheyeIntrinsics, theta_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit ray at every pixel centre and validity within ``theta_max`` [rad]."""

    return camera.camera_rays(theta_max)


def _spherical(points):
    """``theta``, ``cos phi``, ``sin phi`` and ``|p|`` of OpenCV-frame points; ``phi = 0`` on the axis."""

    x, y, z = np.moveaxis(np.asarray(points, dtype=np.float64), -1, 0)
    rho = np.hypot(x, y)
    safe = np.where(rho > 0.0, rho, 1.0)
    return np.arctan2(rho, z), np.where(rho > 0.0, x / safe, 1.0), y / safe, np.hypot(rho, z)


def ray_tangent_basis(points) -> np.ndarray:
    """``(..., 3, 2)``: ``e_theta`` (away from the axis) and ``e_phi`` (around it) at each point.

    With the unit ray as third column they form the right-handed per-point frame of
    ``local_projection_jacobian``; on the axis (``phi = 0``) it is the camera's x and y.
    """

    theta, cos_phi, sin_phi, _ = _spherical(points)
    e_theta = np.stack([np.cos(theta) * cos_phi, np.cos(theta) * sin_phi, -np.sin(theta)], axis=-1)
    e_phi = np.stack([-sin_phi, cos_phi, np.zeros_like(theta)], axis=-1)
    return np.stack([e_theta, e_phi], axis=-1)


def local_projection_jacobian(camera: FisheyeIntrinsics, points) -> np.ndarray:
    """``d pixel / d (x / z, y / z)`` in the frame of ``ray_tangent_basis``, shape ``(..., 2, 2)``.

    ``diag(fx, fy) R(phi) diag(theta_d', theta_d / sin(theta))``, and ``diag(fx, fy)`` on the axis. A pinhole
    EWA with unit focal length in that frame, whose z axis is the ray, chains with it.
    """

    theta, cos_phi, sin_phi, _ = _spherical(points)
    sin_theta = np.sin(theta)
    tangential = np.divide(camera.distort(theta), sin_theta, out=np.ones_like(theta), where=sin_theta > 1.0e-12)
    rotation = np.stack([np.stack([cos_phi, -sin_phi], axis=-1), np.stack([sin_phi, cos_phi], axis=-1)], axis=-2)
    stretch = np.stack([camera.distort_slope(theta), tangential], axis=-1)[..., None, :]
    return np.array([camera.fx, camera.fy])[:, None] * rotation * stretch


def projection_jacobian(camera: FisheyeIntrinsics, points) -> np.ndarray:
    """``d project_points / d point``, shape ``(..., 2, 3)``.

    It is ``local_projection_jacobian @ ray_tangent_basis^T / |p|``. Its first two columns are not an
    image Jacobian, because at 90 degrees they are singular.
    """

    points = np.asarray(points, dtype=np.float64)
    basis_t = np.swapaxes(ray_tangent_basis(points), -1, -2)
    return local_projection_jacobian(camera, points) @ basis_t / np.linalg.norm(points, axis=-1)[..., None, None]
