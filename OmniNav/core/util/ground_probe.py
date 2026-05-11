from typing import Optional


class GroundProbe:
    """Utility that uses NavMesh queries to project a point to walkable ground."""

    def __init__(
        self,
        *,
        robot_path: str,
        prefix: str = "",
        navigation_area: Optional[int] = None,
    ):
        import carb
        import omni.anim.navigation.core as nav

        self._prefix = prefix
        self._robot_path = robot_path.rstrip("/") or "/"
        self._navigation_area = navigation_area
        self._carb = carb
        self._nav_interface = nav.acquire_interface()
        if self._nav_interface is None:
            raise RuntimeError(f"{self._prefix}NavMesh interface unavailable; enable omni.anim.navigation.core.")
        self._navmesh = None

    def project(self, x: float, y: float, world_z: float):
        """Return (ground_z, True) by snapping to the closest NavMesh point."""
        navmesh = self._acquire_navmesh()
        origin = self._carb.Float3(float(x), float(y), float(world_z))
        try:
            if self._navigation_area is None:
                closest = navmesh.query_closest_point(origin)
            else:
                closest = navmesh.query_closest_point(origin, self._navigation_area)
        except Exception as exc:
            raise RuntimeError(f"{self._prefix}NavMesh projection failed: {exc}") from exc

        if closest is None:
            raise RuntimeError(
                f"{self._prefix}NavMesh returned no closest point for ({x:.3f}, {y:.3f}, {world_z:.3f})"
            )

        return float(closest.z), True

    def _acquire_navmesh(self):
        if self._navmesh is not None:
            return self._navmesh
        navmesh = self._nav_interface.get_navmesh()
        if navmesh is None:
            raise RuntimeError(
                f"{self._prefix}NavMesh not ready; bake NavMesh before initializing GroundProbe for {self._robot_path}."
            )
        self._navmesh = navmesh
        return navmesh

