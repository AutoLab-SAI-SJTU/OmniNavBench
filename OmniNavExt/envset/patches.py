"""Runtime patches/workarounds for upstream issues."""

import carb


def install_safe_simtimes_guard():
    """Prevent Replicator from writing empty simTimesToWrite arrays.

    When timeline never entered PLAY, Replicator may attempt to push an empty list into
    DispatchSync.inputs:simTimesToWrite, which raises a TypeError. Guarding the AttributeValueHelper
    avoids the hard failure while we make sure the timeline starts correctly.
    """
    try:
        import omni.graph.core as og
    except ImportError:
        carb.log_warn("[IRA] omni.graph.core not available; simTimes guard skipped")
        return

    helper_cls = og.AttributeValueHelper
    if getattr(helper_cls, "_ira_safe_simtimes_installed", False):
        return

    original_set = helper_cls.set

    def safe_set(self, value, *args, **kwargs):  # type: ignore[override]
        if value is None:
            return

        # We do not transform payload shape; only gate obviously invalid content.
        try:
            # supports __len__?
            if hasattr(value, "__len__") and len(value) == 0:
                carb.log_warn("[IRA] Skip writing empty simTimesToWrite payload to DispatchSync.")
                return

            # supports iteration? verify at least one valid (num,den) pair
            has_any_valid = False
            for entry in value:  # may raise TypeError if not iterable
                try:
                    _, den = entry
                except Exception:
                    continue
                if den not in (0, -1):
                    has_any_valid = True
                    break
            if not has_any_valid:
                carb.log_warn("[IRA] Skip writing simTimesToWrite payload with invalid denominators.")
                return
        except TypeError:
            # Not iterable/length-less: forward as-is
            return original_set(self, value, *args, **kwargs)
        except Exception as exc:
            carb.log_warn(f"[IRA] Failed to inspect simTimesToWrite payload: {exc}")
            return original_set(self, value, *args, **kwargs)

        return original_set(self, value, *args, **kwargs)

    helper_cls.set = safe_set  # type: ignore[assignment]
    helper_cls._ira_safe_simtimes_installed = True
