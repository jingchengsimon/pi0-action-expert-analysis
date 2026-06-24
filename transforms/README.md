# Robot I/O transforms — DROID ↔ LIBERO

Tracking doc for how raw robot/sim observations and actions map into the π0 policy
interface, and how they differ between **DROID** (real-robot data schema π0 was
trained on) and **LIBERO** (sim benchmark used for closed-loop eval).

> This is the "追踪文档" — keep it in sync with `droid_io.py` and `libero_io.py`.

## Schema comparison

| Field | DROID | LIBERO | Notes |
|---|---|---|---|
| RGB cameras | TODO (e.g. exterior_1, wrist) | TODO (agentview, eye_in_hand) | resolution / naming differ |
| Proprio state | TODO | TODO | joint vs eef pose |
| Action space | TODO (e.g. joint velocity) | TODO (e.g. OSC delta) | dimensionality + units |
| Control freq | TODO | TODO | affects action horizon |
| Normalization | TODO (norm stats source) | TODO | must match training stats |

## Open questions / gotchas

- [ ] TODO: confirm camera key naming expected by the pi0 server.
- [ ] TODO: action un-normalization — which norm stats apply per embodiment.
- [ ] TODO: gripper convention (open/close sign) per dataset.
