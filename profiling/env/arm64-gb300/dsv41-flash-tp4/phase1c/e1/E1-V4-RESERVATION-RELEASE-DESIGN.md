# Optional early deletion of the reservation after a verified GPU release — design and CPU test (not enabled)

**Status: designed and CPU-tested only. Not enabled for V4 and not called by any V4 script.** `E1-V4-SCALE-INPUTS.json`
registers `reservation.early_release_after_verified_gpu_release.enabled: false`. The V4 default is the existing lifecycle:
the reservation pod is deleted only after a CLEAN closeout.

## 1. Why it is worth having

After the GPUs are released the run needs about 20 more minutes of CPU-only work (the launcher's export and parse) before
closeout. Under the existing lifecycle the reservation holds four idle GPUs for that time, and for a full E1 it would do so for
about 90 minutes (`feasibility/E1-NEXT-EXPERIMENT-FEASIBILITY.md`). Deleting the pod at the verified release would cut V4's
reservation to 42 – 46 min.

## 2. What the reservation pod is

`k8s/dsv41-gb300/reserve-tray16.yaml` (fixture: `tests/fixtures/e1_v4_scale/reserve_pod.json`): one `pause` container that
requests the tray's four GPUs, pinned to the node, no volume except the service-account token, no host namespace. It runs
nothing on the GPUs. The capture runs host-level, as user `dc`, started over ssh in a tmux session.

## 3. Can a deletion harm the run? Evidence and gaps

| concern | evidence | status |
|---|---|---|
| terminate the importer or any run process | a pod deletion stops that pod's containers. The run's processes are not in the pod: they are host processes in the user's session scope (`/user.slice/user-1000.slice/session-….scope`, read on the tray), not under `kubepods`. Check C4 verifies this for every live recorded process at decision time | structural argument + check; **a deletion while an importer is running has never been performed** |
| destroy preserved files | the pod mounts no host path and no persistent volume (check C5). V1, V2 and V3 reservations were deleted while the preserved streams were in the tray's `/tmp`; the V3 stream's SHA-256 was verified after the deletion and again today | measured for the existing sequence |
| interrupt required CPU-only processing | V3's post-run (offline import, 15.6 GiB importer, export, checkers) ran after the reservation had been deleted | measured: CPU-only processing does not need the pod |
| CPU or memory pressure from a new GPU pod on the node | after the deletion the scheduler may place a GPU pod on the tray. Its host-memory use is not bounded by this protocol; the memory sampler and the 64 GiB floor keep watching | **not excluded**; would have to be accepted |
| invalidate cleanup evidence | the driver's final GPU census, the closeout's "GPU memory idle" lines and gate G6's census are taken **after** the export. With another pod on the GPUs they would show memory held and fail or mislead | **real conflict**: see §5 |

## 4. The decision check (`checks/e1_reservation_release.py`)

Never talks to Kubernetes, sends no signal, deletes nothing. `WOULD_ALLOW` only if all of:

- C1 the controller recorded the GPU release and the server is not alive;
- C2 the **independent** GPU sampler shows at least 5 consecutive released observations over at least 10 s since then, none
  held after the first released one, the newest at most 30 s old;
- C3 a census taken now shows no compute process and idle memory;
- C4 every live recorded process (registries, importer, Nsight agent) has a readable cgroup outside every pod;
- C5 the pod is the reservation pod on this node: pause container only, no host namespace, no host or persistent volume;
- C6 the preserved stream exists with the recorded size;
- and the evidence record is written before any deletion.

Anything unknown refuses. Exit 4 ("would allow, not enabled") is the only non-refusing result possible with the registered
inputs. Tests (`tests/test_e1_reservation_release.py`, 31): the allowing case; 27 single-defect refusals (release not recorded,
sampler still held, compute process, too few or too old observations, another host, census defects, a process inside a pod
cgroup, unreadable cgroup, missing or wrong pod, host PID namespace, host path, non-pause container, stream missing or
truncated); boundaries; the command line on a live process of the test itself; no Kubernetes or signalling call in the source.

## 5. What would have to change before it could be enabled (not done)

1. Gates and closeout must take their GPU-idle evidence from **before** the deletion (the evidence record of the decision
   check plus the sampler), and must tell this run's GPU use from a later pod's (by registered process identity, which the
   pinned census already records). Today G6's census, the closeout's GPU lines and the driver's `t20` census would be affected.
2. The deletion itself must stay an operator action from the laptop; the check runs on the tray and only advises.
3. One CPU-only demonstration on the cluster with a stand-in importer (a pod deleted while a host process runs) would turn the
   structural argument of §3 into a measurement. It needs a reservation pod and therefore its own authorization.
4. A stated position on a new pod's host-memory and CPU use during the export.
