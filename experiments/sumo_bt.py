r"""sumo_bt -- drive SUMO's cars with a discovered behaviour tree, and watch it.

SUMO is the validated simulator; btind's world is a fused model of the same
intersection (its geometry was exported from this very net). Here SUMO does the
dynamics and the collision detection, and the TREE does the driving:

    every 0.5 s     the control interval the tree was trained at -- each car's
                    26 inputs are rebuilt by btind's own `observe_vehicles` from
                    SUMO's positions and speeds, and the tree picks its
                    acceleration (`MemBank`, the numpy reference of the kernel)
    every 0.1 s     that acceleration is applied through `setSpeed`, with SUMO's
                    own car-following and safety checks OFF (`speedMode 0`), as
                    `sumo_test/cosim_sumo.py` does for its controllers
    the signal      runs on btind's side (the fixed plan the car was trained
                    against), and is painted onto SUMO's traffic lights every
                    step, so the lights on screen are the ones the cars read
    crashes         counted by SUMO's own detector (`--collision.action warn`,
                    junctions checked), not by btind -- the independent check
    red-light runs  counted here, from the stop-line crossings

Cars are coloured by what the tree is commanding: red brakes hard (< -1.5 m/s^2),
orange brakes, yellow holds, green accelerates. A car is handed back to SUMO's
own driver once it is 40 m past the box, where btind's world ends.

Runs in the `car-following-sumo` env, which has SUMO and TraCI; btind's numba
kernels are not needed and are replaced by a no-op decorator, so only numpy code
paths execute.

    conda activate car-following-sumo
    python experiments/sumo_bt.py --gui                     # watch it, real time, 300 s
    python experiments/sumo_bt.py --gui --speed 4 --seconds 120
    python experiments/sumo_bt.py --gui --speed 0.25        # slow motion
    python experiments/sumo_bt.py --seconds 300 --seeds 5   # numbers, headless
    python experiments/sumo_bt.py --gui --car hand          # the hand-written follower
    python experiments/sumo_bt.py --gui --vph 1000          # heavier, every approach
    python experiments/sumo_bt.py --gui --vph 400 --north 1200 --split 0.5,0.3,0.2
    python experiments/sumo_bt.py -h                        # every option

The car is the most recently written best tree under `runs/` unless `--car`
says otherwise, so a training run in progress is picked up as it improves.

`--speed` is simulated seconds per wall-clock second, held by this script
against the clock (0 = as fast as the machine goes). SUMO's own delay slider
only ever slows it further. The old `key=value` form (`gui end=120 speed=4`)
still works.
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
SUMO_TEST = os.path.join(os.path.dirname(HERE), "sumo_test")

# btind's compiled kernels are not used here: a no-op numba lets the numpy code
# import in an env that has SUMO but no numba
if "numba" not in sys.modules:
    try:
        import numba  # noqa: F401
    except ImportError:
        nb = types.ModuleType("numba")

        def _njit(*a, **k):
            if a and callable(a[0]) and not k:
                return a[0]
            return lambda f: f
        nb.njit = nb.jit = _njit
        nb.prange = range
        sys.modules["numba"] = nb

import traci  # noqa: E402
import sumolib  # noqa: E402

from btind.envs.intersection import (A_MAX, ACCELS, FAR, FIXED_GREEN, HALF_CONF,  # noqa: E402
                                     N_PHASES, T_AR, T_TTC, TAIL_CONF, V0,
                                     IntersectionBatch)
from btind.memory import MemBank, upgrade_layout  # noqa: E402
from btind.runlog import bank_from_json  # noqa: E402

DT_SUMO = 0.1
DEFAULT_CAR = "latest"
APPROACHES = ("north", "south", "east", "west")


def latest_best(root=os.path.join(HERE, "runs")):
    """The most recently written best vehicle tree under `runs/`.

    A training run writes its best file every time the best improves, and runs
    execute from frozen snapshots (`runs/snap_eXX/runs/...`), so both places are
    searched. Inside a snapshot only that experiment's own best counts -- a
    snapshot also carries COPIES of older bests it was seeded from, and a fresh
    copy would otherwise look newer than the tree it was copied from.
    """
    found = []
    for dirpath, _, files in os.walk(root):
        rel = os.path.relpath(dirpath, root).replace("\\", "/")
        snap = rel.split("/")[0][5:] if rel.startswith("snap_") else None
        for f in files:
            if not (f.endswith(".json") and "best" in f):
                continue
            if snap is not None and not f.startswith(snap.rstrip("abcdefghijklmnopqrstuvwxyz")):
                continue
            path = os.path.join(dirpath, f)
            try:
                with open(path, encoding="utf-8") as fh:
                    if "veh" not in json.load(fh):
                        continue
            except (OSError, ValueError):
                continue
            found.append((os.path.getmtime(path), path))
    if not found:
        raise SystemExit("no best vehicle tree under %s -- pass --car <json> or --car hand" % root)
    return max(found)[1]
NET = os.path.join(SUMO_TEST, "sumo_files", "intersection_tl.net.xml")
COLOR = {"hard": (220, 30, 30, 255), "brake": (255, 140, 0, 255),
         "hold": (240, 220, 40, 255), "go": (40, 200, 60, 255),
         "free": (170, 170, 170, 255)}


def load_car(env, which):
    if which == "hand":
        return env.default_vehicle_bank(), "hand-written follower"
    if which == "latest":
        which = latest_best()
    with open(which, encoding="utf-8") as fh:
        blob = json.load(fh)
    b = bank_from_json(blob["veh"])
    b = upgrade_layout(b, list(env.veh_names), old_names=list(b.get("names") or env.veh_names))
    b["names"] = list(env.veh_names)
    print("car: %s  (written %s, train score %s)"
          % (os.path.relpath(which, HERE), time.strftime("%m-%d %H:%M",
             time.localtime(os.path.getmtime(which))), blob.get("train")), flush=True)
    b["_test_inputs"] = blob.get("test_inputs")
    return b, os.path.basename(which)


def gap_margin(env, s, gap_r, priority=False):
    """For each car, the smallest over conflicting vehicles within `gap_r` m of
    (when THAT vehicle reaches our shared conflict point, if it floored it now)
    minus (when I would be clear of that point, if I floored it now). Positive:
    I am through before it can arrive. FAR when nothing conflicts.

    A TEST INPUT, not part of btind's world: it exists so a tree file that
    declares it (`"test_inputs"`) can be watched here. It is the same quantity
    the scratch kernel of the gate experiments computed, worst-case arrival
    for the rival and full throttle for me, both capped at V0.
    """
    m, S, V, st, dep, X = env._unpack(s)
    n, N = S.shape
    g = env.geom
    act = st == 1.0
    x, y, _, _ = env._xy(m, S)
    dist = np.hypot(x[:, None, :] - x[:, :, None], y[:, None, :] - y[:, :, None])
    eye = np.eye(N, dtype=bool)[None]
    mq, mo = m[:, :, None], m[:, None, :]
    d_me = g["s_cp"][mq, mo] - S[:, :, None]
    d_rv = g["s_cp"][mo, mq] - S[:, None, :]
    ok = (act[:, :, None] & act[:, None, :] & ~eye & (dist <= gap_r) & g["conf"][mq, mo]
          & (d_me > -TAIL_CONF) & (d_rv > -TAIL_CONF))

    def t_reach(d, v):
        ta = (V0 - v) / A_MAX
        da = v * ta + 0.5 * A_MAX * ta * ta
        return np.where(d <= da, (-v + np.sqrt(v * v + 2.0 * A_MAX * d)) / A_MAX,
                        ta + (d - da) / V0)
    eta = t_reach(np.maximum(d_rv - HALF_CONF, 0.0), np.broadcast_to(V[:, None, :], d_rv.shape))
    clr = t_reach(np.maximum(d_me + TAIL_CONF, 0.0), np.broadcast_to(V[:, :, None], d_me.shape))
    if priority:
        # A RIVAL WAITING AT ITS OWN RED STAYS THERE -- when I have priority over
        # it, by a fixed order (approach, then movement) that gives exactly one of
        # any pair priority, so two red-waiters cannot both count on the other
        green = env._green_now(X)                                   # (n, M)
        g_rv = np.take_along_axis(green, m, 1)[:, None, :]
        ap = g["appr"]
        mine = (ap[mq] < ap[mo]) | ((ap[mq] == ap[mo]) & (mq < mo))
        ok = ok & ~((V[:, None, :] < 0.5) & ~g_rv & mine)
    gm = np.where(ok, eta - clr, np.inf).min(2)
    return np.where(np.isfinite(gm), gm, FAR)


def install_test_inputs(env, spec):
    """Apply a tree file's declared test inputs to this world's observation."""
    if not spec:
        return
    slot, gap_r = spec["gap_margin_slot"], float(spec.get("gap_r", 100.0))
    prio = bool(spec.get("priority", False))
    j = list(env.veh_names).index(slot)
    base = env.observe_vehicles

    def observe(s):
        o = base(s)
        n, N = s.shape[0], env.N
        act = (env._unpack(s)[3] == 1.0).reshape(-1)
        col = gap_margin(env, s, gap_r, prio).reshape(-1)
        o[:, j] = np.where(act, col, 0.0)
        return o
    env.observe_vehicles = observe
    print("TEST INPUT: column %r now carries gap_margin over %g m%s (declared by the car "
          "file; btind's world does not have it)"
          % (slot, gap_r, ", red-waiters I have priority over ignored" if prio else ""), flush=True)


def write_routes(path, vph, end, depart_pos=99.6):
    """One flow per movement, as sumo_test's cosim does, over `end` seconds.

    Cars enter where btind's spawn them, 90 m before the stop line: SUMO's
    approach is 190 m and the tree was never trained on its first 100 m."""
    sys.path.insert(0, SUMO_TEST)
    import turns_geom as G
    mv = G.movements(NET)
    with open(path, "w") as f:
        f.write('<routes>\n  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" '
                'minGap="1.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
        for m in mv:
            # `vph` is {movement dir: rate}, the same on every approach, or
            # {approach: {dir: rate}} to load approaches differently
            appr = m.frm.split("_")[0]
            per = vph.get(appr, vph) if isinstance(vph.get(appr), dict) else vph
            rate = per.get(m.dir, 0) / 3600.0
            if rate > 0:
                f.write('  <flow id="f%d" type="car" from="%s" to="%s" begin="0" end="%.0f" '
                        'period="exp(%.5f)" departLane="best" departSpeed="%.2f" '
                        'departPos="%.1f"/>\n'
                        % (m.idx, m.frm, m.to, end, rate, V0, depart_pos))
        f.write('</routes>\n')


class Bridge:
    """btind's state for one episode, kept in step with SUMO."""

    def __init__(self, env, bank, n_slots):
        self.env, self.N = env, n_slots
        self.s = np.zeros((1, env.k))
        self.s[0, 4 * n_slots:5 * n_slots] = np.inf            # no pending arrivals
        self.X = self.s[:, 5 * n_slots:5 * n_slots + env.N_MISC]
        env.plan(self.s)[:] = FIXED_GREEN
        self.pol = MemBank(bank, len(env.veh_names))
        self.pol.reset(n_slots)
        self.slot, self.free = {}, list(range(n_slots))[::-1]
        g = env.geom
        self.mv_of = {str(n).split("(")[0]: i for i, n in enumerate(g["names"])}
        self.acc = np.zeros(n_slots)
        self.released = set()
        self.obs, self._zmat = None, None

    def movement(self, route):
        return self.mv_of.get("%s->%s" % (route[0], route[-1]))

    def admit(self, vid, m):
        q = self.free.pop()
        self.slot[vid] = q
        N = self.N
        self.s[0, q], self.s[0, 3 * N + q] = m, 1.0
        self.pol.latch[q], self.pol.step[q], self.pol.have[q], self.pol.age[q] = -1, 0, False, 0
        if self.pol.slots is not None and self.pol.slots.shape[1]:
            self.pol.slots[q] = 0.0
        self.acc[q] = 0.0
        return q

    def drop(self, vid):
        q = self.slot.pop(vid, None)
        if q is not None:
            self.s[0, 3 * self.N + q] = 2.0
            self.free.append(q)

    def set_car(self, q, s_arc, v):
        N = self.N
        self.s[0, N + q], self.s[0, 2 * N + q] = s_arc, v

    def signal_step(self, dt):
        X = self.X[0]
        if X[2] > 0.5:
            X[3] += dt
            if X[3] >= T_AR - 1e-9:
                X[0], X[1], X[2], X[3] = (X[0] + 1) % N_PHASES, 0.0, 0.0, 0.0
        else:
            X[1] += dt
            if X[1] >= FIXED_GREEN[int(X[0])] - 1e-9:
                X[2], X[3] = 1.0, 0.0

    def green(self, m):
        X = self.X[0]
        return (X[2] < 0.5) and bool(self.env.geom["green"][int(X[0]), m])

    def decide(self):
        """The tree's acceleration for every active slot."""
        o = self.env.observe_vehicles(self.s)
        self.obs, self._zmat = o, None
        a = np.asarray(self.pol.act(o), float).reshape(-1)
        if self.pol.b.get("head") == "argmax":
            # a discrete tree returns an action INDEX, not an acceleration
            a = ACCELS[np.clip(np.rint(a), 0, len(ACCELS) - 1).astype(int)]
        self.acc = np.clip(a, -4.5, 2.6)
        self.X[0, 4] += 1.0

    def why(self, q):
        """Which arm claims slot `q`, in the tree's own evaluation order.

        On a SCRATCH policy: `z` sizes the memory arrays to the rows it is
        given, so asking the live one about a single slot resized the latch it
        drives every car with.
        """
        if self._zmat is None:
            scan = MemBank(self.pol.b, len(self.env.veh_names))
            scan.reset(len(self.obs))
            self._zmat = scan.z(self.obs, update=False)
        z = self._zmat[q]
        for c, cl in enumerate(self.pol.b["clauses"]):
            if all((z[j] <= th) if neg else (z[j] > th) for j, th, neg in cl):
                return c, cl
        return -1, []


class Audit:
    """btind's own collision tests and TTC, run on the mirrored state and held
    against SUMO's collision detector.

    Every SUMO step (0.1 s), on the cars btind is driving:
        rear        `_leaders` gap < 0          (btind's rear-end test)
        cross       both cars within HALF_CONF of their shared conflict point
                    (btind's crossing test)
        lead_ttc    gap / closing speed to the `_leaders` leader
        rival_ttc   `_cross_ttc` between conflicting movements, inside SENSE_R
    and for every collision SUMO reports: did btind's test fire on the SAME
    pair, and did either TTC fall below T_TTC before it -- and how early.
    """

    def __init__(self, env, window=1.0, lookback=6.0):
        self.env, self.window, self.lookback = env, window, lookback
        self.flag = {}          # pair -> [(t, kind)]  btind's collision tests
        self.ttc = {}           # pair -> [(t, kind, ttc)]  when ttc < 2 * T_TTC
        self.hits = []          # SUMO collisions
        self.where = {}         # vid -> (movement, s_arc) at its last tick

    @staticmethod
    def key(a, b):
        return tuple(sorted((a, b)))

    def tick(self, t, br):
        from btind.envs.intersection import HALF_CONF, SENSE_R, T_V_MIN
        if len(br.slot) < 2:
            return
        env, g, N = self.env, self.env.geom, br.N
        vids = list(br.slot)
        qs = np.array([br.slot[v] for v in vids])
        m = br.s[0, qs].astype(int)[None]
        S = br.s[0, N + qs][None]
        V = br.s[0, 2 * N + qs][None]
        K = len(vids)
        act = np.ones((1, K), bool)
        for i, v in enumerate(vids):
            self.where[v] = (int(m[0, i]), float(S[0, i]))
        # rear-end and lead TTC, exactly as observe_vehicles builds them
        has, gap, j = env._leaders(m, S, act)
        dvl = np.where(has, V - np.take_along_axis(V, j, 1), 0.0)
        lttc = np.where(has & (dvl > 1e-6),
                        np.maximum(gap, 0.0) / np.maximum(dvl, 1e-6), np.inf)
        for i in np.nonzero(has[0])[0]:
            k = self.key(vids[i], vids[j[0, i]])
            if gap[0, i] < 0.0:
                self.flag.setdefault(k, []).append((t, "rear"))
            if lttc[0, i] < 2 * T_TTC:
                self.ttc.setdefault(k, []).append((t, "lead", float(lttc[0, i])))
        # crossing test and rival TTC, pairwise
        mq, mo = m[:, :, None], m[:, None, :]
        conf = g["conf"][mq, mo].copy()
        conf[:, np.arange(K), np.arange(K)] = False
        s_me, s_rv = g["s_cp"][mq, mo], g["s_cp"][mo, mq]
        both = (conf & (np.abs(s_me - S[:, :, None]) < HALF_CONF)
                & (np.abs(s_rv - S[:, None, :]) < HALF_CONF))
        for a, b in zip(*np.nonzero(np.triu(both[0], 1))):
            self.flag.setdefault(self.key(vids[a], vids[b]), []).append((t, "cross"))
        x, y, _, _ = env._xy(m, S)
        dist = np.hypot(x[:, None, :] - x[:, :, None], y[:, None, :] - y[:, :, None])
        d_me, d_rv = s_me - S[:, :, None], s_rv - S[:, None, :]
        ok = conf & (dist <= SENSE_R) & (d_me > -HALF_CONF) & (d_rv > -HALF_CONF)
        tc = env._cross_ttc(ok, d_me, d_rv, np.maximum(V[:, :, None], T_V_MIN),
                            np.maximum(V[:, None, :], T_V_MIN))
        for a, b in zip(*np.nonzero(np.triu(np.minimum(tc[0], tc[0].T) < 2 * T_TTC, 1))):
            self.ttc.setdefault(self.key(vids[a], vids[b]), []).append(
                (t, "rival", float(min(tc[0, a, b], tc[0, b, a]))))

    def sumo_hit(self, t, c, br):
        k = self.key(c.collider, c.victim)
        scope = c.collider in br.slot and c.victim in br.slot
        self.hits.append(dict(t=t, pair=k, type=str(getattr(c, "type", "?")),
                              lane=str(getattr(c, "lane", "")), scope=scope,
                              where={v: self.where.get(v) for v in k}))

    def report(self):
        W, L = self.window, self.lookback
        names = self.env.geom["names"]
        n_scope, n_flag, n_warn, leads = 0, 0, 0, []
        print("\nAUDIT -- each SUMO collision against btind's tests on the same pair")
        print("%6s  %-9s %-24s %-10s %-12s %s" % ("t", "sumo type", "where (btind)", "btind test",
                                                  "min TTC", "TTC < %.0fs" % T_TTC))
        for h in self.hits:
            k, t = h["pair"], h["t"]
            fl = [kd for tt, kd in self.flag.get(k, []) if abs(tt - t) <= W]
            tt_ = [r for r in self.ttc.get(k, []) if t - L <= r[0] <= t + 0.05]
            kind = "/".join(sorted(set(fl))) or "-"
            mn = min(tt_, key=lambda r: r[2]) if tt_ else None
            warn = [r[0] for r in tt_ if r[2] < T_TTC]
            lead = t - min(warn) if warn else None
            w = h["where"][k[0]]
            loc = (str(names[w[0]]).split("(")[0] + " s%.0f" % w[1]) if w else h["lane"]
            if h["scope"]:
                n_scope += 1
                n_flag += bool(fl)
                n_warn += lead is not None
                if lead is not None:
                    leads.append(lead)
            print("%6.1f  %-9s %-24s %-10s %-12s %s%s" % (
                t, h["type"][:9], loc[:24], kind,
                "%s %.2fs" % (mn[1], mn[2]) if mn else "none",
                "%.1fs before" % lead if lead is not None else "never",
                "" if h["scope"] else "   (outside btind: a car was released)"))
        hit_pairs = {h["pair"] for h in self.hits}
        extra = {}
        for k, ev in self.flag.items():
            if k not in hit_pairs:
                extra[ev[0][1]] = extra.get(ev[0][1], 0) + 1
        print("\nSUMO collisions %d, between two cars btind was driving %d"
              % (len(self.hits), n_scope))
        print("  btind's collision test fired on the same pair (+-%.0fs): %d / %d"
              % (W, n_flag, n_scope))
        print("  btind's TTC fell below %.0fs on the same pair first:     %d / %d%s"
              % (T_TTC, n_warn, n_scope,
                 "   (median warning %.1fs)" % np.median(leads) if leads else ""))
        print("  pairs btind called a collision that SUMO never reported: %s"
              % (extra or "none"))


def run(car=DEFAULT_CAR, vph=None, end=300.0, seed=0, gui=False, speed=1.0, n_slots=256,
        verbose=True, keep_open=True, report_every=30.0, audit=False, stuck=0.0):
    vph = vph or dict(s=500, l=150, r=150)
    env = IntersectionBatch(n_max=n_slots, veh_reward="car", veh_head="scalar",
                            sig_head="duration")
    env.set_agent("vehicle")
    bank, label = load_car(env, car)
    install_test_inputs(env, bank.pop("_test_inputs", None))
    br = Bridge(env, bank, n_slots)
    aud = Audit(env) if audit else None
    g = env.geom
    routes = os.path.join(HERE, "runs", "_sumo_bt_routes.rou.xml")
    os.makedirs(os.path.dirname(routes), exist_ok=True)
    write_routes(routes, vph, end)
    binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    cmd = [binary, "-n", NET, "-r", routes, "--begin", "0", "--end", str(end),
           "--step-length", str(DT_SUMO), "--seed", str(seed), "--no-step-log", "true",
           "--no-warnings", "true", "--collision.action", "warn",
           "--collision.check-junctions", "true", "--collision.mingap-factor", "0",
           "--time-to-teleport", "-1"]
    if gui:
        # pacing is done here against the wall clock, so SUMO's delay starts at 0
        cmd += ["--start", "--delay", "0"] + ([] if keep_open else ["--quit-on-end"])
    traci.start(cmd)
    tls = traci.trafficlight.getIDList()[0]
    links = traci.trafficlight.getControlledLinks(tls)
    link_mv = []
    for L in links:
        if not L:
            link_mv.append(None)
            continue
        frm, to = L[0][0].rsplit("_", 1)[0], L[0][1].rsplit("_", 1)[0]
        link_mv.append(br.mv_of.get("%s->%s" % (frm, to)))
    if gui:
        try:
            traci.gui.setSchema("View #0", "real world")
            traci.gui.setBoundary("View #0", 120, 120, 280, 280)
        except traci.TraCIException:
            pass
    per_tick = int(round(env.dt / DT_SUMO))
    s_stop, s_exit = g["s_stop"], env.s_exit
    collided, coll_events, red_runs, arrived, speeds = set(), 0, 0, 0, []
    still, held = {}, {}
    last_s, s0, pairs = {}, {}, set()
    n_steps = int(round(end / DT_SUMO))
    rep_every = int(round(report_every / DT_SUMO)) if report_every else 0
    wall0 = time.perf_counter()
    for step in range(n_steps):
        br.signal_step(DT_SUMO)
        state = "".join("r" if m is None or not br.green(m) else "G" for m in link_mv)
        traci.trafficlight.setRedYellowGreenState(tls, state)
        for vid in traci.vehicle.getIDList():
            if vid in br.released:
                continue
            if vid not in br.slot:
                m = br.movement(traci.vehicle.getRoute(vid))
                if m is None or not br.free:
                    continue
                br.admit(vid, m)
                traci.vehicle.setSpeedMode(vid, 0)
                traci.vehicle.setLaneChangeMode(vid, 0)
            q = br.slot[vid]
            m = int(br.s[0, q])
            # btind's arc-length is the front bumper along the movement's path:
            # on the approach edge that is SUMO's lane position, and the
            # odometer carries it through the box onto the exit edge
            if vid not in s0:
                s0[vid] = traci.vehicle.getLanePosition(vid) - traci.vehicle.getDistance(vid)
            s_arc = traci.vehicle.getDistance(vid) + s0[vid]
            v = traci.vehicle.getSpeed(vid)
            br.set_car(q, s_arc, v)
            prev = last_s.get(vid, -1e9)
            if prev < s_stop[m] <= s_arc and not br.green(m):
                red_runs += 1
            last_s[vid] = s_arc
            if s_arc >= s_exit[m]:
                traci.vehicle.setSpeedMode(vid, 31)
                traci.vehicle.setColor(vid, COLOR["free"])
                br.released.add(vid)
                br.drop(vid)
        if aud is not None:
            aud.tick(step * DT_SUMO, br)
        # A CAR THAT STOPS AND STAYS STOPPED, separating the stop that has a
        # reason -- a red light for my own movement, or something close enough
        # ahead -- from the one that does not, and naming the arm that
        # commanded the unexcused part.
        if stuck > 0 and getattr(br, "obs", None) is not None:
            now = step * DT_SUMO
            nm = env.veh_names
            for vid, q in list(br.slot.items()):
                if br.s[0, 2 * br.N + q] > 0.3:
                    still.pop(vid, None)
                    continue
                t0 = still.setdefault(vid, now)
                o = br.obs[q]
                col = lambda k: float(o[nm.index(k)])
                gm, lg, has = col("green"), col("lead_gap"), col("has_lead")
                excused = ((gm < 0.5 and col("d_stop") > 0.0)
                           or (has > 0.5 and lg < 8.0))
                r_ = held.setdefault(vid, dict(
                    t0=t0, secs=0.0, free=0.0, arms={},
                    mv=str(g["names"][int(br.s[0, q])]).split("(")[0]))
                r_["secs"] += DT_SUMO
                if not excused:
                    r_["free"] += DT_SUMO
                    c, cl = br.why(q)
                    gd = " AND ".join("%s%s%.3f" % (nm[j] if j < len(nm) else j,
                                                    "<=" if ng else ">", th)
                                      for j, th, ng in cl) or "default"
                    a_ = r_["arms"].setdefault((c, gd), [0.0, 0.0, 0.0])
                    a_[0] += DT_SUMO
                    a_[1] += float(br.acc[q]) * DT_SUMO
                    a_[2] += (lg if has > 0.5 else 200.0) * DT_SUMO
        if step % per_tick == 0:
            br.decide()
        for vid, q in br.slot.items():
            a = br.acc[q]
            v = br.s[0, 2 * br.N + q]
            traci.vehicle.setSpeed(vid, float(min(max(v + a * DT_SUMO, 0.0), V0)))
            speeds.append(v)
            if gui:
                traci.vehicle.setColor(vid, COLOR["hard" if a < -1.5 else "brake" if a < -0.2
                                                  else "hold" if a < 0.2 else "go"])
        traci.simulationStep()
        arrived += traci.simulation.getArrivedNumber()
        for vid in traci.simulation.getArrivedIDList():
            br.drop(vid)
            br.released.discard(vid)
            last_s.pop(vid, None)
        # A COLLISION IS COUNTED ONCE. With `--collision.action warn` the cars
        # stay where they are and SUMO reports the same overlapping pair every
        # step, so raw reports were counted in the thousands for a few dozen
        # crashes.
        try:
            cols = traci.simulation.getCollisions()
        except AttributeError:
            cols = []
        for c in cols:
            key = tuple(sorted((c.collider, c.victim)))
            if key not in pairs:
                pairs.add(key)
                coll_events += 1
                if aud is not None:
                    aud.sumo_hit((step + 1) * DT_SUMO, c, br)
        for vid in traci.simulation.getCollidingVehiclesIDList():
            collided.add(vid)
        if verbose and rep_every and (step + 1) % rep_every == 0:
            print("  t %5.0fs  on road %3d  arrived %4d  collisions %3d  red runs %3d"
                  % ((step + 1) * DT_SUMO, traci.vehicle.getIDCount(), arrived,
                     coll_events, red_runs), flush=True)
        if gui and speed > 0:
            # hold `speed` simulated seconds per wall-clock second
            ahead = wall0 + (step + 1) * DT_SUMO / speed - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
    traci.close()
    if aud is not None:
        aud.report()
    if stuck > 0:
        rows = [r_ for r_ in held.values() if r_["secs"] >= stuck]
        print("\nSTOPPED %.0fs OR MORE: %d cars, %.0f car-seconds -- of which "
              "%.0f UNEXCUSED (green for its own movement, nothing within 8 m)"
              % (stuck, len(rows), sum(r_["secs"] for r_ in rows),
                 sum(r_["free"] for r_ in rows)))
        by = {}
        for r_ in rows:
            for k, v in r_["arms"].items():
                b_ = by.setdefault(k, [0.0, 0.0, 0.0, 0])
                b_[0] += v[0]
                b_[1] += v[1]
                b_[2] += v[2]
                b_[3] += 1
        for (c, gd), (secs, wa, wl, n) in sorted(by.items(), key=lambda kv: -kv[1][0])[:8]:
            print("    arm %2d  %6.1f car-seconds, %3d cars  command %+.2f  "
                  "mean gap ahead %5.1f m   %s"
                  % (c, secs, n, wa / max(secs, 1e-9), wl / max(secs, 1e-9), gd))
        for r_ in sorted(rows, key=lambda x: -x["free"])[:6]:
            print("    worst: %-22s stopped %5.1fs, %5.1fs unexcused (from t=%.0f)"
                  % (r_["mv"], r_["secs"], r_["free"], r_["t0"]))
    res = dict(car=label, vph=vph, end=end, seed=seed, arrived=arrived, audit=aud,
               collisions=coll_events, collided_cars=len(collided), red_runs=red_runs,
               mean_speed=float(np.mean(speeds)) if speeds else 0.0)
    if verbose:
        print("%-24s seed %d  %gs  arrived %4d  collisions %3d (%d cars)  red runs %3d  "
              "mean speed of controlled cars %.1f m/s"
              % (label, seed, end, arrived, coll_events, len(collided), red_runs,
                 res["mean_speed"]), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Drive SUMO's cars with a discovered behaviour tree.")
    ap.add_argument("--gui", action="store_true", help="open sumo-gui and watch")
    ap.add_argument("--seconds", "--end", dest="seconds", type=float, default=300.0,
                    help="simulated seconds to run (default 300)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="GUI playback: simulated s per wall-clock s; 1 = real time, "
                         "0 = as fast as possible (default 1)")
    ap.add_argument("--car", default=DEFAULT_CAR,
                    help="'latest' (default: the most recently written best tree "
                         "under runs/, including a training run in progress), a "
                         "best-tree json, or 'hand' for the hand-written follower")
    ap.add_argument("--vph", type=float, default=800.0,
                    help="demand on EACH approach, veh/h (default 800; the e39 tree "
                         "was trained on 300-1000)")
    ap.add_argument("--split", default="0.625,0.1875,0.1875",
                    help="through,left,right fractions of an approach's demand "
                         "(default 0.625,0.1875,0.1875)")
    for ap_name in APPROACHES:
        ap.add_argument("--" + ap_name, type=float, default=None,
                        help="veh/h on the %s approach, overriding --vph" % ap_name)
    ap.add_argument("--s", type=float, default=None,
                    help="through veh/h per approach, overriding the split")
    ap.add_argument("--l", type=float, default=None, help="left-turn veh/h per approach")
    ap.add_argument("--r", type=float, default=None, help="right-turn veh/h per approach")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat on k seeds (only the first is shown in the GUI)")
    ap.add_argument("--report", type=float, default=30.0,
                    help="print a progress line every this many simulated s (0 = off)")
    ap.add_argument("--stuck", type=float, default=0.0, metavar="SECONDS",
                    help="report every car that stays stopped this long, with "
                         "what it saw and which arm commanded it")
    ap.add_argument("--audit", action="store_true",
                    help="check each SUMO collision against btind's own collision "
                         "tests and TTC on the same pair")
    ap.add_argument("--quit", action="store_true",
                    help="close the GUI when the run ends (default: leave it open)")
    argv = sys.argv[1:] if argv is None else argv
    # the older `gui end=120 speed=4` form
    argv = ["--" + a if not a.startswith("-") and ("=" in a or a in ("gui", "quit"))
            else a for a in argv]
    a = ap.parse_args(argv)

    car = a.car
    # a relative path is tried where the command was typed first, then from the
    # repo root -- `--car ../runs/x.json` from experiments/ and `--car
    # runs/x.json` from anywhere both name the same file
    if car not in ("hand", "latest") and not os.path.isabs(car):
        car = (os.path.abspath(car) if os.path.exists(car)
               else os.path.join(HERE, car))
        if not os.path.exists(car):
            ap.error("--car %s: no such file (looked in %s and %s)"
                     % (a.car, os.getcwd(), HERE))
    frac = [float(x) for x in a.split.split(",")]
    if len(frac) != 3 or min(frac) < 0 or sum(frac) <= 0:
        ap.error("--split needs three non-negative fractions: through,left,right")
    frac = [f / sum(frac) for f in frac]
    vph = {}
    for name in APPROACHES:
        tot = getattr(a, name)
        tot = a.vph if tot is None else tot
        per = dict(zip("slr", (tot * f for f in frac)))
        for d in "slr":
            # an explicit per-turn rate applies where no approach total was given
            if getattr(a, d) is not None and getattr(a, name) is None:
                per[d] = getattr(a, d)
        vph[name] = per
    print("demand veh/h (through/left/right): " + "  ".join(
        "%s %.0f/%.0f/%.0f" % (n, v["s"], v["l"], v["r"]) for n, v in vph.items()),
        flush=True)
    out = [run(car, vph, a.seconds, a.seed + k, a.gui and k == 0, a.speed,
               keep_open=not a.quit, report_every=a.report, audit=a.audit,
               stuck=a.stuck) for k in range(a.seeds)]
    if a.seeds > 1:
        tot = lambda key: sum(r[key] for r in out)
        print("\n%d seeds: arrived %d, collisions %d, red runs %d"
              % (a.seeds, tot("arrived"), tot("collisions"), tot("red_runs")))


if __name__ == "__main__":
    main()
