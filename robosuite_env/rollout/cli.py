# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Command-line contract for one simulator worker."""

from __future__ import annotations

import argparse
import os

from real2sim.wrist_profiles import WRIST_BACKENDS
from scene.gs_background_protocol import DEFAULT_KB4_NEAR_CULL, KB4_NEAR_CULL_MODES
from robosuite_env.robots.fr3.controllers import FR3_CONTROLLERS
from robosuite_env.paths import PACKAGE_ROOT
from robosuite_env.rendering.wrist_render_profiles import (
    DEFAULT_POLICY_WRIST_RENDER_PROFILE,
    WRIST_RENDER_PROFILE_NAMES,
)
from robosuite_env.robots.profiles import (
    SUPPORTED_ROBOT_NAMES,
    normalize_robot_name,
)


_DIR = str(PACKAGE_ROOT)

_GROUP_TITLES = (
    "Official/launcher",
    "Source",
    "Robot & IK",
    "Gripper",
    "Timing",
    "Scene/3DGS",
    "Objects & init-state",
    "Render/Output",
    "Success criteria",
    "Debug/Dump",
    "Grasp-test script",
)

_OFFICIAL_OPTIONS = """
--strict-assets --official-run-manifest --official-scene-tag
--official-worker-contract
""".split()

_SOURCE_OPTIONS = """
--replay --policy --policy-batch-server --ik-replay --state-log-replay
--grasp-test --state-replay-policy-obs-dir --state-replay-policy-obs-display
--instruction --config --config-src --ah --policy-batch-client-id
--policy-batch-timeout-sec --policy-delta-anchor --policy-pose-repr
--policy-state-quat-hemisphere
--policy-action-clamp --rf-overlap --rf-overlap-sampler
--rf-overlap-prev-weight --rf-overlap-step-shift --policy-image-source
--policy-yubi-render-profile --policy-wrist-backend --policy-wrist-contrast --policy-image-episode
--policy-image-index --policy-image-loop --policy-pose-source
--policy-image-resize --policy-obs-inset --policy-anchor --anchor-table-origin
""".split()

_ROBOT_OPTIONS = """
--robot --franka --physics --fr3-controller --fr3-nullspace
--fr3-joint-kp --fr3-joint-kd --fr3-impedance
--joint-ik-max-velocity --joint-ik-max-acceleration
--joint-ik-max-jerk --joint-ik-nullspace-gain
--joint-ik-ready-position-tol-mm --joint-ik-ready-orientation-tol-deg
--initial-arm-joints-json --initial-arm-joints-tol-deg --joint-ik-fail-hold
--joint-ik-desk-collision-clearance-mm --abs-frame --world-offset --jump-deg
--ik-jump-reject --ik-jump-retries
""".split()

_GRIPPER_OPTIONS = """
--gripper-xml --gripper-xml-left --gripper-calibration --grip-ctrl
--dynamic-grip-cbpc --grip-cfg --torque-scale --grip-torque-mult
--grip-friction --rubber-friction --solref --grip-priority --rubber-solref
--rubber-solimp --rubber-priority --grasp-contact-tol-mm --grasp-mode
--rule-grasp-hand --rule-grasp-effort-eps --rule-grasp-policy-eps-rad
--rule-grasp-release-eps-rad
""".split()

_TIMING_OPTIONS = """
--dynamic-control-hz --dynamic-ready-sec --frames --exec
--dynamic-control-repeats --substeps --dynamic-subframes --mujoco-integrator --mujoco-timestep
--frame-hz --policy-action-hz --max-wall-sec --success-exit-grace-frames
""".split()

_SCENE_OPTIONS = """
--gs --gs-background --gs-background-index --gs-transform
--gs-calibration-world-offset --gs-rot --gs-port --gs-server-token
--gs-render-timeout-sec --gs-render-attempts --gs-render-retry-delay-sec
--gs-composite-mode --gs-kb4-near-cull --gs-color-gain-rgb --gs-color-gamma-rgb --gs-pivot
--gs-pivz --desk --desk-tex --desk-world-aligned --base-placement --table-z
""".split()

_OBJECT_OPTIONS = """
--dynamic-object-settle-max-sec --dynamic-object-settle-min-sec
--dynamic-object-settle-stable-sec --dynamic-object-settle-linear-speed-mm-s
--dynamic-object-settle-angular-speed-rad-s
--dynamic-object-settle-linear-acceleration-m-s2
--dynamic-object-settle-angular-acceleration-rad-s2
--dynamic-object-settle-position-drift-mm
--dynamic-object-settle-orientation-drift-deg --objects
--box-pack-task-index
--box-pack-layout-json
--box-pack-layout-index
--cup-mass --replay-object-source
--init-state-in --init-state-out --init-state-only --mimicgen-init-episode
--mimicgen-init-state-log-root --mimicgen-init-position-tol-mm
--mimicgen-init-orientation-tol-deg
""".split()

_RENDER_OPTIONS = """
--real-video --out --no-wrist --clean-video --no-render
--defer-overview-render --render-lock --black-background --show-origin
--no-origin --wrist-gs --no-obs-gs --obs-gs-only --logdir --summary-path --view-azimuth
--view-elevation --view-distance --view-lookat
""".split()

_SUCCESS_OPTIONS = """
--start-state-guard --start-state-penetration-tol-mm
--goal-condition --score-condition --success-contact-tol-mm
--success-contact-penetration-tol-mm
--success-place-contact-penetration-tol-mm --success-plane-gap-tol-mm
--success-plane-penetration-tol-mm --success-bottom-contact-tol-mm
--success-footprint-slack-mm
--cup-above-plate-test
--success-upright-tol-deg --success-mode
--success-stable-sec --success-linear-speed-mm-s
--success-angular-speed-rad-s
""".split()

_DEBUG_OPTIONS = """
--dump-obs --debug-grip-actuator --debug-grasp-contact --policy-freeze-left
--anchor-dump --deterministic-seed --state-dump-dir --state-dump-frames
--state-dump-on-rule-grasp --state-dump-on-success --help-all
""".split()

_GRASP_SCRIPT_OPTIONS = """
--grasp-hand --grasp-span --grasp-pose-json --grasp-z --pregrasp-z --lift-h --grasp-dx --grasp-dy
--grasp-yaw --approach --approach-azim --side-standoff --grasp-reach
--close-grip --open-grip --ph-settle --ph-descend --ph-close --ph-lift
--ph-hold --ph-open --ph-drop
""".split()

_GROUP_OPTIONS = dict(
    zip(
        _GROUP_TITLES,
        (
            _OFFICIAL_OPTIONS,
            _SOURCE_OPTIONS,
            _ROBOT_OPTIONS,
            _GRIPPER_OPTIONS,
            _TIMING_OPTIONS,
            _SCENE_OPTIONS,
            _OBJECT_OPTIONS,
            _RENDER_OPTIONS,
            _SUCCESS_OPTIONS,
            _DEBUG_OPTIONS,
            _GRASP_SCRIPT_OPTIONS,
        ),
        strict=True,
    )
)
_OPTION_GROUP = {
    option: title
    for title, options in _GROUP_OPTIONS.items()
    for option in options
}
if len(_OPTION_GROUP) != sum(len(options) for options in _GROUP_OPTIONS.values()):
    raise RuntimeError("rollout CLI option is assigned to more than one help group")

_HIDDEN_DEFAULT_OPTIONS = frozenset(
    """
    --franka --grip-torque-mult --grip-friction --rubber-friction --solref
    --grip-priority --rubber-solref --rubber-solimp --rubber-priority
    --grasp-hand --grasp-span --grasp-pose-json --grasp-z --pregrasp-z --lift-h --grasp-dx --grasp-dy
    --grasp-yaw --approach --approach-azim --side-standoff --grasp-reach
    --close-grip --open-grip --ph-settle --ph-descend --ph-close --ph-lift
    --ph-hold --ph-open --ph-drop
    """.split()
)


class _HelpAllAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: V107  # argparse.Action.__call__ signature
        parser.print_help_all()
        parser.exit()


class _RolloutArgumentParser(argparse.ArgumentParser):
    """Route unchanged argument declarations into stable help groups."""

    def configure_rollout_groups(self):
        self._rollout_groups = {}
        for title in _GROUP_TITLES:
            self._rollout_groups[title] = super().add_argument_group(title)
        self._hidden_help = []

    def rollout_group(self, title):
        return self._rollout_groups[title]

    def add_argument(self, *args, **kwargs):
        if not hasattr(self, "_rollout_groups"):
            return super().add_argument(*args, **kwargs)
        option = next(
            (
                value
                for value in args
                if isinstance(value, str) and value.startswith("--")
            ),
            None,
        )
        title = _OPTION_GROUP.get(option)
        if title is None:
            raise RuntimeError(f"rollout CLI option has no help group: {option!r}")
        original_help = kwargs.get("help")
        if option in _HIDDEN_DEFAULT_OPTIONS:
            kwargs["help"] = argparse.SUPPRESS
        action = self._rollout_groups[title].add_argument(*args, **kwargs)
        if option in _HIDDEN_DEFAULT_OPTIONS:
            self._hidden_help.append((action, original_help))
        return action

    def print_help_all(self, file=None):
        for action, help_text in self._hidden_help:
            action.help = help_text  # noqa: V101  # argparse print_help reads it
        try:
            self.print_help(file=file)
        finally:
            for action, _help_text in self._hidden_help:
                action.help = argparse.SUPPRESS  # noqa: V101  # argparse print_help reads it

# ---- Repository-local robot defaults; 3DGS scenes are external. ----
#   YUBI mesh : local generated gripper assets under robosuite_env/assets/
#               assets/{yubi_right,yubi_left}/  (relative refs from the gripper MJCFs)
DEFAULT_GRIPPER_XML = os.path.join(_DIR, "assets", "yubi_right.xml")
DEFAULT_GRIPPER_XML_LEFT = os.path.join(_DIR, "assets", "yubi_left.xml")
DEFAULT_GRIPPER_CALIBRATION = os.path.join(
    _DIR, "assets", "gripper_calib_rad.json"
)
DEFAULT_GS_PLY = os.environ.get("REAL2SIM_GS_PLY", "")
GS_BACKGROUND_INDEX = os.path.join(_DIR, "assets", "gs_backgrounds", "index.json")
LEGACY_GS_ALIGN = os.path.join(_DIR, "obj_calib", "gs_w2gs.npy")

# ---------------- args ----------------


_WRIST_OPTION_DEFAULTS = (
    ("policy_yubi_render_profile", DEFAULT_POLICY_WRIST_RENDER_PROFILE),
    ("policy_wrist_backend", "direct_kb4"),
    ("gs_kb4_near_cull", DEFAULT_KB4_NEAR_CULL),
)


def parse(argv=None):
    p = _RolloutArgumentParser(
        description="FR3 robosuite-env sim (replay/policy, GS wrist)."
    )
    p.configure_rollout_groups()
    p.add_argument(
        "--strict-assets",
        action="store_true",
        help=(
            "require the repository asset manifest and a schema-v2 3DGS scene "
            "contract before constructing MuJoCo. The schema-v2 launcher sets "
            "this for official rollouts; direct runs without it are diagnostic."
        ),
    )
    p.add_argument(
        "--official-run-manifest",
        default=os.environ.get("REAL2SIM_OFFICIAL_RUN_MANIFEST", ""),
        help=(
            "launcher-written run_manifest.json required by --strict-assets; "
            "direct simulations without it are diagnostic"
        ),
    )
    p.add_argument(
        "--official-scene-tag",
        default="",
        help=(
            "exact scene key in the launcher run manifest; required by "
            "--strict-assets so one worker cannot substitute another scene "
            "from the same comparison"
        ),
    )
    p.add_argument(
        "--official-worker-contract",
        default="",
        help=(
            "batch-runner declaration of this process's exact argv; required "
            "with --strict-assets and forbidden for diagnostic runs"
        ),
    )
    p.add_argument("--robot", default="fr3", type=normalize_robot_name,
                   choices=list(SUPPORTED_ROBOT_NAMES),
                   help="robot embodiment: fr3 = two independent FR3 arms, "
                        "fr3_duo = two FR3s on the shared Duo mount. The accepted set is "
                        "robosuite_env.robots.profiles.SUPPORTED_ROBOT_NAMES; "
                        "do not restate it here.")
    p.add_argument("--franka", action="store_true", help="compatibility alias for --robot fr3")
    p.add_argument("--physics", default="kinematic", choices=["kinematic", "dynamic"],
                   help="kinematic = solve IK and FORCE qpos (no feedback) + log penetration; "
                        "dynamic = embodiment controller with full MuJoCo dynamics "
                        "(FR3 selectable controller).")
    p.add_argument("--fr3-controller", choices=list(FR3_CONTROLLERS), default="joint_ik",
                   help="FR3 arm controller: Cartesian impedance torques, or absolute whole-pose IK goals "
                        "with a jerk-limited joint trajectory in dynamic mode and direct "
                        "joint pinning in kinematic mode.")
    p.add_argument("--fr3-impedance", default="{}",
                   help="JSON object of physical Cartesian impedance gains and target/torque limits.")
    p.add_argument(
        "--fr3-nullspace",
        choices=["home", "off"],
        default="home",
        help=(
            "joint_ik null-space posture mode. home keeps the historical "
            "canonical-home posture objective weighted by "
            "--joint-ik-nullspace-gain; off solves the pure whole-pose IK "
            "(effective gain 0)."
        ),
    )
    p.add_argument("--dynamic-control-hz", type=int, default=20,
                   help="dynamic mode inner robosuite controller rate. Explicit FR3 policy cadence "
                        "belongs in --policy-action-hz; non-divisor rates are phase-scheduled without "
                        "changing this controller clock.")
    p.add_argument("--dynamic-ready-sec", type=float, default=12.0,
                   help="dynamic arm-only ready-drive duration in physical seconds. It is converted to "
                        "controller steps so changing the inner control rate cannot shorten initialization.")
    p.add_argument(
        "--fr3-joint-kp",
        type=float,
        default=None,
        help=(
            "optional FR3 JOINT_POSITION proportional gain. Unset keeps the "
            "repo-local controller JSON value."
        ),
    )
    p.add_argument(
        "--fr3-joint-kd",
        type=float,
        default=None,
        help=(
            "optional direct FR3 JOINT_POSITION damping gain. Unset keeps "
            "robosuite's damping_ratio-derived value."
        ),
    )
    p.add_argument("--dynamic-object-settle-max-sec", type=float, default=0.0,
                   help="dynamic joint_ik mode only: after the arm ready drive, restore the raw cup / plate pose and "
                        "let both objects settle under gravity for at most this many physical seconds. "
                        "A consecutive velocity / acceleration / support-contact gate must pass before policy "
                        "inference. 0 preserves legacy no-settle initialization.")
    p.add_argument("--dynamic-object-settle-min-sec", type=float, default=0.5,
                   help="minimum physical object-settle duration before the quiet gate may pass.")
    p.add_argument("--dynamic-object-settle-stable-sec", type=float, default=0.2,
                   help="consecutive quiet/support-contact window required for object settle.")
    p.add_argument("--dynamic-object-settle-linear-speed-mm-s", type=float, default=1.0,
                   help="maximum free-object linear speed accepted by the settle gate.")
    p.add_argument("--dynamic-object-settle-angular-speed-rad-s", type=float, default=0.03,
                   help="maximum free-object angular speed accepted by the settle gate.")
    p.add_argument("--dynamic-object-settle-linear-acceleration-m-s2", type=float, default=0.1,
                   help="instantaneous free-object linear acceleration telemetry reference; contact-solver qacc is not an acceptance gate.")
    p.add_argument("--dynamic-object-settle-angular-acceleration-rad-s2", type=float, default=1.0,
                   help="instantaneous free-object angular acceleration telemetry reference; contact-solver qacc is not an acceptance gate.")
    p.add_argument("--dynamic-object-settle-position-drift-mm", type=float, default=0.25,
                   help="maximum free-object position drift over the consecutive quiet window.")
    p.add_argument("--dynamic-object-settle-orientation-drift-deg", type=float, default=0.1,
                   help="maximum free-object orientation drift over the consecutive quiet window.")
    p.add_argument("--joint-ik-max-velocity", type=float, default=1.0,
                   help="joint_ik online trajectory velocity limit in rad/s.")
    p.add_argument("--joint-ik-max-acceleration", type=float, default=20.0,
                   help="joint_ik online trajectory acceleration limit in rad/s^2.")
    p.add_argument("--joint-ik-max-jerk", type=float, default=200.0,
                   help="joint_ik stateful online trajectory jerk limit in rad/s^3.")
    p.add_argument(
        "--joint-ik-nullspace-gain",
        type=float,
        default=0.2,
        help=(
            "FR3 joint_ik gain for the projected canonical-home posture "
            "objective. 0 disables null-space posture selection."
        ),
    )
    p.add_argument("--joint-ik-ready-position-tol-mm", type=float, default=2.0,
                   help="joint_ik fail-fast TCP position tolerance after the dynamic ready drive.")
    p.add_argument("--joint-ik-ready-orientation-tol-deg", type=float, default=2.0,
                   help="joint_ik fail-fast TCP orientation tolerance after the dynamic ready drive.")
    p.add_argument(
        "--initial-arm-joints-json",
        default="",
        metavar="JSON",
        help=(
            "optional strict JSON containing left_arm/right_arm j1..j7 in radians. "
            "In dynamic joint_ik mode these vectors replace only the rollout ready "
            "joint goals; the existing ready duration, hold, object settle, and policy "
            "configuration remain unchanged. schema_version 2 may additionally "
            "declare a gripper block (raw motor radians, hand_left_motor_joint / "
            "hand_right_motor_joint); it replaces only the ready-drive gripper "
            "command and cannot be combined with --mimicgen-init-episode."
        ),
    )
    p.add_argument(
        "--initial-arm-joints-tol-deg",
        type=float,
        default=0.5,
        help=(
            "maximum requested-vs-actual per-joint error accepted after a JSON "
            "ready drive."
        ),
    )
    p.add_argument(
        "--joint-ik-fail-hold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "reject non-converged absolute IK results and continue the previous "
            "validated jerk-limited target instead of adopting the best iterate."
        ),
    )
    p.add_argument(
        "--joint-ik-desk-collision-clearance-mm",
        type=float,
        default=0.0,
        help=(
            "minimum arm-link / desk clearance for absolute IK targets. "
            "Positive values enable scratch-state path clipping; 0 disables it. "
            "Gripper/object contacts are never checked by this gate."
        ),
    )
    p.add_argument("--gripper-xml", default="", metavar="PATH",
                   help="legacy option retained for old command compatibility; FR3 uses YubiGripper.")
    p.add_argument("--gripper-xml-left", default="", metavar="PATH",
                   help="legacy option retained for old command compatibility; FR3 uses YubiGripper.")
    p.add_argument(
        "--gripper-calibration",
        default=(
            os.environ.get("YUBI_GRIPPER_CALIBRATION", "").strip()
            or DEFAULT_GRIPPER_CALIBRATION
        ),
        metavar="JSON",
        help=(
            "validated bilateral YUBI zero-angle calibration JSON (defaults to "
            "the repository asset). left_offset_rad and right_offset_rad are "
            "applied to the respective finger body frames without changing "
            "policy/action/observation coordinates."
        ),
    )
    src = p.rollout_group("Source").add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", help="path to a recorded episode parquet (replay its deltas).")
    src.add_argument("--policy", help="path to a policy checkpoint dir (run inference).")
    src.add_argument("--policy-batch-server",
                     help="host:port of robosuite_env.policy.policy_batch_server. This runs policy rollout "
                          "without loading the checkpoint in this sim process; observations are sent to "
                          "the shared batch server at each --exec chunk boundary.")
    src.add_argument("--ik-replay", help="path to an added_ik parquet; drive FR3 DIRECTLY with its STORED IK joints "
                     "(observation.joint_states = [grip(2), R fr3 joint1-7, L fr3 joint1-7]); no online IK solving.")
    src.add_argument("--state-log-replay", help="path to sim_log.jsonl; render logged arm qpos, cup position, and gripper actual angles without rerunning policy.")
    p.add_argument(
        "--state-replay-policy-obs-dir",
        default="",
        metavar="DIR",
        help=(
            "state-log replay only: use the exact saved 224x224 policy wrist "
            "PNGs in DIR as the source for held video insets, avoiding new "
            "wrist EGL / 3DGS renders. The display mode controls whether the "
            "full policy tensor or its metadata-derived unpadded content is "
            "shown; every state row must reference a "
            "complete, side- and frame-consistent primary_L/R pair"
        ),
    )
    p.add_argument(
        "--state-replay-policy-obs-display",
        choices=("aspect-fit", "raw224"),
        default="aspect-fit",
        help=(
            "state-log replay saved-OBS display: aspect-fit removes the "
            "recorded OpenPI letterbox for presentation; raw224 preserves the "
            "entire 224x224 policy tensor, including its black letterbox, when "
            "resizing it into the video inset"
        ),
    )
    src.add_argument("--grasp-test", action="store_true",
                     help="scripted grasp->close->lift->open->drop probe (no policy/replay). "
                          "Use --physics dynamic to test the selected robosuite controller path.")
    p.add_argument("--instruction", default="Pick up the cup with your right hand and set it on the plate",
                   help="language prompt for the policy.")
    p.add_argument(
        "--config",
        default="",
        help="OpenPI config name matching --policy; required with --policy.",
    )
    p.add_argument("--config-src", default="deploy", choices=["deploy","training"], help="config source (--policy).")
    p.add_argument("--ah", type=int, default=0, help="override model action_horizon (0=keep config). ah32 ckpts (152000 family) need --ah 32 with the deploy config.")
    p.add_argument("--policy-batch-client-id", default=os.environ.get("POLICY_BATCH_CLIENT_ID", ""),
                   help="stable client id used in policy batch server logs. Empty derives one from host/pid/logdir.")
    p.add_argument("--policy-batch-timeout-sec", type=float, default=float(os.environ.get("POLICY_BATCH_TIMEOUT_SEC", "0")),
                   help="socket timeout for --policy-batch-server. Official runs set a finite timeout from runner.max_wall_sec; 0 is a blocking direct-run diagnostic.")
    p.add_argument("--gs", default=DEFAULT_GS_PLY, help="background 3DGS .ply path (started/served + composited into wrist view). "
                   "Defaults to REAL2SIM_GS_PLY or disabled.")
    p.add_argument("--gs-background", "--gs-bg", dest="gs_background", default=None,
                   help="3DGS background preset name, preset directory, JSON definition file, or direct .ply. "
                        "Presets are read from --gs-background-index when present. Resolves ply, world->GS matrix, "
                        "rotation, pivot, and port. Pass '', none, or off to disable. --gs remains a "
                        "legacy direct .ply override.")
    p.add_argument("--gs-background-index", default=GS_BACKGROUND_INDEX,
                   help="JSON index of named 3DGS background presets.")
    p.add_argument("--gs-transform", default=None,
                   help="world(MuJoCo)->GS-native .npy alignment matrix. Overrides preset alignment. "
                        "Use none/off to disable alignment.")
    p.add_argument(
        "--gs-calibration-world-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help=(
            "translation added to active MuJoCo world positions before the "
            "world->GS matrix. Use this when GS calibration used a different "
            "world origin (for example 0 0 0.8 when calibration table z=0.8 "
            "but the active registered desk is normalized to table z=0)."
        ),
    )
    p.add_argument("--gs-rot", default="", help="GS frame rotation e.g. x90 (about table center).")
    p.add_argument("--gs-port", type=int, default=8097)
    p.add_argument(
        "--gs-server-token",
        default="",
        help="per-launch identity token required by the selected 3DGS server",
    )
    p.add_argument("--gs-render-timeout-sec", type=float,
                   default=float(os.environ.get("GS_RENDER_TIMEOUT_SEC", "30")),
                   help="timeout for each 3DGS render request. The default tolerates queued requests from batched rollouts.")
    p.add_argument("--gs-render-attempts", type=int,
                   default=int(os.environ.get("GS_RENDER_ATTEMPTS", "3")),
                   help="attempts per 3DGS render. Exhaustion is fatal so a missing background cannot pass silently.")
    p.add_argument("--gs-render-retry-delay-sec", type=float,
                   default=float(os.environ.get("GS_RENDER_RETRY_DELAY_SEC", "0.25")),
                   help="delay between failed 3DGS render attempts.")
    p.add_argument(
        "--gs-composite-mode",
        choices=["depth", "foreground"],
        default="depth",
        help=(
            "depth z-tests MuJoCo against GS geometry; foreground always "
            "pastes visible MuJoCo geometry over GS. A background.json may "
            "override this when the GS still contains task-object ghosts."
        ),
    )
    p.add_argument(
        "--gs-kb4-near-cull",
        choices=KB4_NEAR_CULL_MODES,
        default=None,  # parse() fills znear; a state-log replay takes the run's (recorded_wrist)
        help=(
            "near cull of the direct_kb4 wrist 3DGS render: znear drops only Gaussians closer than the "
            "GS client's znear (0.01 m, behind the wrist camera's 10 mm setback); legacy drops those "
            "within 0.2 GS units (0.2 m x gs->world scale), as every run recorded before this option did"
        ),
    )
    p.add_argument(
        "--gs-color-gain-rgb",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("R", "G", "B"),
        help=(
            "black-preserving sRGB gains applied only to the rendered 3DGS "
            "background before MuJoCo foreground compositing; default is unchanged"
        ),
    )
    p.add_argument(
        "--gs-color-gamma-rgb",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("R", "G", "B"),
        help=(
            "positive sRGB gamma exponents applied only to the rendered 3DGS "
            "background before --gs-color-gain-rgb; default is unchanged"
        ),
    )
    p.add_argument("--gs-pivot", type=float, nargs=3, default=None, metavar=("X","Y","Z"),
                   help="GS rotation pivot used with --gs-rot. Overrides preset pivot and --gs-pivz.")
    p.add_argument("--gs-pivz", type=float, default=0.95, help="GS rotation pivot z (table top in robosuite world).")
    p.add_argument("--desk", default="scene",
                   help="registered desk mesh .obj, or 'scene' for a validated schema-v2 scene contract (default: scene; the legacy 'robosuite' TableArena environment was removed).")
    p.add_argument("--desk-tex", default="", help="desk texture png (defaults next to --desk).")
    p.add_argument(
        "--desk-world-aligned",
        action="store_true",
        help=(
            "treat --desk and its sibling coacd/*.obj as already baked in "
            "MuJoCo world meters; disables legacy desk width/z normalization"
        ),
    )
    p.add_argument("--objects", default="cup_plate",
                   help="objects to place: 'cup_plate', "
                        "'box_pack', "
                        "or a positions json with cup_xy/plate_xy.")
    # These do not default from the environment:
    # the flag carries only the profile's choice, so a conflicting harness
    # RS_*_LAYOUT_JSON override is refused instead of silently taking over
    # (robosuite_env/tasks/layout_selection.py).
    for _layout_option, _layout_task in (
        ("--box-pack-layout-json", "box_pack"),
    ):
        p.add_argument(
            _layout_option,
            default="",
            help=(
                f"{_layout_task}: hand-authored layout to place the objects from "
                "(simulation.parameters."
                f"{_layout_option[2:].replace('-', '_')}). Empty uses the task "
                "default; the layout must be placed for this run's scene."
            ),
        )
    p.add_argument(
        "--box-pack-layout-index",
        type=int,
        default=None,
        help=(
            "zero-based provenance index of --box-pack-layout-json in the "
            "profile's layout list; set by the batch runner"
        ),
    )
    p.add_argument(
        "--box-pack-task-index",
        type=int,
        choices=range(12),
        default=int(os.environ.get("BOX_PACK_TASK_INDEX", "0")),
        help=(
            "box_pack: which of the twelve prompts to run. The index fixes the "
            "instruction, the target object and the hand together; "
            "--instruction must match it exactly."
        ),
    )
    p.add_argument("--frames", type=int, default=0, help="cap number of source frames (0 = all).")
    p.add_argument("--exec", "--adopted-action-chunks", dest="exec", type=int, default=16,
                   help="policy: how many chunk steps to EXECUTE before re-inferring = deploy config "
                        "'adopted_action_chunks' (16 in the Franka YUBI deployment, the default here).")
    p.add_argument("--policy-delta-anchor", choices=["observation", "live"],
                   default=os.environ.get("POLICY_DELTA_ANCHOR", "observation"),
                   help="policy arm target integration: observation integrates the complete relative-action "
                        "chunk from its observation-time TCP pose so tracking residual is retained; live is the "
                        "legacy behavior that rebases every delta on the current TCP and can discard residual.")
    p.add_argument(
        "--policy-state-quat-hemisphere",
        choices=["positive", "negative"],
        default="positive",
        help=(
            "sign of the left-to-right hand quaternion in the policy state "
            "(simulation.parameters.policy_state_quat_hemisphere, resolved by the "
            "launcher from the checkpoint norm stats). positive: the historical "
            "fixed qw >= 0 rule. negative: qw <= 0 at the rollout's first "
            "observation, then sign continuity between observations. The worker "
            "refuses a value that disagrees with the policy server's norm stats."
        ),
    )
    p.add_argument("--policy-pose-repr", choices=["delta", "anchored"],
                   default=os.environ.get("POLICY_POSE_REPR", "delta"),
                   help="policy arm pose output representation: delta = per-step local SE(3) deltas "
                        "(current behavior); anchored = every row is a waypoint relative to the single "
                        "observation-time pose and is composed with that fixed anchor independently.")
    p.add_argument("--policy-action-clamp", action=argparse.BooleanOptionalAction, default=True,
                   help="FR3 policy only: clamp relative translation/rotation to deploy limits; "
                        "use --no-policy-action-clamp for raw policy deltas.")
    p.add_argument("--dynamic-control-repeats", type=int, default=0,
                   help="dynamic mode only: env.step calls per source action. 0 derives the count from "
                        "--policy-action-hz / --frame-hz and --dynamic-control-hz, or uses one "
                        "controller interval when both rates are off. Policy --exec never changes "
                        "physical hold time.")
    p.add_argument("--rf-overlap", "--action-overlap", dest="rf_overlap", type=int, default=0,
                   help="lagged RF action overlap (0=legacy). The single sampler conditions the next "
                        "trajectory prefix and executes [overlap:overlap+exec].")
    p.add_argument("--rf-overlap-sampler", choices=["single"], default="single",
                   help="RF overlap sampler. single is the only supported value: it keeps the existing "
                        "one-OBS RF prefix inpainting.")
    p.add_argument("--rf-overlap-prev-weight", type=float, default=None,
                   help="Paired RF overlap samplers were removed; this compatibility option must remain unset (null).")
    p.add_argument("--rf-overlap-step-shift", type=int, default=0,
                   help="Paired RF overlap samplers were removed; this compatibility option must remain fixed at 0.")
    p.add_argument("--substeps", type=int, default=int(os.environ.get("SUBSTEPS", "10")),
                   help="MuJoCo physics substeps per source frame for kinematic+CbPC. env SUBSTEPS overrides default 10.")
    p.add_argument("--dynamic-subframes", type=int, default=int(os.environ.get("DYNAMIC_SUBFRAMES", "1")),
                   help="dynamic mode only: render/export this many intermediate env.step states per policy frame. "
                        "1 keeps legacy behavior; e.g. 4 samples the dynamic control repeats four times so object motion "
                        "does not appear to jump between rendered frames.")
    p.add_argument("--mujoco-timestep", type=float, default=None,
                   help="Explicit physics timestep in seconds; must divide the control period.")
    p.add_argument("--mujoco-integrator", choices=["keep", "euler", "rk4", "implicit", "implicitfast"],
                   default="keep",
                   help="MuJoCo integrator override for diagnostics. keep uses the model default; an "
                        "explicit different choice is recorded as a non-parity diagnostic override.")
    p.add_argument("--frame-hz", type=float, default=0.0,
                   help="source policy/action and output-video rate. In dynamic mode this also derives "
                        "control repeats when --dynamic-control-repeats=0 (0=off).")
    p.add_argument(
        "--policy-action-hz",
        type=float,
        default=0.0,
        help=(
            "FR3 dynamic-policy source cadence. A positive value phase-schedules "
            "policy rows onto --dynamic-control-hz without changing the controller "
            "clock (for example 30 Hz onto 50 Hz uses 2,1,2 ticks). It is mutually "
            "exclusive with --frame-hz and --dynamic-control-repeats, and playback "
            "uses the nominal source cadence. The total source duration is preserved; "
            "individual action holds are quantized to complete controller ticks. "
            "0 keeps the legacy --frame-hz behavior."
        ),
    )
    p.add_argument("--max-wall-sec", type=float, default=0.0,
                   help="wall-clock budget (seconds) for the rollout LOOP only (excludes one-time policy load). "
                        "0=off. When exceeded, stop after the current frame and finalize video/logs with the frames "
                        "done so far.")
    p.add_argument("--success-exit-grace-frames", type=int, default=0,
                   help="DISABLED BY DEFAULT (0) and it must stay that way until the policy batch server can "
                        "retire a client. The batch server waits for exactly policy_batch_size requests, so a "
                        "worker that exits early leaves its wave blocked until the socket timeout, which is "
                        "max_wall_sec -- hours. Official completion also requires every run to have "
                        "exactly runner.frames frames, and the grid renderer refuses runs of unequal length. "
                        "When set: stop the rollout this many frames after the success criterion first reports "
                        "success. "
                        "0=off, run the full frame budget. The grace frames are simulated and logged like any "
                        "other, so the criterion is still evaluated over them and a success that does not hold "
                        "is visible.")
    p.add_argument("--dump-obs", default="", metavar="DIR",
                   help="policy only: save the exact 224px wrist fisheye images the policy consumes (observation.image.left/right) "
                        "to DIR at every re-inference (per chunk). deterministic, works with --no-render.")
    p.add_argument("--policy-image-source", choices=["sim", "lerobot"], default="sim",
                   help="policy only: source for observation.image.left/right. sim = render current sim wrist fisheye; "
                        "lerobot = feed real LeRobot wrist video frames while keeping sim pose/joint state/initial state.")
    p.add_argument(
        "--policy-yubi-render-profile",
        choices=WRIST_RENDER_PROFILE_NAMES,
        default=None,  # parse() fills the default; a state-log replay takes the run's (recorded_wrist)
        help=(
            "versioned wrist-render contract of the sim policy and video "
            "wrist images, defined in robosuite_env/rendering/wrist_render_profiles.py"
        ),
    )
    p.add_argument(
        "--policy-wrist-backend",
        choices=WRIST_BACKENDS,
        default=None,  # parse() fills direct_kb4; a state-log replay takes the run's (recorded_wrist)
        help="wrist projection backend for policy and video; direct_kb4 needs a GS server with the GaussianWrapping KB4 rasterizer",
    )
    p.add_argument(
        "--policy-wrist-contrast",
        type=float,
        default=1.0,
        help=(
            "policy-image-source=sim only: RGB contrast applied to both left "
            "and right rendered wrist images around mid-gray 127.5 before "
            "--policy-image-resize, so pad-mode black bands remain RGB 0; "
            "1.0 is unchanged"
        ),
    )
    p.add_argument("--policy-image-episode", default="", metavar="PARQUET",
                   help="policy-image-source=lerobot: episode parquet used to resolve videos/<chunk>/observation.image.left|right/<episode>.mp4. "
                        "Empty default uses the deploy anchor episode.")
    p.add_argument("--policy-image-index", choices=["infer", "frame"], default="infer",
                   help="policy-image-source=lerobot frame selection. infer = advance one video frame per policy inference; "
                        "frame = use the current sim frame index. Out-of-range indices clamp unless --policy-image-loop is set.")
    p.add_argument("--policy-image-loop", action="store_true",
                   help="policy-image-source=lerobot: wrap LeRobot video indices instead of clamping at the last frame.")
    p.add_argument("--policy-pose-source", choices=["sim", "lerobot"], default="sim",
                   help="policy only: source for observation.pose.left_hand_root_to_right_hand_root.absolute and "
                        "observation.joint_states. sim = compute from current sim state; lerobot = feed the real LeRobot "
                        "per-frame values (teacher-forcing obs; combine with --policy-image-source lerobot to reproduce the "
                        "demo trajectory). Uses --policy-image-episode/--policy-image-index/--policy-image-loop for the frame source.")
    p.add_argument("--policy-image-resize", choices=["stretch", "pad"], default="stretch",
                   help="policy only: how the wrist obs image is downsized to 224. stretch = cv.resize (aspect-distorting, "
                        "legacy default). pad = openpi resize_with_pad (aspect-preserving letterbox) to MATCH training's "
                        "ResizeImages(resize_with_pad). Non-square sources (real 640x480 LeRobot wrist) are geometrically "
                        "distorted by stretch; use pad for faithful LeRobot-OBS teacher forcing.")
    p.add_argument("--policy-obs-inset", action="store_true",
                   help="policy only: use the actual saved policy-input wrist image for each chunk's held video inset. "
                        "The inference tensor is unchanged; display removes deterministic letterbox padding and preserves "
                        "the declared source aspect ratio. Already implicit for --policy-image-source lerobot; this extends "
                        "it to sim-rendered observations.")
    p.add_argument("--grip-ctrl", choices=["cbpc", "kinematic"], default=os.environ.get("GRIP_CTRL", "cbpc"),
                   help="finger controller selection. cbpc = historical current-based position control; "
                        "kinematic = legacy direct qpos/position actuator target.")
    p.add_argument("--dynamic-grip-cbpc", action="store_true",
                   help="dynamic mode: disable the built-in finger position actuators and apply CbPC torque "
                        "through qfrc_applied before every 2 ms MuJoCo step.")
    p.add_argument("--grip-cfg", default=os.environ.get("YUBI_CTRL", os.path.join(_DIR, "robots", "yubi", "control", "configs", "spec_nominal.yaml")),
                   help="YUBI CbPC config yaml. env YUBI_CTRL overrides.")
    p.add_argument("--torque-scale", type=float, default=float(os.environ.get("TORQUE_SCALE", "1.0")),
                   help="CbPC torque_scale override. env TORQUE_SCALE overrides.")
    p.add_argument("--grip-torque-mult", type=float, default=float(os.environ.get("GRIP_TORQUE_MULT", "0.5")),
                   help="dynamic mode: multiply YUBI finger position actuator force ranges. "
                        "0.5 halves XML values and is the current rollout default; "
                        "1 keeps XML values. env GRIP_TORQUE_MULT overrides.")
    p.add_argument("--grip-friction", default=os.environ.get("GRIP_FRICTION", "keep"),
                   help="YUBI generic finger collision friction override as 'slide spin roll', or 'keep'. "
                        "XML default is typically '0.5 0.05 0.0001'. env GRIP_FRICTION overrides.")
    p.add_argument("--rubber-friction", default=os.environ.get("RUBBER_FRICTION", "keep"),
                   help="YUBI rubber pad collision friction override as 'slide spin roll', or 'keep'. "
                        "XML default is typically '4 0.1 0.0001'. env RUBBER_FRICTION overrides.")
    p.add_argument("--solref", default=os.environ.get("SOLREF", "0.04 1"),
                   help="finger collision solref override, or 'keep'. env SOLREF overrides. default '0.04 1'.")
    p.add_argument("--grip-priority", type=int, default=int(os.environ.get("GRIP_PRIORITY", "1")),
                   help="finger collision priority for the solref override. env GRIP_PRIORITY overrides. default 1.")
    p.add_argument("--rubber-solref", default=os.environ.get("RUBBER_SOLREF", "0.005 1"),
                   help="YUBI rubber pad collision solref override, or 'keep'. Smaller timeconst = stiffer contact. "
                        "Default '0.005 1' keeps the high-friction pad much stiffer than the generic finger hulls.")
    p.add_argument("--rubber-solimp", default=os.environ.get("RUBBER_SOLIMP", "0.99 0.999 0.0001"),
                   help="YUBI rubber pad solimp override, or 'keep'. Default makes pad contact close to hard.")
    p.add_argument("--rubber-priority", type=int, default=int(os.environ.get("RUBBER_PRIORITY", "3")),
                   help="YUBI rubber pad contact priority. Default 3 makes pad settings win over surrounding hulls.")
    p.add_argument("--cup-mass", type=float, default=float(os.environ.get("CUP_MASS", "0.0")),
                   help="override cup body mass in kg (0=keep XML-computed mass, currently about 0.0115kg). "
                        "inertia is scaled consistently. e.g. --cup-mass 0.036 for real weight. env CUP_MASS overrides.")
    p.add_argument("--real-video", default="", help="real episode video (.mp4) shown beside the sim replay.")
    p.add_argument("--out", default=os.path.join(_DIR, "sim_out.mp4"))
    p.add_argument("--no-wrist", action="store_true", help="skip wrist fisheye insets.")
    p.add_argument("--clean-video", action="store_true",
                   help="write only the overview camera frame: no wrist insets, real-video panel, or text overlays.")
    p.add_argument("--no-render", action="store_true",
                   help="skip replay / video rendering; live policy wrist observations are still rendered when required.")
    p.add_argument(
        "--defer-overview-render",
        action="store_true",
        help=(
            "record MuJoCo states during the rollout, close every wrist EGL "
            "renderer, then render the overview in an isolated second pass; "
            "avoids striped, noisy, or black overview frames caused by "
            "interleaved EGL contexts. Automatically enabled when "
            "--policy-batch-server is used"
        ),
    )
    p.add_argument("--render-lock", default=os.environ.get("MUJOCO_RENDER_LOCK", ""), metavar="PATH",
                   help="optional process-shared file lock serializing MuJoCo / wrist renderer creation and rendering. "
                        "Use the same path for parallel rollout workers; MUJOCO_RENDER_LOCK supplies the default.")
    p.add_argument("--black-background", action="store_true",
                   help="render MuJoCo sky/background as black. Intended for no-GS background comparisons.")
    p.add_argument("--show-origin", action="store_true", default=True, help="draw a marker (sphere+axis triad) at world origin (0,0,0).")
    p.add_argument("--no-origin", dest="show_origin", action="store_false")
    p.add_argument("--wrist-gs", action="store_true", help="composite 3DGS into the DISPLAY wrist fisheye insets (policy obs is controlled by --no-obs-gs).")
    obs_gs = p.rollout_group("Render/Output").add_mutually_exclusive_group()
    obs_gs.add_argument("--no-obs-gs", action="store_true", help="disable the 3DGS background in the POLICY-input wrist fisheye (default: composited when --gs is set, to close the obs domain gap vs real images).")
    obs_gs.add_argument(
        "--obs-gs-only",
        action="store_true",
        help=(
            "diagnostic wrist observation mode: keep the normal "
            "fisheye and policy preprocessing geometry, but remove every "
            "MuJoCo foreground pixel from the 3DGS composite"
        ),
    )
    p.add_argument("--logdir", default="", help="directory for IK/penetration logs (sim_log.jsonl, sim_summary.json). default: alongside sim.py.")
    p.add_argument("--debug-grip-actuator", action="store_true",
                   help="include per-frame Yubi finger actuator ctrl/force/range and per-finger qpos/qvel in sim_log.jsonl.")
    p.add_argument("--debug-grasp-contact", action="store_true",
                   help="include per-frame gripper-cup contact counts, distances, and normal forces in sim_log.jsonl.")
    p.add_argument("--grasp-contact-tol-mm", type=float, default=float(os.environ.get("GRASP_CONTACT_TOL_MM", "3.0")),
                   help="max gripper-cup contact distance counted in --debug-grasp-contact summaries.")
    p.add_argument("--grasp-mode", choices=["physics", "rule"], default=os.environ.get("GRASP_MODE", "rule"),
                   help="physics = normal MuJoCo contacts only. rule = attach the cup when both inner pads contact "
                        "the same object and the policy target is below both measured contact angles; clear the "
                        "grasp when either measured finger opens above its saved contact angle.")
    p.add_argument("--rule-grasp-hand", choices=["auto", "R", "L"], default=os.environ.get("RULE_GRASP_HAND", "auto"),
                   help="which hand can create a rule-based grasp. auto checks R then L.")
    p.add_argument("--rule-grasp-effort-eps", type=float, default=float(os.environ.get("RULE_GRASP_EFFORT_EPS", "1e-3")),
                   help="fallback effort deadband when no policy gripper target is available. "
                        "Negative effort closes; positive effort opens.")
    p.add_argument("--rule-grasp-policy-eps-rad", type=float,
                   default=float(os.environ.get("RULE_GRASP_POLICY_EPS_RAD", "1e-3")),
                   help="contact-angle margin for starting rule grasp. The policy target must be below both "
                        "contact angles by this margin.")
    p.add_argument("--rule-grasp-release-eps-rad", type=float,
                   default=float(os.environ.get("RULE_GRASP_RELEASE_EPS_RAD", "5e-2")),
                   help="release margin for active rule grasp. Either actual finger must open above its saved "
                        "contact angle by this amount before the grasp clears.")
    p.add_argument("--summary-path", default=os.environ.get("SIM_SUMMARY_PATH", ""),
                   help="optional explicit sim_summary.json output path. The default LOGDIR/sim_summary.json is still written.")
    p.add_argument("--view-azimuth", type=float, default=float(os.environ.get("VIEW_AZIMUTH", "130")),
                   help="overview free-camera azimuth in degrees for rendered videos.")
    p.add_argument("--view-elevation", type=float, default=float(os.environ.get("VIEW_ELEVATION", "-22")),
                   help="overview free-camera elevation in degrees for rendered videos.")
    p.add_argument("--view-distance", type=float, default=float(os.environ.get("VIEW_DISTANCE", "2.0")),
                   help="overview free-camera distance for rendered videos.")
    p.add_argument("--view-lookat", default=os.environ.get("VIEW_LOOKAT", ""),
                   help="overview free-camera lookat as 'x y z'. Empty uses table center at table_z+0.15.")
    p.add_argument("--goal-condition", "--success-criterion", dest="success_criterion",
                   default=os.environ.get("GOAL_CONDITION", os.environ.get("SUCCESS_CRITERION", "auto")),
                   help="task goal/success condition for per-frame geometric evaluation. "
                        "Named tasks should select their explicit criterion (for example object_in_box); "
                        "auto retains the legacy cup_on_plate mapping. Use none/off to disable.")
    p.add_argument("--base-placement", default=os.environ.get("RS_BASE_PLACEMENT", ""),
                   help="calibration file naming where this rig's mounted bases "
                        "stand, table-top relative. Omitted selects the rig's "
                        "registered default (robosuite_env.robots.base_placement."
                        "DEFAULT_BASE_PLACEMENTS). A file calibrated for another "
                        "rig is refused rather than applied: the flat FR3 "
                        "pedestals and the tilted Duo mount are not "
                        "interchangeable. A launched profile sets this from "
                        "simulation.parameters.base_placement.")
    p.add_argument("--score-condition", default=os.environ.get("SCORE_CONDITION", "auto"),
                   help="per-frame scoring condition. Named criteria interpret auto using their task default; "
                        "use none/off to disable.")
    p.add_argument("--start-state-guard",
                   choices=("off", "warn", "fail"),
                   default=os.environ.get("START_STATE_GUARD", "warn"),
                   help="check the first frame for objects that begin the episode inside "
                        "one another, before any step is taken. warn prints and records "
                        "the overlaps; fail refuses the run. An episode that starts "
                        "interpenetrating is not a rollout of the task -- its first tenth "
                        "of a second is the solver separating a scene nobody placed -- and "
                        "it otherwise completes and scores like any other run.")
    p.add_argument("--start-state-penetration-tol-mm", type=float,
                   default=float(os.environ.get("START_STATE_PENETRATION_TOL_MM", "2.0")),
                   help="how deep a start-of-episode contact has to be before the guard "
                        "calls it an overlap, in mm. Compliant contacts settle a fraction "
                        "of a millimetre in by design, so zero would flag every object "
                        "resting on the table.")
    p.add_argument("--success-contact-tol-mm", type=float, default=float(os.environ.get("SUCCESS_CONTACT_TOL_MM", "3.0")),
                   help="cup_on_plate: max MuJoCo cup-plate contact distance counted as contact, in mm.")
    p.add_argument("--success-contact-penetration-tol-mm", type=float, default=float(os.environ.get("SUCCESS_CONTACT_PENETRATION_TOL_MM", "5.0")),
                   help="cup_on_plate: max allowed negative MuJoCo cup-plate contact distance, in mm.")
    p.add_argument("--success-place-contact-penetration-tol-mm", type=float,
                   default=float(os.environ.get("SUCCESS_PLACE_CONTACT_PENETRATION_TOL_MM", "15.0")),
                   help="cup_on_plate: relaxed max negative cup-plate contact distance for place success, in mm. "
                        "Strict contact metrics still use --success-contact-penetration-tol-mm.")
    p.add_argument("--success-plane-gap-tol-mm", type=float, default=float(os.environ.get("SUCCESS_PLANE_GAP_TOL_MM", "10.0")),
                   help="cup_on_plate: allowed gap above the measured plate collision z-range, in mm.")
    p.add_argument("--success-plane-penetration-tol-mm", type=float, default=float(os.environ.get("SUCCESS_PLANE_PENETRATION_TOL_MM", "2.0")),
                   help="cup_on_plate: allowed downward penetration below the measured plate collision z-range, in mm.")
    p.add_argument("--success-bottom-contact-tol-mm", type=float, default=float(os.environ.get("SUCCESS_BOTTOM_CONTACT_TOL_MM", "12.0")),
                   help="cup_on_plate scoring: contact point height tolerance for classifying cup bottom contact, in mm.")
    p.add_argument("--success-footprint-slack-mm", type=float, default=float(os.environ.get("SUCCESS_FOOTPRINT_SLACK_MM", "3.0")),
                   help="cup_on_plate: radial slack for checking cup center inside the measured plate disc, in mm.")
    p.add_argument("--cup-above-plate-test", choices=("middle", "lowest"), default="middle",
                   help="cup_on_plate: which point of the cup must be over the plate -- its middle (Duo scoring) or its lowest collision point (Flat's stricter test).")
    p.add_argument("--success-upright-tol-deg", type=float, default=float(os.environ.get("SUCCESS_UPRIGHT_TOL_DEG", "35.0")),
                   help="cup_on_plate: maximum allowed tilt from the support normal.")
    p.add_argument(
        "--success-stable-sec",
        type=float,
        default=float(os.environ.get("SUCCESS_STABLE_SEC", "2.0")),
        help=(
            "box packing: consecutive control-rate "
            "seconds for which the task's success geometry must hold."
        ),
    )
    p.add_argument(
        "--success-linear-speed-mm-s",
        type=float,
        default=float(os.environ.get("SUCCESS_LINEAR_SPEED_MM_S", "1.0")),
        help=(
            "box packing: maximum linear speed of each scored "
            "object during success stability."
        ),
    )
    p.add_argument(
        "--success-angular-speed-rad-s",
        type=float,
        default=float(os.environ.get("SUCCESS_ANGULAR_SPEED_RAD_S", "0.03")),
        help=(
            "box packing: maximum angular speed of each scored "
            "object during success stability."
        ),
    )
    p.add_argument("--success-mode", choices=["bbox"], default=os.environ.get("SUCCESS_MODE", "bbox"),
                   help="cup_on_plate geometry: plate disc/z-range auto-measured from collision geometry at init; "
                        "success=the cup touches the plate with its middle over the plate and its lowest point not below it, "
                        "upright or toppled; strict_success keeps the seated bottom-contact-and-upright rule.")
    p.add_argument("--table-z", type=float, default=0.0, help="world z of the table TOP surface. DEFAULT 0.0 = the calibrated "
                   "standard scene (FR3 base from blend + gs_w2gs.npy assume surface at z=0).")
    p.add_argument("--abs-frame", action="store_true", help="place EE targets at the DATA absolute coords directly (sim world == absolute frame, no ready-anchor offset).")
    p.add_argument("--replay-object-source", choices=["none", "parquet"], default=os.environ.get("REPLAY_OBJECT_SOURCE", "none"),
                   help="replay only: set to parquet to force cup/plate poses each frame from observation.object.* columns in --replay. "
                        "Used for debugging generated object trajectories; default none keeps physical object motion.")
    p.add_argument("--world-offset", type=float, nargs=3, default=[0.0,0.0,0.0], metavar=("X","Y","Z"),
                   help="add T to the EE TARGET position in WORLD coords, then solve IK. Only the EE target moves; "
                        "robot base, desk, cup/plate, 3DGS backdrop and origin marker stay FIXED. default 0 0 0. "
                        "use to study IK reachability / jumps as a function of EE position (base-relative target genuinely changes).")
    p.add_argument("--jump-deg", type=float, default=30.0,
                   help="IK jump detector: flag a frame if ANY joint moves more than this many degrees vs the previous frame "
                        "(joint-space discontinuity / elbow-wrist flip). default 30. Set 0 to disable flagging.")
    p.add_argument("--ik-jump-reject", choices=["off", "hold", "retry-hold"], default="retry-hold",
                   help="diagnostic guard for kinematic IK: when a solved q jumps beyond --jump-deg, optionally hold the "
                        "previous q, or retry a non-deterministic IK backend before holding. The deterministic local "
                        "joint_ik backend holds immediately. default retry-hold.")
    p.add_argument("--ik-jump-retries", type=int, default=2,
                   help="number of extra IK calls for --ik-jump-reject=retry-hold. default 2.")
    p.add_argument("--policy-freeze-left", action="store_true",
                   help="diagnostic policy mode: keep the left-arm target at the initial ready pose while still rolling out "
                        "the right arm. Useful for isolating left-wrist obs corruption.")
    p.add_argument("--policy-anchor", choices=["deploy", "ready"], default="deploy",
                   help="initial pose for --policy. deploy matches the source pose-alignment profile; "
                        "ready is a hand-written diagnostic pose.")
    p.add_argument("--anchor-table-origin", default="0.25 0.0 0.60",
                   help="deploy anchor odom->table_origin translation as 'X Y Z'. Default keeps current sim anchor.")
    p.add_argument("--anchor-dump", default="", metavar="DIR",
                   help="write deploy anchor JSON (and frame0 png when rendering is enabled), then exit.")
    p.add_argument("--init-state-in", default=os.environ.get("INIT_STATE_JSON", ""), metavar="JSON",
                   help="load an exact simulator initialization state after ready-pose setup and before policy/replay starts.")
    p.add_argument("--init-state-out", default="", metavar="JSON",
                   help="write the exact simulator initialization state after ready-pose setup and optional --init-state-in.")
    p.add_argument("--init-state-only", action="store_true",
                   help="after writing --init-state-out, exit before connecting to a policy server or running frames.")
    p.add_argument("--mimicgen-init-episode", default=os.environ.get("MIMICGEN_INIT_EPISODE", ""), metavar="PARQUET",
                   help="after optional --init-state-in, replace both arms, both YUBI fingers, cup, and plate with "
                        "frame 0 from this coordinate-v2 MimicGen training episode.")
    p.add_argument("--mimicgen-init-state-log-root", default=os.environ.get("MIMICGEN_INIT_STATE_LOG_ROOT", ""), metavar="DIR",
                   help="root containing <source_sample_id>/logs/sim_log.jsonl with the generation-time FR3 IK joints "
                        "used by --mimicgen-init-episode.")
    p.add_argument("--mimicgen-init-position-tol-mm", type=float, default=0.1,
                   help="fail exact MimicGen initialization when an arm TCP or object position differs from frame 0 "
                        "by more than this tolerance. default 0.1 mm.")
    p.add_argument("--mimicgen-init-orientation-tol-deg", type=float, default=0.1,
                   help="fail exact MimicGen initialization when an arm or object orientation differs from frame 0 "
                        "by more than this tolerance. default 0.1 degrees.")
    p.add_argument("--deterministic-seed", type=int, default=int(os.environ.get("DETERMINISTIC_SEED", "-1")),
                   help="seed Python/numpy/OpenCV RNGs when >=0. Use together with --init-state-in for reproducible runs.")
    p.add_argument("--state-dump-dir", default="", metavar="DIR",
                   help="write simulator plant snapshots to DIR during rollout. dynamic mid-rollout "
                        "snapshots are diagnostic and are not accepted as fresh-enable --init-state-in.")
    p.add_argument("--state-dump-frames", default="", metavar="CSV",
                   help="comma-separated frame numbers to dump via --state-dump-dir.")
    p.add_argument("--state-dump-on-rule-grasp", action="store_true",
                   help="write a state snapshot whenever rule-grasp transitions into an active grasp.")
    p.add_argument("--state-dump-on-success", action="store_true",
                   help="write the first state snapshot whose success criterion is true.")
    p.add_argument("--grasp-hand", choices=["R","L","both"], default="R",
                   help="grasp-test hand. both drives the two hands through the same phases from "
                        "opposite ends of the target, which is what a two-handed lift needs.")
    p.add_argument("--grasp-pose-json", default="",
                   help="grasp-test --grasp-hand both: JSON giving each hand's grasp pose relative to the "
                        "target body origin, {\"R\": {\"pos_rel_m\": [x,y,z], \"quat_wxyz\": [w,x,y,z]}, \"L\": ...}. "
                        "Use it to script the lift from a pose a policy actually reached, instead of the "
                        "top-down default.")
    p.add_argument("--grasp-span", type=float, default=0.0,
                   help="grasp-test --grasp-hand both: half the distance between the two TCPs, in metres, "
                        "measured along world x from the target centre. 0 reads it from the target body's "
                        "own collision extent.")
    p.add_argument("--grasp-z", type=float, default=0.095, help="grasp-test TCP world z at grasp pose.")
    p.add_argument("--pregrasp-z", type=float, default=0.22, help="grasp-test TCP world z before descending.")
    p.add_argument("--lift-h", type=float, default=0.15, help="grasp-test TCP lift height from grasp-z.")
    p.add_argument("--grasp-dx", type=float, default=0.0, help="grasp-test TCP x offset from cup center.")
    p.add_argument("--grasp-dy", type=float, default=0.0, help="grasp-test TCP y offset from cup center.")
    p.add_argument("--grasp-yaw", type=float, default=0.0, help="grasp-test extra yaw in degrees about world Z.")
    p.add_argument("--approach", choices=["topdown","side"], default="topdown",
                   help="grasp-test approach: topdown(descend) or side(horizontal-in then lift).")
    p.add_argument("--approach-azim", type=float, default=180.0,
                   help="side approach azimuth deg; a=(cos,sin,0), gripper comes from -a. default 180 (+X side).")
    p.add_argument("--side-standoff", type=float, default=0.13,
                   help="side pre-grasp horizontal standoff (m) along -approach.")
    p.add_argument("--grasp-reach", type=float, default=0.07,
                   help="side: TCP placed this far behind cup center along -approach so fingers surround the cup. "
                        "0.07 is the empirical optimum; do NOT go below 0.05 (shallow reach -> cup pitches out on lift).")
    p.add_argument("--close-grip", type=float, default=0.0, help="grasp-test finger command for close/lift/hold phases.")
    p.add_argument("--open-grip", type=float, default=0.7854, help="grasp-test finger command for settle/open/drop phases.")
    p.add_argument("--ph-settle", type=int, default=20, help="grasp-test settle phase length in source frames.")
    p.add_argument("--ph-descend", type=int, default=25, help="grasp-test descend phase length in source frames.")
    p.add_argument("--ph-close", type=int, default=30, help="grasp-test close phase length in source frames.")
    p.add_argument("--ph-lift", type=int, default=35, help="grasp-test lift phase length in source frames.")
    p.add_argument("--ph-hold", type=int, default=20, help="grasp-test hold phase length in source frames.")
    p.add_argument("--ph-open", type=int, default=15, help="grasp-test open phase length in source frames.")
    p.add_argument("--ph-drop", type=int, default=35, help="grasp-test drop phase length in source frames.")
    p.add_argument(
        "--help-all",
        action=_HelpAllAction,
        help="show all options, including advanced contact and grasp-test controls",
    )
    a = p.parse_args(argv)
    # A --state-log-replay takes omitted wrist options from the run and refuses
    # contradicting ones (rollout.replay.recorded_wrist), so it must know which were given.
    a._explicit_wrist_options = frozenset(
        dest for dest, default in _WRIST_OPTION_DEFAULTS if getattr(a, dest) is not None
    )
    for dest, default in _WRIST_OPTION_DEFAULTS:
        if getattr(a, dest) is None:
            setattr(a, dest, default)
    if a.policy and not a.config:
        p.error("--config is required with --policy and must match the checkpoint")
    if a.franka:
        a.robot = "fr3"
    return a

__all__ = [
    "DEFAULT_GRIPPER_CALIBRATION",
    "DEFAULT_GRIPPER_XML",
    "DEFAULT_GRIPPER_XML_LEFT",
    "DEFAULT_GS_PLY",
    "GS_BACKGROUND_INDEX",
    "LEGACY_GS_ALIGN",
    "parse",
]
