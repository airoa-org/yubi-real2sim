# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Rule-grasp and contact predicates with explicit per-run session state."""

import mujoco
import numpy as np

from robosuite_env.rollout.control.grasp_intent import (
    YUBI_JAW_MAX_RAD,
    classify_policy_gripper_intent,
    evaluate_contact_angle_grasp,
)
from robosuite_env.tasks.equalities import RULE_GRASP_WELD_NAMES

#: MuJoCo weld ``torquescale`` (``eq_data[10]``) for a latched rule grasp.
#: Zero holds the object's position in the hand and leaves its orientation to
#: the solver, so the closing jaws can still turn it; one freezes the pose it
#: happened to have at first contact.
RULE_GRASP_WELD_TORQUESCALE = 0.0

#: A latched grasp starts in ``closing``: the jaws keep squeezing and ordinary
#: contact does the work, which is what rolls a pen straight. The weld is the
#: backstop for what contact cannot hold, so it waits until the jaws stop
#: making progress. These govern when "stopped" is declared.
#:
#: Measured in simulated seconds, not in calls. The call rate is not one
#: number: kinematic CbPC updates after every ``mj_step``, a dynamic path that
#: updates once per ``env.step`` has ten 2 ms steps inside it, and the replay paths
#: update after a bare ``mj_forward``. Counting calls would have made this
#: window 50 ms on one path and 500 ms on another.
#:
#: For the same reason the "how slow is stopped" test is a rate: as a per-call
#: angle, 2e-5 rad meant 0.01 rad/s sampled every 2 ms and 0.001 rad/s sampled
#: every 20 ms, so a once-per-env-step path called a creeping jaw "still closing"
#: and sat unwelded until the one-second cap.
RULE_GRASP_STALL_RATE_RAD_PER_SEC = 0.01
#: How long it has to stay that slow.
RULE_GRASP_STALL_SEC = 0.05
#: And weld regardless after this long, so a jaw that creeps forever does not
#: leave the object held by friction alone for the rest of the episode.
RULE_GRASP_CLOSING_MAX_SEC = 1.0

#: How opposed the two pads' contact normals have to be before a grasp counts.
#:
#: "Both pads touch the same object" is not a grip. In box_tape run_04 the hand
#: came down on the side of the tape and both pads landed on the same face,
#: pointing the same way; the latch fired, the jaws went on closing into a
#: surface that could not stop them, and the run ended with 12.7 mm of the pad
#: inside the object. A real pinch has the pads pressing toward each other, so
#: their contact normals point in opposite directions.
#:
#: -1 is exactly opposed and 0 is perpendicular. Where to put the line is not a
#: matter of taste: two contacts can hold an object still only if their friction
#: cones between them contain a pair of opposite directions. With cone
#: half-angles a_L = atan(mu_L) and a_R = atan(mu_R) that is
#:
#:     angle(n_L, n_R) >= 180 deg - (a_L + a_R)   i.e.   dot <= -cos(a_L + a_R)
#:
#: computed per candidate from the friction the solver actually used, and from
#: the cone it actually uses: MuJoCo's default pyramidal cone only holds
#: mu/sqrt(2) along its diagonals, so the honest mu at the pads' mu=1 is 0.707
#: and the bound lands near -1/3, not at 0. A profile that overrides pad
#: friction moves the line with it.
#:
#: This constant is the ceiling on that bound, not the bound: however sticky the
#: pads, two normals pointing the same way are never a grip.
RULE_GRASP_MAX_NORMAL_DOT = 0.0
#: Subtracted from the friction bound so a grasp sitting exactly on the
#: analytic boundary -- held by friction with nothing to spare -- is not latched
#: on solver noise.
RULE_GRASP_NORMAL_DOT_MARGIN = 0.1

#: When the pads have taken all the object they are going to take.
#:
#: Pinning translation at the latch stopped a tape roll translating out of the
#: jaws, but it left the object free to turn for as long as it was held, and
#: turning is how a seated grip works itself loose again. What the rotation is
#: *for* is the seating: a pen caught crooked is rolled square by the closing
#: pads, and that is finished once the pads have as much of the object against
#: them as they are going to get. After that, further rotation is not the grip
#: improving, it is the grip moving.
#:
#: When the grip has finished seating, the jaws stop and the orientation is
#: pinned -- one transition, no threshold of its own.
#:
#: The rotation is left free after the latch so the closing pads can roll a
#: crooked object square. The question is when that is finished, and three
#: answers were tried and rejected before this one.
#:
#: The inner-pad CONTACT COUNT is not a measure of engaged area: it counts
#: narrow-phase manifold points across a convex decomposition -- 17 collision
#: geoms on the tape, 2 on the pen, 38 on the cup, six per pad -- and moves
#: under a tenth of a degree of pose change. On a pen it never exceeds three.
#: PENETRATION DEPTH is comparable across objects but measures how hard the
#: pads press in, so treating deeper as better-seated rewards the behaviour
#: this exists to remove. RELATIVE ANGULAR VELOCITY looked right and fails on
#: exactly the runs that need it: a cup that turns slowly and continuously for
#: four hundred frames never settles, so the lock never fires.
#:
#: What does carry it is the jaws' own progress. While a crooked object is
#: turning toward the orientation the pads can hold, the jaws gain aperture;
#: once they can gain no more under closing load, that motion is finished.
#: Rotation at constant aperture is symmetry, sliding or drift -- not seating.
#:
#: So the existing stall test does the work, on the same two numbers the
#: closing phase already used: a closing rate under
#: ``RULE_GRASP_STALL_RATE_RAD_PER_SEC`` held for ``RULE_GRASP_STALL_SEC``,
#: with ``RULE_GRASP_CLOSING_MAX_SEC`` as the backstop for a jaw that creeps.
#: Nothing new to tune.

#: Slack on the ratchet, both ways. A pad pressed against an object settles
#: by oscillating around its equilibrium; without this the deepest excursion
#: becomes the floor and every rebound is teleported back into the object,
#: which drives penetration and contact-force spikes rather than holding
#: anything. Well under the 0.003-0.005 rad the real jaw varies by while it
#: holds, so the observation still moves.
RULE_GRASP_RATCHET_DEADBAND_RAD = 1e-4

#: How fast the ratchet may pull a jaw back toward its floor. Without a limit
#: the clamp teleports: on the dynamic path ten 2 ms substeps run between
#: calls, so a jaw forced open by a stiff object arrives far from its floor and
#: is put back in a single assignment -- driving the pad through the surface,
#: which generates penetration rather than a hold. Capped, the jaw re-closes
#: over a few calls, which is what a motor holding against a load does.
#: 1.5 rad/s shuts the 0.785 rad range in about half a second.
RULE_GRASP_RATCHET_MAX_RATE_RAD_PER_SEC = 1.5

def _finger_actual(state,key):
    return float(0.5*(state.d.qpos[state.ARM[key]["fj"][0]]+state.d.qpos[state.ARM[key]["fj"][1]]))

def _finger_angles(state,key):
    return {"left":float(state.d.qpos[state.ARM[key]["fj"][0]]),
            "right":float(state.d.qpos[state.ARM[key]["fj"][1]])}

def gname(state,g): return mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_GEOM,g) or f"g{g}"

def _finger_side_for_geom(state,key, gid):
    gid=int(gid)
    if gid in state.FINGER_SIDE_GEOMS[key]["left"]:
        return "left"
    if gid in state.FINGER_SIDE_GEOMS[key]["right"]:
        return "right"
    return None

def _cone_is_pyramidal(m):
    """Which friction cone this model compiled with. Default is pyramidal."""
    cone=getattr(getattr(m,"opt",None),"cone",None)
    if cone is None:
        return True
    return int(cone)==int(mujoco.mjtCone.mjCONE_PYRAMIDAL)

def _cone_friction(mu,pyramidal):
    """The coefficient the solver will actually honour in the worst direction.

    MuJoCo's default friction cone is pyramidal, not circular: it is built from
    the tangent directions, so a slide along a diagonal of the pyramid is held
    by only ``mu/sqrt(2)``. Deriving the opposition bound from the raw ``mu``
    therefore trusts a grip the solver cannot hold -- at mu=1 it admits normals
    96 degrees apart when the pyramid can only balance 109. Read
    ``opt.cone`` and use the coefficient that matches.
    """
    mu=max(0.0,float(mu))
    return mu/float(np.sqrt(2.0)) if pyramidal else mu

def _opposition_bound(side_friction,pyramidal=True):
    """How far from opposed the two pads' normals may be and still hold.

    Two contacts can produce a pair of equal and opposite forces only if their
    friction cones between them span opposite directions. Each is confined to
    its cone of half-angle ``atan(mu)``, so with the normals ``theta`` apart
    that is possible exactly when ``theta >= 180 - (a_L + a_R)``:

        dot <= -cos(a_L + a_R)

    This is about the contact forces balancing, not about the object being
    immobilised in the full wrench sense -- two points do not pin every
    rotation, and this weld deliberately leaves rotation free.

    Sticky pads widen the cones and so admit shallower grips; this is why the
    number is read from the model instead of chosen. It is then held at
    ``RULE_GRASP_MAX_NORMAL_DOT`` -- past a total cone of 180 degrees the
    inequality would admit anything, including two normals pointing the same
    way, which is the failure this exists to refuse.
    """
    if "left" not in side_friction or "right" not in side_friction:
        return RULE_GRASP_MAX_NORMAL_DOT-RULE_GRASP_NORMAL_DOT_MARGIN
    half_angles=sum(float(np.arctan(_cone_friction(side_friction[s],pyramidal)))
                    for s in ("left","right"))
    bound=-float(np.cos(min(half_angles,np.pi)))
    return min(bound,RULE_GRASP_MAX_NORMAL_DOT)-RULE_GRASP_NORMAL_DOT_MARGIN

def _rule_grasp_contact_sides(state,key,prefer_body_id=None):
    tol_m=float(state.A.grasp_contact_tol_mm)/1000.0
    by_object={}
    inner_pads=state.INNER_PAD_SIDE_GEOMS[key]["left"]|state.INNER_PAD_SIDE_GEOMS[key]["right"]
    for ci in range(state.d.ncon):
        c=state.d.contact[ci]
        g1,g2=int(c.geom1),int(c.geom2)
        if g1 in state.GRASPABLE_GEOMS and g2 in inner_pads:
            cup_geom, grip_geom=g1,g2
        elif g2 in state.GRASPABLE_GEOMS and g1 in inner_pads:
            cup_geom, grip_geom=g2,g1
        else:
            continue
        if float(c.dist) > tol_m:
            continue
        side=_finger_side_for_geom(state,key, grip_geom)
        if side is None:
            continue
        object_body_id=int(state.m.geom_bodyid[cup_geom])
        obj=by_object.setdefault(
            object_body_id,
            {
                "left":0,
                "right":0,
                "normal_force_sum_n":0.0,
                "max_pen_mm":0.0,
                "contacts":[],
            },
        )
        force6=np.zeros(6,dtype=np.float64)
        mujoco.mj_contactForce(state.m,state.d,ci,force6)
        normal_force=max(0.0,float(force6[0]))
        penetration_mm=max(0.0,-float(c.dist)*1000.0)
        obj[side]+=1
        obj["normal_force_sum_n"]+=normal_force
        # The contact frame's first row is the normal, pointing from geom1 to
        # geom2; flip it when the *object* is geom1 so every side's normal is
        # measured the same way round -- from the pad into the object.
        # (Which of the two ways round hardly matters, since reversing both
        # sides leaves their dot product alone; being consistent does.)
        normal=np.array(c.frame[0:3],dtype=np.float64)
        if cup_geom==g1:
            normal=-normal
        obj.setdefault("normals",{"left":np.zeros(3),"right":np.zeros(3)})
        # Weighted by how hard this point presses, with no floor under the
        # weight. A floor turned "nothing is pressing at all" into an unweighted
        # average of directions with a respectable magnitude, so a hand resting
        # its pads either side of an object it had not yet gripped read as a
        # firm pinch. A contact carrying no force carries no vote; waiting a
        # solver tick for the force to appear costs nothing.
        obj["normals"][side]+=normal*normal_force
        obj.setdefault("side_force",{"left":0.0,"right":0.0})[side]+=normal_force
        # The friction the solver used here, which sets how far from opposed
        # two contacts may be and still hold. Weakest contact on a side governs.
        mu=float(c.friction[0])
        frictions=obj.setdefault("side_friction",{})
        frictions[side]=mu if side not in frictions else min(frictions[side],mu)
        obj["max_pen_mm"]=max(
            float(obj["max_pen_mm"]),
            penetration_mm,
        )
        obj["contacts"].append({"finger_side":side,"gripper_geom":gname(state,grip_geom),
                                "cup_geom":gname(state,cup_geom),"dist_mm":round(float(c.dist)*1000.0,3),
                                "normal_force_n":round(normal_force,4)})
    pyramidal=_cone_is_pyramidal(state.m)

    def grip_quality(sides):
        """both / pressing / normal_dot / bound / opposed for one candidate."""
        both=bool(sides["left"]>0 and sides["right"]>0)
        forces=sides.get("side_force") or {}
        pressing=(float(forces.get("left",0.0))>0.0
                  and float(forces.get("right",0.0))>0.0)
        # Opposed, or merely both present? Two pads on one face of a tape roll
        # are both touching and are not holding anything.
        normal_dot=1.0
        normals=sides.get("normals")
        if both and pressing and normals is not None:
            left=normals["left"]; right=normals["right"]
            ln=float(np.linalg.norm(left)); rn=float(np.linalg.norm(right))
            if ln>1e-12 and rn>1e-12:
                normal_dot=float(np.dot(left/ln,right/rn))
        bound=_opposition_bound(sides.get("side_friction") or {},
                                pyramidal=pyramidal)
        return both,pressing,normal_dot,bound,bool(both and pressing
                                                   and normal_dot<=bound)

    if not by_object:
        return {"left":0,"right":0,"both":False,"same_object":False,
                "inner_pad_only":True,"normal_dot":1.0,"opposed":False,
                "normal_dot_bound":_opposition_bound({},pyramidal=pyramidal),
                "pads_pressing":False,
                "object_body_id":None,"object_body":None,"contacts":[]}
    # Opposition outranks force. Sorting on force alone let a hand pressing
    # hard on the side of one object hide a proper pinch on another: in a
    # box_pack scene every free-joint object is a candidate, so a 20 N shove
    # against a distractor beat a 2 N grip on the target and the run reported
    # the distractor. Among genuine grips, rank by how hard the pads press
    # rather than by how many contact points the solver happened to generate,
    # so a cup brushed at three points does not outrank one squeezed at two.
    quality={body:grip_quality(sides) for body,sides in by_object.items()}
    def rank(item):
        body,sides=item
        both,pressing,_,_,opposed=quality[body]
        return (opposed,both and pressing,sides["normal_force_sum_n"],
                sides["left"]+sides["right"])
    # An active grasp reads its own object. Otherwise a distractor that gets
    # squeezed harder mid-grasp would silently become "the" contact, and the
    # closing phase would decide it had lost the thing it is holding.
    if prefer_body_id is not None and int(prefer_body_id) in by_object:
        object_body_id=int(prefer_body_id)
        sides=by_object[object_body_id]
    else:
        object_body_id,sides=max(by_object.items(),key=rank)
    both,pressing,normal_dot,bound,opposed=quality[object_body_id]
    return {"left":int(sides["left"]),"right":int(sides["right"]),
            "both":both,"same_object":both,"inner_pad_only":True,
            "normal_dot":round(normal_dot,4),"opposed":opposed,
            "normal_dot_bound":round(bound,4),"pads_pressing":pressing,
            "normal_force_sum_n":round(float(sides["normal_force_sum_n"]),4),
            "max_pen_mm":round(float(sides["max_pen_mm"]),4),
            "object_body_id":int(object_body_id),
            "object_body":mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_BODY,object_body_id),
            "contacts":sides["contacts"][:8]}

def _grip_effort_direction(state,key, grip_cmds=None, grip_intents=None):
    if grip_intents is not None and key in grip_intents:
        return dict(grip_intents[key])
    if grip_cmds is not None and key in grip_cmds:
        return classify_policy_gripper_intent(
            grip_cmds[key],
            _finger_actual(state,key),
            state.A.rule_grasp_policy_eps_rad,
        )

    value=0.0
    source="none"
    if state.YUBI is not None:
        side="right" if key=="R" else "left"
        tel=getattr(state.YUBI,side).telemetry
        if isinstance(tel,dict) and "tau_cmd" in tel:
            value=float(tel.get("tau_cmd",0.0))
            source="yubi.tau_cmd"
    if source=="none":
        vals=[]
        for aa in state.ARM[key]["fa"]:
            if aa >= 0:
                vals.append(float(state.d.actuator_force[aa]))
        if vals:
            value=float(np.mean(vals))
            source="actuator_force_mean"
    eps=float(state.A.rule_grasp_effort_eps)
    if value < -eps:
        direction="closing"
    elif value > eps:
        direction="opening"
    else:
        direction="neutral"
    return {"value":round(float(value),6),"direction":direction,"source":source}

def _has_free_joint_ancestor(state,body_id):
    """Whether ``body_id`` belongs to a freely moving graspable object."""
    if body_id is None:
        return False
    b=int(body_id)
    while b>0:
        adr=int(state.m.body_jntadr[b]); num=int(state.m.body_jntnum[b])
        for j in range(adr,adr+num):
            if j>=0 and int(state.m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_FREE):
                return True
        b=int(state.m.body_parentid[b])
    return False

def _relpose(d,b1,b2):
    """Return body2's pose in body1 coordinates as (position, wxyz)."""
    p1,q1=d.xpos[b1].copy(),d.xquat[b1].copy()
    p2,q2=d.xpos[b2].copy(),d.xquat[b2].copy()
    q1i=np.zeros(4); mujoco.mju_negQuat(q1i,q1)
    dp=np.zeros(3); mujoco.mju_rotVecQuat(dp,p2-p1,q1i)
    dq=np.zeros(4); mujoco.mju_mulQuat(dq,q1i,q2)
    return dp,dq

def _rule_grasp_weld_eq_id(state,hand):
    try:
        name=RULE_GRASP_WELD_NAMES[hand]
    except KeyError as exc:
        raise ValueError(f"unknown rule-grasp hand {hand!r}") from exc
    eq_id=int(mujoco.mj_name2id(state.m,mujoco.mjtObj.mjOBJ_EQUALITY,name))
    if eq_id<0:
        raise RuntimeError(f"compiled model is missing rule-grasp weld {name!r}")
    if int(state.m.eq_type[eq_id])!=int(mujoco.mjtEq.mjEQ_WELD):
        raise RuntimeError(f"rule-grasp equality {name!r} is not an mjEQ_WELD")
    return eq_id

def _activate_rule_grasp_weld(state,hand,object_body_id):
    """Retarget and activate the hand-specific weld at the current relative pose."""
    eq_id=_rule_grasp_weld_eq_id(state,hand)
    if state.RULE_GRASP.get("weld_eq_id") is not None:
        raise RuntimeError("rule-grasp weld activation attempted while another weld is latched")
    if bool(state.d.eq_active[eq_id]):
        raise RuntimeError(
            f"rule-grasp weld {RULE_GRASP_WELD_NAMES[hand]!r} was already active"
        )
    object_body_id=int(object_body_id)
    # eef_pose() reads this exact body, so the constraint and telemetry use the
    # same hand frame rather than the policy TCP or a gripper child body.
    hand_body_id=int(state.ARM[hand]["eef"])
    # The hand is body 1 and the object body 2, which at torquescale 0 holds
    # the object's own origin at a fixed point in the hand. The other order
    # pins a point of the object to the hand's origin instead, and the object
    # then orbits the hand as it turns -- a pen being squeezed straight would
    # swing its centre out of the jaws.
    dp,dq=_relpose(state.d,hand_body_id,object_body_id)
    state.m.eq_obj1id[eq_id]=hand_body_id
    state.m.eq_obj2id[eq_id]=object_body_id
    state.m.eq_data[eq_id,0:3]=0.0
    state.m.eq_data[eq_id,3:6]=dp
    state.m.eq_data[eq_id,6:10]=dq
    # torquescale 0: the weld holds the object's position in the hand and
    # leaves its orientation to the solver. A real grasp does the same -- the
    # jaws stop the object going anywhere, and closing them further rolls a
    # pen or a tape roll straight rather than carrying it at whatever angle it
    # happened to be at first contact. At torquescale 1 the object freezes at
    # the latch orientation, so a pen picked up crooked stays crooked and the
    # jaws have nothing left to do.
    state.m.eq_data[eq_id,10]=RULE_GRASP_WELD_TORQUESCALE
    state.d.eq_active[eq_id]=1
    return eq_id

def _tighten_rule_grasp_weld(state,hand):
    """Pin the orientation too, at the pose the seating actually reached.

    The relative pose is re-read here rather than reused from the latch. The
    whole point of the first phase is that the object turns during it, so the
    orientation recorded at the latch is the crooked one; writing torquescale 1
    against it would snap the object back to however it was first caught, which
    is the behaviour the deferred weld existed to avoid.
    """
    eq_id=state.RULE_GRASP.get("weld_eq_id")
    if eq_id is None:
        raise RuntimeError("cannot tighten a rule grasp that is not welded")
    eq_id=int(eq_id)
    hand_body_id=int(state.ARM[hand]["eef"])
    object_body_id=int(state.RULE_GRASP["object_body_id"])
    _,dq=_relpose(state.d,hand_body_id,object_body_id)
    # Only the orientation. Rewriting the translation too would absorb whatever
    # position error the soft weld happens to be carrying at this instant and
    # make it permanent, and the translation was never the thing being freed.
    state.m.eq_data[eq_id,6:10]=dq
    state.m.eq_data[eq_id,10]=1.0
    state.RULE_GRASP["rotation_pinned"]=True

def _seating_contact_is_valid(state,contacts):
    """Are the pads still holding the object this grasp latched onto?

    Without this the dwell advances while the grip is falling apart: contacts
    thinning out, the pads coming off, the selector falling back to a different
    object entirely. Every one of those stops the jaws gaining aperture, which
    is exactly what "seated" looks like, and the weld would finalise onto
    something the hand is not holding and stay that way for the episode.
    """
    if not contacts:
        return False
    if not contacts.get("opposed"):
        return False
    latched=state.RULE_GRASP.get("object_body_id")
    return latched is None or contacts.get("object_body_id")==latched

def _engage_rule_grasp(state,hand,frame,substep,contacts,reason):
    """Stop the jaws where they are and pin the orientation, together.

    One transition rather than two, because they answer the same question. The
    jaws stop at their *current* angles, which is not the latch angle: the
    seating interval is real and the jaws close during it. ``pinned_finger_q``
    stays the latch reading -- it is what the release test compares against --
    so the angles they actually came to rest at are recorded separately.
    """
    state.RULE_GRASP["stopped_finger_q"]=_finger_qpos(state,hand)
    _tighten_rule_grasp_weld(state,hand)
    _rule_grasp_event(state,"weld_tightened", hand, frame, substep, reason,
                      contact_sides=contacts)

def _update_seating(state,hand,frame,substep,contacts):
    """Watch the jaws close, and finish the grasp when they stop closing.

    The dwell is consecutive and any guard failure resets it: the thing being
    waited for is a grip that has settled, not an average over one that keeps
    losing and regaining the object.
    """
    now=float(state.d.time)
    if state.RULE_GRASP.get("seating_started_sec") is None:
        state.RULE_GRASP["seating_started_sec"]=now
    started=float(state.RULE_GRASP["seating_started_sec"])
    last_sec=state.RULE_GRASP.get("seating_last_sec")
    elapsed=0.0 if last_sec is None else max(0.0,now-float(last_sec))
    state.RULE_GRASP["seating_last_sec"]=now

    if not _seating_contact_is_valid(state,contacts):
        # Not holding it at this instant. Drop the baseline as well as the
        # dwell: keeping the aperture reading from an invalid sample would let
        # the next valid one claim the whole gap as settled time.
        state.RULE_GRASP["seating_settled_sec"]=0.0
        state.RULE_GRASP["seating_last_rad"]=None
        state.RULE_GRASP["seating_valid_sec"]=0.0
        # The deadline still applies, and it finishes the grasp rather than
        # throwing it away: finishing rather than abandoning keeps latched
        # grasps. A latch is a decision that the pads are opposed on the
        # object; nothing after it should quietly undo that.
        #
        # It cannot hold forever either way: the grasp is finalised here, and
        # from then on it is a full weld that the policy releases by opening.
        if (now-started)>=RULE_GRASP_CLOSING_MAX_SEC:
            _engage_rule_grasp(state,hand,frame,substep,contacts,
                               "seating_deadline_without_a_settled_grip")
        return
    state.RULE_GRASP["seating_valid_sec"]=(
        float(state.RULE_GRASP.get("seating_valid_sec",0.0))+elapsed)

    aperture=_closure_measure(state,hand)
    previous=state.RULE_GRASP.get("seating_last_rad")
    state.RULE_GRASP["seating_last_rad"]=aperture
    settled=float(state.RULE_GRASP.get("seating_settled_sec",0.0))
    if previous is None or elapsed<=0.0:
        # No baseline yet, or no simulated time passed -- the replay paths call
        # mj_forward only. A rate is undefined; counting calls instead would
        # invent a cadence matching neither the 2 ms nor the 20 ms runtime one.
        pass
    elif (float(previous)-aperture)/elapsed>RULE_GRASP_STALL_RATE_RAD_PER_SEC:
        settled=0.0
    else:
        settled+=elapsed
    state.RULE_GRASP["seating_settled_sec"]=settled

    if settled>=RULE_GRASP_STALL_SEC:
        _engage_rule_grasp(state,hand,frame,substep,contacts,"jaws_stopped_closing")
        return
    # A jaw that creeps forever would otherwise hold the object half-welded for
    # the rest of the episode. Finish it, but only once the pads have actually
    # held it for a window -- a grasp that spent the whole second flickering
    # has not earned a full weld.
    if (now-started)>=RULE_GRASP_CLOSING_MAX_SEC:
        _engage_rule_grasp(state,hand,frame,substep,contacts,
                           "seating_time_cap")

def _deactivate_rule_grasp_weld(state,hand):
    """Release the active weld without modifying object generalized velocity."""
    eq_id=state.RULE_GRASP.get("weld_eq_id")
    if eq_id is None:
        raise RuntimeError("active rule grasp has no weld equality ID")
    eq_id=int(eq_id)
    expected=_rule_grasp_weld_eq_id(state,hand)
    if eq_id!=expected:
        raise RuntimeError(
            f"active rule-grasp weld ID {eq_id} does not belong to hand {hand}"
        )
    state.d.eq_active[eq_id]=0

def _rule_grasp_hands(state):
    return ("R","L") if state.A.rule_grasp_hand=="auto" else (state.A.rule_grasp_hand,)

def _rule_grasp_event(state, kind, hand, frame=None, substep=None, reason=None,
                      intent=None, contact_sides=None, predicate=None):
    ev={"event":kind,"hand":hand}
    if frame is not None: ev["frame"]=int(frame)
    if substep is not None: ev["substep"]=int(substep)
    if reason: ev["reason"]=reason
    if intent is not None: ev["intent"]=dict(intent)
    if contact_sides is not None: ev["contact_sides"]=dict(contact_sides)
    if predicate is not None: ev["predicate"]=dict(predicate)
    state.RULE_GRASP["events"].append(ev)
    return ev

def _finger_qpos(state,hand):
    """Raw per-joint jaw angles, not the two-joint mean _finger_actual returns.

    Writing a mean back to both joints would move an asymmetric grasp.
    """
    return [float(state.d.qpos[adr]) for adr in state.ARM[hand]["fj"]]

def _hold_finger_joints(state,hand,angles):
    """Keep the jaws at these angles, and take their motion out of the solver.

    ``qacc_warmstart`` as well as velocity and acceleration: MuJoCo warm-starts
    the next solve from it, so leaving the accelerations of a joint just moved
    by hand in there re-applies the motion this exists to stop.
    """
    for index,(qadr,dadr) in enumerate(zip(state.ARM[hand]["fj"],
                                           state.ARM[hand]["fd"])):
        if index>=len(angles):
            break
        state.d.qpos[qadr]=float(angles[index])
        state.d.qvel[dadr]=0.0
        state.d.qacc[dadr]=0.0
        state.d.qacc_warmstart[dadr]=0.0

def _ratchet_grasped_fingers(state,hand):
    """Let the jaws keep closing on the object, and never let them open.

    Holding the jaws at the latch angle -- which is what this did -- made the
    gripper channel of the observation a constant for the whole of a grasp:
    ``grip_state_obs`` reads these joints, so from the policy's side the hand
    stopped responding the instant it closed. The real rig does not do that.
    Over the hold phase of sixty-four recorded tape episodes the right jaw
    carries a standard deviation of 0.003-0.005 rad and takes six to nine
    distinct values; a real gripper closing on a rigid object stalls near the
    object's width, but it does not become a single number.

    So the jaws are left to the solver, with one constraint: they may close
    but not open. The floor is the most-closed angle reached so far, which is
    what a ratchet does and what a closed gripper under load does. Smaller
    angles are more closed here.

    This cannot make a grasp impossible to release. ``opened_above_contact``
    compares the *policy's* target against the contact angle saved at latch,
    not the measured jaws (``grasp_intent.evaluate_contact_angle_grasp``), and
    it is evaluated before this runs. Clamping the jaws is therefore invisible
    to the release decision, and an earlier version that skipped the clamp on
    a commanded open bought nothing while letting a welded grasp fall open.
    """
    stopped=state.RULE_GRASP.get("stopped_finger_q")
    if stopped:
        # Seated. The jaws hold where they came to rest, in both directions --
        # capping only the recorded floor would have left the actual jaw free
        # to keep closing below it, which is the closure that drove the fingers
        # into a cup whose origin is welded and cannot move out of the way.
        _hold_finger_joints(state,hand,stopped)
        return
    ratchet=state.RULE_GRASP.get("ratchet_finger_q")
    if not ratchet:
        return
    # qpos and dof addresses are not interchangeable: the model keeps both
    # because a joint's position and velocity indices only coincide by
    # accident of compilation order. Indexing qvel with a qpos address happens
    # to work on one build and would silently corrupt another.
    qpos_addresses=state.ARM[hand]["fj"]
    dof_addresses=state.ARM[hand]["fd"]
    # The clamp is a rate, so it needs to know how much simulated time the
    # caller ran. Its own clock, because this runs both before and after the
    # weld while the closing-phase timers only run before it.
    jaw_low,jaw_high=_jaw_limits(state,hand)
    now=float(state.d.time)
    previous=state.RULE_GRASP.get("ratchet_last_sec")
    elapsed=0.0 if previous is None else max(0.0,now-float(previous))
    state.RULE_GRASP["ratchet_last_sec"]=now
    # No simulated time since the last call earns no allowance at all. An
    # earlier version let this case move the jaw without limit, on the argument
    # that nothing had been integrated so nothing could be driven anywhere --
    # which is wrong as soon as a physics step follows, because that step then
    # starts from a configuration the clamp just pushed into the object. The
    # first call after a latch lands here too and needs nothing: its floor was
    # copied from the very jaws it is about to check.
    max_close=RULE_GRASP_RATCHET_MAX_RATE_RAD_PER_SEC*elapsed
    for index,(qadr,dadr) in enumerate(zip(qpos_addresses,dof_addresses)):
        q=float(state.d.qpos[qadr])
        floor=float(ratchet[index])
        if q<floor-RULE_GRASP_RATCHET_DEADBAND_RAD:
            # Closed appreciably further: that is the new floor. A deadband,
            # rather than every transient minimum, because a compliant contact
            # oscillates around its equilibrium and taking the deepest
            # excursion as permanent would teleport the jaw back into the
            # object on every rebound -- which is how a ratchet turns into a
            # penetration and force-spike generator.
            #
            # Held inside the joint's own range. The solver can overshoot a
            # limit by a hair on a hard contact, and recording that as the
            # floor would let the ratchet keep driving a jaw back through its
            # physical stop for the rest of the grasp.
            ratchet[index]=min(max(q,jaw_low[index]),jaw_high[index])
            continue
        if q>floor+RULE_GRASP_RATCHET_DEADBAND_RAD:
            # Toward the floor at a bounded speed, not onto it. A jaw levered
            # open by a stiff object arrives a long way out -- ten substeps run
            # between calls on the dynamic path -- and putting it back in one
            # assignment drives the pad through the surface. The floor stays
            # where it is, so nothing is given up; the jaw just takes a few
            # calls to get back to it, as a loaded motor would.
            state.d.qpos[qadr]=max(floor,q-max_close)
            state.d.qvel[dadr]=0.0
            state.d.qacc[dadr]=0.0
            # MuJoCo warm-starts the next solve from qacc_warmstart; leaving
            # the accelerations of a jaw we just moved in there re-applies the
            # motion this clamp exists to undo.
            state.d.qacc_warmstart[dadr]=0.0

def _carry_weight_while_closing(state,body_id):
    """Hold the object's weight while the jaws are still closing on it.

    Between the latch and the weld nothing holds the object but friction, so a
    pen caught lightly falls out of the hand before the jaws have finished
    shutting on it -- and what the closing phase is for is exactly the time it
    takes contact to settle an object that was caught badly.

    Cancelling the weight leaves everything else to the solver: the object can
    still roll, slide and be turned square by the pads. It just does not sink
    while that happens. Read from ``opt.gravity`` rather than assumed, so a
    model with gravity off or tilted stays consistent.
    """
    body_id=int(body_id)
    mass=float(state.m.body_subtreemass[body_id])
    state.d.xfrc_applied[body_id,0:3]=-mass*np.asarray(state.m.opt.gravity,dtype=float)
    state.d.xfrc_applied[body_id,3:6]=0.0
    # Who is being carried, recorded where the state machine can find it. An
    # applied force is not cleared by MuJoCo between steps, so the one thing
    # that must never happen is losing track of a body we are holding up.
    state.RULE_GRASP["carried_body_id"]=body_id


def _drop_carried_weight(state,body_id=None):
    """Give the weight back: the weld has it now, or the grasp is over.

    Takes whatever is on record when no body is named, so a caller cannot
    clear the wrong one -- or, worse, clear nothing because the grasp fields
    were reset a line too early.
    """
    if body_id is None:
        body_id=state.RULE_GRASP.get("carried_body_id")
    if body_id is None:
        return
    state.d.xfrc_applied[int(body_id),0:6]=0.0
    state.RULE_GRASP["carried_body_id"]=None


def _cleared_rule_grasp_fields():
    """Everything a latch owns, back to nothing. One definition, three callers."""
    return {"active":False,"hand":None,"weld_eq_id":None,
            "object_body_id":None,"object_body":None,
            "pinned_finger_q":None,"ratchet_finger_q":None,"welded_finger_q":None,
            "closing_started_sec":None,"closing_last_sec":None,
            "ratchet_last_sec":None,
            "closing_stalled_sec":0.0,"closing_lost_contact_sec":0.0,
            "closing_opposed_sec":0.0,
            "closing_last_rad":None,
            "rotation_pinned":False,"seating_settled_sec":0.0,
            "seating_last_rad":None,"seating_last_sec":None,
            "seating_started_sec":None,"seating_valid_sec":0.0,
            "stopped_finger_q":None,
            "contact_angles_rad":None,"policy_target_at_grasp_rad":None}

def _jaw_limits(state,hand):
    """Per-jaw travel, from the model where the model says.

    The solver can push a joint a hair past its stop on a hard contact. The
    ratchet records where the jaw got to, so without this an overshoot becomes
    a floor outside the jaw's physical range and the clamp spends the rest of
    the grasp driving the jaw back through its own limit. Falls back to the
    declared YUBI range for the benches, which build a hand without the arm
    bookkeeping around it.
    """
    jids=state.ARM[hand].get("finger_jids") if hasattr(state.ARM[hand],"get") else None
    low=[];high=[]
    for index in range(len(state.ARM[hand]["fj"])):
        lo,hi=0.0,YUBI_JAW_MAX_RAD
        if jids is not None and index<len(jids):
            rng=state.m.jnt_range[int(jids[index])]
            if float(rng[1])>float(rng[0]):
                lo,hi=float(rng[0]),float(rng[1])
        low.append(lo);high.append(hi)
    return low,high

def _closure_measure(state,hand):
    """Total aperture. Smaller is more closed.

    The sum rather than the wider jaw: with a lopsided grasp one jaw can still
    be travelling while the other has stopped, and taking the maximum would
    call that stalled and weld in the middle of the motion.
    """
    return sum(float(state.d.qpos[adr]) for adr in state.ARM[hand]["fj"])

def _abandon_rule_grasp(state,hand,contacts=None):
    """Give up a latch that never became a grip, and do not take it again.

    Clearing the fields is not enough on its own: the caller goes straight on
    to the loop that looks for something to latch, and the very contacts that
    were just rejected are still there, so the same grasp is taken again in
    the same call -- an abandon and a latch in one update, forever.

    So the object is remembered as refused. The refusal is dropped as soon as
    the pads are off it, or on something else, which is the point at which a
    fresh approach could genuinely be different.
    """
    # The object that was latched, not whichever one the pads happen to be
    # touching now. When the latched body has rolled clear of the pads
    # entirely, ``contacts`` describes some other object, and refusing that
    # one would blame a body this hand never grasped.
    refused=state.RULE_GRASP.get("object_body_id")
    if refused is None and contacts is not None:
        refused=contacts.get("object_body_id")
    # The weld goes on at the latch now, so an abandoned grasp has one to take
    # down. Clearing weld_eq_id without deactivating the equality leaves it
    # holding the object while the state machine believes nothing is held, and
    # the next latch dies on "weld was already active".
    if state.RULE_GRASP.get("weld_eq_id") is not None:
        _deactivate_rule_grasp_weld(state,hand)
    _drop_carried_weight(state)
    state.RULE_GRASP.update(_cleared_rule_grasp_fields())
    refusals=dict(state.RULE_GRASP.get("refused_body_id") or {})
    refusals[hand]=None if refused is None else int(refused)
    state.RULE_GRASP["refused_body_id"]=refusals

def _relatch_refused(state,hand,contacts):
    """Is this the object this hand just gave up on, still in the same grip?

    Per hand. One shared field meant the idle hand's empty contact list read as
    "not opposed" and cleared the other hand's refusal, so the grasp that had
    just been abandoned was taken again on the next update -- the same-call
    loop, moved one tick later and no less endless.
    """
    refusals=state.RULE_GRASP.get("refused_body_id") or {}
    refused=refusals.get(hand)
    if refused is None:
        return False
    if not contacts.get("opposed") or contacts.get("object_body_id")!=refused:
        # Off it, or holding it differently now. Let the next approach try.
        refusals=dict(refusals); refusals[hand]=None
        state.RULE_GRASP["refused_body_id"]=refusals
        return False
    return True

def _apply_rule_grasp_hold(state,target_rad=None,frame=None,substep=None,
                           contacts=None):
    """Squeeze while the squeezing is working; weld once it stops working.

    Welding at first contact fixed the object at whatever pose it had when two
    pads happened to touch it -- a pen caught across the jaws stayed across
    them. Contact can fix that on its own if it is given the chance, so a fresh
    latch starts unwelded and the jaws simply keep closing. The weld goes on
    when the jaws stop making progress, which is the point where contact has
    done what it can and the object needs holding rather than shaping.
    """
    hand=state.RULE_GRASP.get("hand")
    if not hand:
        return
    _ratchet_grasped_fingers(state,hand)

    eq_id=state.RULE_GRASP.get("weld_eq_id")
    if eq_id is not None:
        if not bool(state.d.eq_active[int(eq_id)]):
            raise RuntimeError("active rule grasp lost its weld equality")
        if not state.RULE_GRASP.get("rotation_pinned"):
            _update_seating(state,hand,frame,substep,contacts)
        return

    now=float(state.d.time)
    if state.RULE_GRASP.get("closing_started_sec") is None:
        state.RULE_GRASP["closing_started_sec"]=now
    started=float(state.RULE_GRASP["closing_started_sec"])
    last_sec=state.RULE_GRASP.get("closing_last_sec")
    elapsed=0.0 if last_sec is None else max(0.0,now-float(last_sec))
    state.RULE_GRASP["closing_last_sec"]=now

    # Nothing is welded yet, so a latch whose object has gone is a latch on
    # nothing. Without this the weld eventually fires on a pen already on the
    # floor and holds it wherever it fell to.
    holding=(contacts is None or
             (contacts.get("opposed")
              and contacts.get("object_body_id")==state.RULE_GRASP.get("object_body_id")))
    if holding:
        state.RULE_GRASP["closing_lost_contact_sec"]=0.0
        # Consecutive, not cumulative. Summed, a hand that flickered -- 40 ms
        # off, 20 ms on, three times -- reached the window without ever once
        # holding the object for it, and welded on the third isolated sample.
        state.RULE_GRASP["closing_opposed_sec"]=(
            float(state.RULE_GRASP.get("closing_opposed_sec",0.0))+elapsed)
    else:
        lost=float(state.RULE_GRASP.get("closing_lost_contact_sec",0.0))+elapsed
        state.RULE_GRASP["closing_lost_contact_sec"]=lost
        if lost>RULE_GRASP_STALL_SEC:
            _rule_grasp_event(state,"grasp_abandoned", hand, frame, substep,
                              "lost_both_pads_before_weld",contact_sides=contacts)
            _abandon_rule_grasp(state,hand,contacts)
            return
        # Inside the grace window, but not pinched at this instant -- and the
        # weld must not fire here. The stall window or the time cap can already
        # be satisfied, and welding on them now would pin an object the pads are
        # not opposed on: the side grasp this phase exists to refuse, reached by
        # the back door instead of the front. So return before the weld.
        #
        # Both clocks restart. Freezing the stall clock while resetting the
        # opposed one built a trap: after a single dropped sample the stall
        # clock could mature first, and the very next opposed sample would meet
        # the weld condition with almost no opposed time behind it and be
        # abandoned -- so a grip that had just steadied itself was thrown away
        # and never given the chance to earn a fresh window. Restarting both
        # keeps them in step, which makes "the jaws stopped" mean "the jaws
        # stopped while actually holding it".
        #
        # The closure is recorded so the next call measures progress across the
        # gap rather than against a stale reading. And the weight goes back,
        # because nothing is holding the object just now.
        state.RULE_GRASP["closing_last_rad"]=_closure_measure(state,hand)
        state.RULE_GRASP["closing_opposed_sec"]=0.0
        state.RULE_GRASP["closing_stalled_sec"]=0.0
        _drop_carried_weight(state)
        return

    closure=_closure_measure(state,hand)
    previous=state.RULE_GRASP.get("closing_last_rad")
    stalled=float(state.RULE_GRASP.get("closing_stalled_sec",0.0))
    closing_rate=(0.0 if previous is None or elapsed<=0.0
                  else (float(previous)-closure)/elapsed)
    if previous is None or closing_rate>RULE_GRASP_STALL_RATE_RAD_PER_SEC:
        stalled=0.0
    else:
        stalled+=elapsed
    state.RULE_GRASP["closing_last_rad"]=closure
    state.RULE_GRASP["closing_stalled_sec"]=stalled

    if stalled<RULE_GRASP_STALL_SEC and (now-started)<RULE_GRASP_CLOSING_MAX_SEC:
        _carry_weight_while_closing(state,state.RULE_GRASP["object_body_id"])
        return
    # The jaws have settled. Weld at the pose the object reached, not the one
    # it was caught at.
    reason=("jaws_stopped_closing" if stalled>=RULE_GRASP_STALL_SEC
            else "closing_phase_time_cap")
    # The cap is a backstop for a jaw that creeps, not a licence to weld onto
    # whatever happens to be touching when the second runs out. A grasp that
    # spent that second flickering in and out of opposition has not been held
    # at any point, and the one opposed sample that lands on the cap is not
    # evidence that it is held now. Let it go instead.
    #
    # Both reasons, not just the cap. The stall clock is frozen rather than
    # reset across a flicker, so it too can reach its window without the pads
    # ever having held the object for one; guarding only the cap left
    # "jaws_stopped_closing" as a second way in.
    opposed_sec=float(state.RULE_GRASP.get("closing_opposed_sec",0.0))
    if opposed_sec<RULE_GRASP_STALL_SEC:
        _rule_grasp_event(state,"grasp_abandoned", hand, frame, substep,
                          "never_held_the_object_for_a_whole_window",
                          contact_sides=contacts)
        _abandon_rule_grasp(state,hand,contacts)
        return
    _drop_carried_weight(state)
    state.RULE_GRASP["weld_eq_id"]=_activate_rule_grasp_weld(
        state,hand,state.RULE_GRASP["object_body_id"]
    )
    state.RULE_GRASP["welded_finger_q"]=_finger_qpos(state,hand)
    _rule_grasp_event(state,"weld", hand, frame, substep, reason)

def _rule_grasp_update(state,grip_cmds=None, frame=None, substep=None, grip_intents=None):
    # The carried weight lives for exactly one step. MuJoCo keeps
    # ``xfrc_applied`` until something clears it, so rather than trusting every
    # exit from the state machine to tidy up, it is dropped here at the top of
    # every update and re-applied below only while a grasp is still closing.
    # A force that outlived its grasp would hold an object in mid-air, which is
    # both a plausible mistake and an almost invisible one -- the object simply
    # does not fall, and nothing in the summary says why. This also covers the
    # non-rule return below: turning rule grasp off mid-run must not leave a
    # previously carried object floating.
    _drop_carried_weight(state)
    if state.A.grasp_mode!="rule":
        return _rule_grasp_status(state)
    last={}
    if state.RULE_GRASP["active"]:
        hand=state.RULE_GRASP["hand"]
        effort=_grip_effort_direction(state,hand, grip_cmds, grip_intents)
        contacts=_rule_grasp_contact_sides(
            state,hand,prefer_body_id=state.RULE_GRASP.get("object_body_id"))
        actual_angles=_finger_angles(state,hand)
        contact_angles=state.RULE_GRASP.get("contact_angles_rad")
        target=float(effort.get("target_rad",grip_cmds.get(hand) if grip_cmds else
                               state.RULE_GRASP.get("policy_target_at_grasp_rad")))
        predicate=evaluate_contact_angle_grasp(
            target,contact_angles,actual_angles,state.A.rule_grasp_policy_eps_rad,
            state.A.rule_grasp_release_eps_rad,YUBI_JAW_MAX_RAD)
        last[hand]={"effort":effort,"contact_sides":contacts,
                    "contact_angle_predicate":predicate}
        if predicate["opened_above_contact"]:
            # A grasp still in its closing phase has no weld to drop, but it
            # does have the object's weight held off; that has to go back
            # either way, or a released object hangs in the air.
            _drop_carried_weight(state)
            if state.RULE_GRASP.get("weld_eq_id") is not None:
                _deactivate_rule_grasp_weld(state,hand)
            _rule_grasp_event(state,"release", hand, frame, substep,
                              "finger_angle_above_latched_contact_angle",
                              intent=effort,contact_sides=contacts,predicate=predicate)
            state.RULE_GRASP.update(_cleared_rule_grasp_fields())
            # This hand opened. Whatever it refused earlier, its next approach
            # is a new one -- and only its own: the other hand's refusal is
            # none of its business.
            refusals=dict(state.RULE_GRASP.get("refused_body_id") or {})
            refusals[hand]=None
            state.RULE_GRASP["refused_body_id"]=refusals
            state.RULE_GRASP["last"]=last
            return _rule_grasp_status(state)
        else:
            _apply_rule_grasp_hold(state,target,frame,substep,contacts)
            if not state.RULE_GRASP["active"]:
                # It gave up this call. Going straight on to look for something
                # to latch would take a decision about a hand whose contacts
                # were just judged, in the same breath as judging them; let the
                # next update look with a fresh reading.
                state.RULE_GRASP["last"]=last
                return _rule_grasp_status(state)
    if not state.RULE_GRASP["active"]:
        for hand in _rule_grasp_hands(state):
            contacts=_rule_grasp_contact_sides(state,hand)
            effort=_grip_effort_direction(state,hand, grip_cmds, grip_intents)
            actual_angles=_finger_angles(state,hand)
            target=float(effort.get("target_rad",grip_cmds.get(hand) if grip_cmds else
                                   _finger_actual(state,hand)))
            predicate=evaluate_contact_angle_grasp(
                target,actual_angles,actual_angles,state.A.rule_grasp_policy_eps_rad,
                state.A.rule_grasp_release_eps_rad,YUBI_JAW_MAX_RAD)
            last[hand]={"effort":effort,"contact_sides":contacts,
                        "contact_angle_predicate":predicate}
            if (contacts["both"] and contacts["same_object"]
                    and not contacts["opposed"]
                    and predicate["closing_below_contact"]):
                # Both pads on one face is not a grip. Say so once per hand per
                # step rather than silently declining, so a run can be read for
                # how much of its time it spent pressing on a side.
                _rule_grasp_event(state,"grasp_skipped", hand, frame, substep,
                                  "pads_not_opposed",
                                  intent=effort,contact_sides=contacts,predicate=predicate)
                continue
            if _relatch_refused(state,hand,contacts):
                # Abandoned this object moments ago and still holding it the
                # same way. Taking it again in the same update is how an
                # abandon turns into an endless latch/abandon pair.
                continue
            if (contacts["both"] and contacts["same_object"]
                    and contacts["opposed"]
                    and predicate["closing_below_contact"]):
                if not predicate["release_reachable"]:
                    # The jaws are at their stop, so this is not a grip the
                    # policy can ever undo: contact + release_epsilon clips
                    # back onto the stop. Refuse rather than weld an object to
                    # a hand that is, mechanically, wide open.
                    _rule_grasp_event(state,"grasp_skipped", hand, frame, substep,
                                      "release_edge_unreachable_at_jaw_stop",
                                      intent=effort,contact_sides=contacts,predicate=predicate)
                    continue
                if not _has_free_joint_ancestor(state,contacts.get("object_body_id")):
                    # The pads agree on a body that no free joint can move.
                    # Welding a static or robot body to the EEF would
                    # overconstrain the model, so refuse and record why.
                    _rule_grasp_event(state,"grasp_skipped", hand, frame, substep,
                                      "grasped_body_has_no_free_joint",
                                      intent=effort,contact_sides=contacts,predicate=predicate)
                    continue
                # No weld yet. The jaws keep closing and ordinary contact
                # settles the object; _apply_rule_grasp_hold welds once they
                # stop making progress.
                state.RULE_GRASP.update({"active":True,
                                   "hand":hand,
                                   "weld_eq_id":None,
                                   "welded_finger_q":None,
                                   "closing_started_sec":None,
                                   "closing_last_sec":None,
                                   "closing_stalled_sec":0.0,
                                   "closing_lost_contact_sec":0.0,
                                   "closing_opposed_sec":0.0,
                                   # The ratchet's clock belongs to this grasp,
                                   # not the last one: a stale reading would
                                   # hand the first clamp a second's allowance.
                                   "ratchet_last_sec":None,
                                   "closing_last_rad":None,
                                   "pinned_finger_q":_finger_qpos(state,hand),
                                   # A fresh list: the ratchet is mutated in
                                   # place as the jaws close, so it must not
                                   # alias the latch angles kept for telemetry.
                                   "ratchet_finger_q":list(_finger_qpos(state,hand)),
                                   "object_body_id":contacts.get("object_body_id"),
                                   "object_body":contacts.get("object_body"),
                                   "contact_angles_rad":dict(actual_angles),
                                   "policy_target_at_grasp_rad":target})
                _rule_grasp_event(state,"grasp", hand, frame, substep,
                                  "both_inner_pads_same_object_and_policy_below_contact_angles",
                                  intent=effort,contact_sides=contacts,predicate=predicate)
                # Pinned the moment it is caught, not once the jaws settle.
                #
                # Deferring the weld gave contact a window to turn a badly
                # caught object square, and for a pen that worked. For the
                # tape it did the opposite: a roll is free to move in that
                # window, and it rolls straight back out of the jaws. Across
                # twenty box_tape rollouts the hand latched 72 times and gave
                # up 48 of them, every one to "lost both pads before weld",
                # and the median lift fell from 113 mm to 102 mm.
                #
                # So the weld goes on at the latch. At torquescale 0 it holds
                # the object's origin at a fixed point in the hand and leaves
                # its orientation entirely to the solver, so everything the
                # closing phase was for still happens -- the jaws keep closing
                # under the ratchet, contact still turns a crooked pen square
                # -- and the one thing that cannot happen is the object
                # translating out of the grip.
                state.RULE_GRASP["weld_eq_id"]=_activate_rule_grasp_weld(
                    state,hand,state.RULE_GRASP["object_body_id"])
                state.RULE_GRASP["welded_finger_q"]=_finger_qpos(state,hand)
                _rule_grasp_event(state,"weld", hand, frame, substep,
                                  "pinned_at_latch_rotation_left_free")
                _apply_rule_grasp_hold(state,target,frame,substep,contacts)
                if state.A.state_dump_on_rule_grasp:
                    # Not while holding the object up. The dump serialises
                    # ``xfrc_applied`` and restores it, but it does not restore
                    # the rule-grasp state machine -- so a dump taken here would
                    # reload as a support force with no grasp that owns it, and
                    # the object would hang in the air with nothing to explain
                    # it. The next update re-applies the carry a step later.
                    _drop_carried_weight(state)
                    event_idx=sum(1 for ev in state.RULE_GRASP["events"] if ev.get("event")=="grasp")
                    state._dump_state_once(state,f"rule_grasp_{hand}_event{event_idx:04d}", frame)
                break
    state.RULE_GRASP["last"]=last
    return _rule_grasp_status(state)

def _new_rule_grasp_state():
    # Built from the clearing helper plus the two fields that deliberately
    # outlive a reset, so a field added to one can never be missing from the
    # other -- a missing key reads as None, which is neither False nor 0.0.
    return {**_cleared_rule_grasp_fields(),
            "events":[],"last":{},
            # Deliberately not in _cleared_rule_grasp_fields: if an exit ever
            # forgets to drop the weight, the record has to outlive the reset
            # so the next update can find the body and clear it. Same for the
            # refusal, which exists precisely to outlive the reset.
            "carried_body_id":None,
            "refused_body_id":{}}

def _rule_grasp_status(state):
    return {"mode":state.A.grasp_mode,
            "hold_mode":(None if state.A.grasp_mode!="rule"
                         else ("contact_only"
                               if (state.RULE_GRASP.get("active")
                                   and state.RULE_GRASP.get("weld_eq_id") is None)
                               else "weld_equality")),
            # Which of the two weld modes is holding it. Without this a frame
            # cannot say whether the object was still free to turn.
            "rotation_pinned":bool(state.RULE_GRASP.get("rotation_pinned",False)),
            "seating_settled_sec":round(
                float(state.RULE_GRASP.get("seating_settled_sec",0.0)),4),
            "active":bool(state.RULE_GRASP.get("active",False)),
            "hand":state.RULE_GRASP.get("hand"),
            "weld_eq_id":state.RULE_GRASP.get("weld_eq_id"),
            "object_body":state.RULE_GRASP.get("object_body"),
            "pinned_finger_q":state.RULE_GRASP.get("pinned_finger_q"),
            "ratchet_finger_q":state.RULE_GRASP.get("ratchet_finger_q"),
            "welded_finger_q":state.RULE_GRASP.get("welded_finger_q"),
            "phase":("welded" if state.RULE_GRASP.get("weld_eq_id") is not None
                     else ("closing" if state.RULE_GRASP.get("active") else None)),
            "closing_started_sec":state.RULE_GRASP.get("closing_started_sec"),
            "closing_stalled_sec":state.RULE_GRASP.get("closing_stalled_sec"),
            "object_body_id":state.RULE_GRASP.get("object_body_id"),
            "predicate":"opposed_pinch_seated_v2",
            "contact_angles_rad":state.RULE_GRASP.get("contact_angles_rad"),
            "policy_target_at_grasp_rad":state.RULE_GRASP.get("policy_target_at_grasp_rad"),
            "last":state.RULE_GRASP.get("last",{})}

def _assert_rule_grasp_pristine(state,context):
    failures=[]
    expected={
        "active":False,
        "hand":None,
        "weld_eq_id":None,
        "pinned_finger_q":None,
        "ratchet_finger_q":None,
        "welded_finger_q":None,
        "closing_started_sec":None,
        "closing_last_sec":None,
        "closing_stalled_sec":0.0,
        "closing_lost_contact_sec":0.0,
        "closing_opposed_sec":0.0,
        "closing_last_rad":None,
        "carried_body_id":None,
        # refused_body_id is checked below: None is "no refusal" as well as absent.
        "rotation_pinned":False,
        "seating_settled_sec":0.0,
        "seating_last_rad":None,
        "seating_last_sec":None,
        "seating_started_sec":None,
        "seating_valid_sec":0.0,
        "stopped_finger_q":None,
        "ratchet_last_sec":None,
        "object_body_id":None,
        "object_body":None,
        "events":[],
        "last":{},
        "contact_angles_rad":None,
        "policy_target_at_grasp_rad":None,
    }
    for name,value in expected.items():
        if state.RULE_GRASP.get(name)!=value:
            failures.append(name)
    # A hand with no refusal is recorded as None (released) or absent (never
    # refused); either is pristine. Only a live refusal is not.
    if any(v is not None for v in (state.RULE_GRASP.get("refused_body_id") or {}).values()):
        failures.append("refused_body_id")
    # Not just the bookkeeping: the force itself. A support force that outlives
    # its grasp holds an object in mid-air and says nothing about it in any
    # summary, so the one place that declares "no grasp is in progress" is the
    # right place to insist on it. A run that applies its own body forces for
    # other reasons would need this narrowed to the bodies rule grasp touches.
    applied=getattr(state.d,"xfrc_applied",None)
    if applied is not None and float(np.abs(np.asarray(applied)).max(initial=0.0))>0.0:
        failures.append("xfrc_applied_not_zero")
    for hand in ("R","L"):
        try:
            eq_id=_rule_grasp_weld_eq_id(state,hand)
        except RuntimeError:
            failures.append(f"{RULE_GRASP_WELD_NAMES[hand]}_missing")
            continue
        if bool(state.d.eq_active[eq_id]):
            failures.append(f"{RULE_GRASP_WELD_NAMES[hand]}_active")
    if failures:
        raise RuntimeError(
            f"rule-grasp state is not pristine during {context}: {failures}"
        )
