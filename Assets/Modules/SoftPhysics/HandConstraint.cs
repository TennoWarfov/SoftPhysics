using System.Collections.Generic;
using UnityEngine;

namespace Modules.SoftPhysics
{
    // Constrains a proxy hand root against an abdomen point cloud surface.
    //
    // Architecture:
    //   - targetHandRoot     : the raw XR-tracked hand (drives desired pose)
    //   - constrainedHandRoot: the proxy hand visible to the patient
    //   - targetBones[]      : XR hand bone transforms (local pose source)
    //   - constrainedBones[] : proxy hand bone transforms (mirrored every frame)
    //
    // Flow each LateUpdate:
    //   1. Mirror bone local poses (hand shape follows XR hand exactly)
    //   2. Blend weight moves 0→1 on trigger enter, 1→0 on trigger exit
    //   3. When blend > 0: project desired root position against point cloud surface
    //   4. Root position is lerped toward the blended target (free vs constrained)
    //
    // Constraint algorithm (SolveForPosition):
    //   - Query K nearest surface points within searchRadius (spatial hash, O(1) avg)
    //   - Inverse-distance weight their positions and normals → local surface estimate
    //   - Signed distance from weighted surface plane determines penetration depth
    //   - SmoothStep resistance curve: 0 at surface contact, 1 at maxPenetrationDepth
    //   - Lerp between desiredPos and surfacePos by resistance → soft tissue feel
    public class HandConstraint : MonoBehaviour
    {
        [Header("Hands")]
        [SerializeField]
        private Transform targetHandRoot;

        [SerializeField]
        private Transform constrainedHandRoot;

        [SerializeField]
        private Transform[] targetBones;

        [SerializeField]
        private Transform[] constrainedBones;

        [Header("Point Cloud")]
        [SerializeField]
        private MeshPointCloudGenerator pointCloud;

        [Header("Constraint")]
        [Tooltip("Approximate radius of the hand proxy sphere at the palm center.")]
        [SerializeField]
        private float handRadius = 0.03f;

        [Tooltip("Radius within which surface points are gathered for the local normal estimate.")]
        [SerializeField]
        private float searchRadius = 0.10f;

        [Tooltip("Number of nearest surface points used for weighted normal/position estimate.")]
        [SerializeField]
        private int kNeighbors = 8;

        [Tooltip(
            "Penetration depth at which the constraint reaches full resistance (100% blocked)."
        )]
        [SerializeField]
        private float maxPenetrationDepth = 0.04f;

        [Header("Follow")]
        [SerializeField]
        private float positionFollowSpeed = 25f;

        [SerializeField]
        private float rotationFollowSpeed = 20f;

        [Tooltip("Speed at which the free↔constrained blend transitions on trigger enter/exit.")]
        [SerializeField]
        private float blendSpeed = 6f;

        [Tooltip(
            "How strongly the rotation constraint aligns Z to the surface tangent plane. 0 = free rotation, 1 = fully clamped."
        )]
        [Range(0f, 1f)]
        [SerializeField]
        private float rotationConstraintStrength = 1f;

        // ── Runtime state ─────────────────────────────────────────────────────

        private bool _insideConstraintVolume;
        private float _blendWeight;

        // Pre-allocated to avoid per-frame GC; KNearestResult is a value type.
        private readonly List<KNearestResult> _knnResults = new(16);

        private Vector3 _surfaceNormal = Vector3.up;
        private Vector3 _surfaceTangent = Vector3.forward;

        // ── Trigger detection ─────────────────────────────────────────────────

        private void OnTriggerEnter(Collider other)
        {
            if (other.TryGetComponent(out MeshPointCloudGenerator _))
                _insideConstraintVolume = true;
        }

        private void OnTriggerExit(Collider other)
        {
            if (other.TryGetComponent(out MeshPointCloudGenerator _))
                _insideConstraintVolume = false;
        }

        // ── Per-frame update ──────────────────────────────────────────────────

        // LateUpdate ensures XR hand poses are fully settled before we read them.
        private void LateUpdate()
        {
            // MirrorBonePoses();

            var blendTarget = _insideConstraintVolume ? 1f : 0f;
            _blendWeight = Mathf.MoveTowards(
                _blendWeight,
                blendTarget,
                blendSpeed * Time.deltaTime
            );

            var desiredPos = targetHandRoot.position;
            var desiredRot = targetHandRoot.rotation;

            var targetPos = desiredPos;

            if (_blendWeight > 0.01f)
            {
                var constrainedPos = SolveForPosition(desiredPos);
                targetPos = Vector3.Lerp(desiredPos, constrainedPos, _blendWeight);
            }

            targetPos += SolveBonesPenetrationCorrection();

            constrainedHandRoot.position = Vector3.Lerp(
                constrainedHandRoot.position,
                targetPos,
                positionFollowSpeed * Time.deltaTime
            );

            var targetRot = desiredRot;
            // if (_blendWeight > 0.01f)
            // {
            //     var constrainedRot = SolveForRotation(desiredRot);
            //     targetRot = Quaternion.Slerp(desiredRot, constrainedRot, _blendWeight * rotationConstraintStrength);
            // }

            constrainedHandRoot.rotation = Quaternion.Slerp(
                constrainedHandRoot.rotation,
                targetRot,
                rotationFollowSpeed * Time.deltaTime
            );

            MirrorBonePoses();
            // for (var i = 0; i < constrainedBones.Length; i++)
            // {
            //     var bone = constrainedBones[i];
            //     var desired = targetBones[i].position;
            //     var target = desired;
            //
            //     if (_blendWeight > 0.01f)
            //     {
            //         var constrainedPos = SolveForPosition(desired);
            //         target = Vector3.Lerp(desired, constrainedPos, _blendWeight);
            //     }
            //
            //     bone.position = Vector3.Lerp(
            //         bone.position,
            //         target,
            //         positionFollowSpeed * Time.deltaTime
            //     );
            // }
        }

        // ── Constraint solve ──────────────────────────────────────────────────

        // Returns the constrained world position for the hand root.
        // When not penetrating: returns desiredWorldPos unchanged.
        // When penetrating: progressively pushes back toward surface using a
        // SmoothStep resistance curve that mimics soft tissue compliance.
        private Vector3 SolveForPosition(Vector3 desiredWorldPos)
        {
            // Query in local space of the point cloud transform.
            var localPos = pointCloud.transform.InverseTransformPoint(desiredWorldPos);
            var count = pointCloud.QueryKNearest(localPos, kNeighbors, searchRadius, _knnResults);

            if (count == 0)
                return desiredWorldPos;

            // Inverse-distance weighted surface point and normal (local space).
            var wPoint = Vector3.zero;
            var wNormal = Vector3.zero;
            var wSum = 0f;

            for (var i = 0; i < count; i++)
            {
                var idx = _knnResults[i].Index;
                var w = 1f / (Mathf.Sqrt(_knnResults[i].DistanceSqr) + 1e-4f);
                wPoint += (Vector3)pointCloud.Points[idx] * w;
                wNormal += (Vector3)pointCloud.Normals[idx] * w;
                wSum += w;
            }

            var surfacePointWS = pointCloud.transform.TransformPoint(wPoint / wSum);
            var surfaceNormalWS = pointCloud
                .transform.TransformDirection(wNormal / wSum)
                .normalized;

            _surfaceNormal = surfaceNormalWS;

            // Tangent: direction from nearest to second nearest point, projected onto tangent plane.
            if (count >= 2)
            {
                var p0 = pointCloud.transform.TransformPoint(
                    (Vector3)pointCloud.Points[_knnResults[0].Index]
                );
                var p1 = pointCloud.transform.TransformPoint(
                    (Vector3)pointCloud.Points[_knnResults[1].Index]
                );
                var tangentCandidate = Vector3.ProjectOnPlane(p1 - p0, surfaceNormalWS);
                if (tangentCandidate.sqrMagnitude > 1e-6f)
                    _surfaceTangent = tangentCandidate.normalized;
            }

            // Signed distance: positive means the hand is above the surface.
            // penetration > 0 means the palm sphere is crossing into the surface.
            var signedDist = Vector3.Dot(desiredWorldPos - surfacePointWS, surfaceNormalWS);
            var penetration = handRadius - signedDist;

            if (penetration <= 0f)
                return desiredWorldPos;

            // SmoothStep gives a "compliant tissue" feel:
            //   near surface  → small correction  (soft, easy to press)
            //   deep press    → full correction   (hard stop at maxPenetrationDepth)
            var t = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(penetration / maxPenetrationDepth));
            var correctedPos = surfacePointWS + surfaceNormalWS * handRadius;
            return Vector3.Lerp(desiredWorldPos, correctedPos, t);
        }

        // ── Bone penetration correction ───────────────────────────────────────

        // Checks each constrained bone against the surface and returns the world-space
        // offset to add to constrainedHandRoot so the deepest-penetrating bone is pushed out.
        // Bone positions are from the previous frame (MirrorBonePoses runs after this).
        private Vector3 SolveBonesPenetrationCorrection()
        {
            var maxCorrection = Vector3.zero;
            var maxPenetration = 0f;

            foreach (var bone in constrainedBones)
            {
                var bonePos = bone.position;
                var localPos = pointCloud.transform.InverseTransformPoint(bonePos);
                var count = pointCloud.QueryKNearest(
                    localPos,
                    kNeighbors,
                    searchRadius,
                    _knnResults
                );

                if (count == 0)
                    continue;

                var wPoint = Vector3.zero;
                var wNormal = Vector3.zero;
                var wSum = 0f;

                for (var i = 0; i < count; i++)
                {
                    var idx = _knnResults[i].Index;
                    var w = 1f / (Mathf.Sqrt(_knnResults[i].DistanceSqr) + 1e-4f);
                    wPoint += (Vector3)pointCloud.Points[idx] * w;
                    wNormal += (Vector3)pointCloud.Normals[idx] * w;
                    wSum += w;
                }

                var surfacePointWS = pointCloud.transform.TransformPoint(wPoint / wSum);
                var surfaceNormalWS = pointCloud
                    .transform.TransformDirection(wNormal / wSum)
                    .normalized;

                var signedDist = Vector3.Dot(bonePos - surfacePointWS, surfaceNormalWS);
                var penetration = -signedDist;

                if (penetration > maxPenetration)
                {
                    maxPenetration = penetration;
                    var t = Mathf.SmoothStep(
                        0f,
                        1f,
                        Mathf.Clamp01(penetration / maxPenetrationDepth)
                    );
                    maxCorrection = surfaceNormalWS * (penetration * t);
                }
            }

            return maxCorrection;
        }

        // ── Rotation constraint ───────────────────────────────────────────────

        // Clamps the X (pitch) rotation so the hand's Z axis lies in the surface tangent plane.
        // Projects desiredForward onto the plane perpendicular to _surfaceNormal; falls back
        // to _surfaceTangent when the hand points nearly straight into the surface.
        private Quaternion SolveForRotation(Quaternion desiredRot)
        {
            var forward = Vector3.ProjectOnPlane(desiredRot * Vector3.forward, _surfaceNormal);
            if (forward.sqrMagnitude < 0.001f)
                forward = _surfaceTangent;
            else
                forward.Normalize();
            return Quaternion.LookRotation(forward, desiredRot * Vector3.up);
        }

        // ── Bone pose mirroring ───────────────────────────────────────────────

        // Copies local pose from each target bone to its constrained counterpart.
        // The root constraint offsets the whole proxy hand in world space;
        // bone-local poses preserve hand shape and finger configuration.
        private void MirrorBonePoses()
        {
            var count = Mathf.Min(targetBones.Length, constrainedBones.Length);
            for (var i = 0; i < count; i++)
            {
                constrainedBones[i]
                    .SetLocalPositionAndRotation(
                        targetBones[i].localPosition,
                        targetBones[i].localRotation
                    );
            }
        }
    }
}
