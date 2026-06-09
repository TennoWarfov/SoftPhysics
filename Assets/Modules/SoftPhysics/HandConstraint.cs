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

        // ── Runtime state ─────────────────────────────────────────────────────

        private bool _insideConstraintVolume;
        private float _blendWeight;

        // Pre-allocated to avoid per-frame GC; KNearestResult is a value type.
        private readonly List<KNearestResult> _knnResults = new(16);

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

            constrainedHandRoot.position = Vector3.Lerp(
                constrainedHandRoot.position,
                targetPos,
                positionFollowSpeed * Time.deltaTime
            );

            constrainedHandRoot.rotation = Quaternion.Slerp(
                constrainedHandRoot.rotation,
                desiredRot,
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
