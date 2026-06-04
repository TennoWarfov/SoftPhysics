using System.Collections.Generic;
using UnityEngine;

namespace Modules.SoftPhysics
{
    public class PhysicallyHand : MonoBehaviour
    {
        [SerializeField]
        private Transform[] originalBones;

        [SerializeField]
        private Transform[] cloneBones;

        [SerializeField]
        private float positionGain = 30f;

        [SerializeField]
        private float rotationGain = 30f;

        private List<Rigidbody> _bones;

        private void Awake()
        {
            _bones = new List<Rigidbody>();
            foreach (var bone in cloneBones)
            {
                var rb = bone.gameObject.AddComponent<Rigidbody>();
                rb.useGravity = false;
                _bones.Add(rb);
            }

            var colliders = GetComponentsInChildren<Collider>();
            for (var i = 0; i < colliders.Length; i++)
            for (var j = i + 1; j < colliders.Length; j++)
                if (colliders[i] && colliders[j])
                    Physics.IgnoreCollision(colliders[i], colliders[j], true);
        }

        private void FixedUpdate()
        {
            for (var i = 0; i < Mathf.Min(originalBones.Length, _bones.Count); i++)
                FollowPose(_bones[i], originalBones[i]);
        }

        private void FollowPose(Rigidbody rb, Transform target)
        {
            var positionError = target.position - rb.position;
            rb.linearVelocity = positionError * positionGain;

            var rotationError = target.rotation * Quaternion.Inverse(rb.rotation);

            rotationError.ToAngleAxis(out var angle, out var axis);

            if (angle > 180f)
                angle -= 360f;

            if (Mathf.Abs(angle) > 0.01f)
            {
                rb.angularVelocity = axis.normalized * angle * Mathf.Deg2Rad * rotationGain;
            }
            else
            {
                rb.angularVelocity = Vector3.zero;
            }
        }
    }
}
