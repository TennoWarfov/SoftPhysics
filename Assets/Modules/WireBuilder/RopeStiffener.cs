using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Makes a ConfigurableJoint chain behave like a stiff plastic or metal rod when grabbed in VR.
/// Add to the same GameObject as WireController.
/// Disable or remove RopeBendingConstraint if present to avoid conflicting joint drives.
/// </summary>
[RequireComponent(typeof(WireController))]
public class RopeStiffener : MonoBehaviour
{
    public enum StiffnessPreset
    {
        Custom,
        Rubber,
        Plastic,
        Metal,
    }

    [Header("Preset")]
    [Tooltip("Applies a set of tuned values. Switch to Custom to edit individual fields.")]
    public StiffnessPreset preset = StiffnessPreset.Metal;

    [Header("Angular Drives — resist bending")]
    [Tooltip(
        "Slerp drive is more numerically stable than XYAndZ at high spring values and drives toward the joint's initial orientation."
    )]
    public bool useSlerpDrive = true;

    [Min(0f)]
    public float angularSpring = 50000f;

    [Min(0f)]
    public float angularDamper = 500f;

    [Header("Linear Joints — resist stretching / compression")]
    [Tooltip(
        "Locks all linear DOF on each joint at its initial anchor offset, preventing segments from drifting apart."
    )]
    public bool lockLinearMotion = true;

    [Tooltip(
        "Used only when lockLinearMotion is false — spring drive resists drift without hard locking."
    )]
    [Min(0f)]
    public float linearSpring = 100000f;

    [Min(0f)]
    public float linearDamper = 1000f;

    [Header("Rigidbody Damping")]
    public bool overrideDamping = true;

    [Min(0f)]
    public float linearDamping = 3f;

    [Min(0f)]
    public float angularDamping = 15f;

    [Header("Solver Iterations")]
    public bool overrideSolverIterations = true;

    [Range(1, 255)]
    public int solverIterations = 60;

    [Range(1, 255)]
    public int solverVelocityIterations = 20;

    [Header("Position Straightening")]
    [Tooltip(
        "Each FixedUpdate, spring-pulls every interior segment toward the midpoint of its neighbours, counteracting compounding joint deflection."
    )]
    public bool applyPositionStraightening = true;

    [Min(0f)]
    public float positionSpring = 5000f;

    [Min(0f)]
    public float positionDamper = 100f;

    private WireController _wire;
    private List<Rigidbody> _bodies = new List<Rigidbody>();
    private List<ConfigurableJoint> _joints = new List<ConfigurableJoint>();

    private void Start()
    {
        _wire = GetComponent<WireController>();
        if (!_wire.usePhysics)
        {
            enabled = false;
            return;
        }

        if (GetComponent<RopeBendingConstraint>() != null)
            Debug.LogWarning(
                "[RopeStiffener] RopeBendingConstraint is also attached — its angular drives will conflict. Disable or remove it.",
                this
            );

        Apply();
    }

    /// <summary>Call this after the rope is rebuilt at runtime to re-apply all constraints.</summary>
    public void Apply()
    {
        if (_wire == null)
            _wire = GetComponent<WireController>();

        if (preset != StiffnessPreset.Custom)
            ApplyPreset();

        CacheBodiesAndJoints();
        ConfigureJoints();
        ConfigureRigidbodies();
    }

    private void ApplyPreset()
    {
        switch (preset)
        {
            case StiffnessPreset.Rubber:
                angularSpring = 500f;
                angularDamper = 50f;
                linearSpring = 10000f;
                linearDamper = 100f;
                linearDamping = 1f;
                angularDamping = 5f;
                positionSpring = 500f;
                positionDamper = 20f;
                solverIterations = 20;
                solverVelocityIterations = 8;
                break;

            case StiffnessPreset.Plastic:
                angularSpring = 10000f;
                angularDamper = 200f;
                linearSpring = 50000f;
                linearDamper = 500f;
                linearDamping = 2f;
                angularDamping = 10f;
                positionSpring = 2000f;
                positionDamper = 50f;
                solverIterations = 40;
                solverVelocityIterations = 15;
                break;

            case StiffnessPreset.Metal:
                angularSpring = 50000f;
                angularDamper = 500f;
                linearSpring = 100000f;
                linearDamper = 1000f;
                linearDamping = 6f;
                angularDamping = 30f;
                positionSpring = 5000f;
                positionDamper = 100f;
                solverIterations = 60;
                solverVelocityIterations = 20;
                break;
        }
    }

    private void CacheBodiesAndJoints()
    {
        _bodies.Clear();
        _joints.Clear();
        foreach (Transform seg in _wire.segments)
        {
            Rigidbody rb = seg.GetComponent<Rigidbody>();
            if (rb != null)
                _bodies.Add(rb);

            foreach (ConfigurableJoint j in seg.GetComponents<ConfigurableJoint>())
                _joints.Add(j);
        }
    }

    private void ConfigureJoints()
    {
        JointDrive angDrive = new JointDrive
        {
            positionSpring = angularSpring,
            positionDamper = angularDamper,
            maximumForce = Mathf.Infinity,
        };

        JointDrive linDrive = new JointDrive
        {
            positionSpring = linearSpring,
            positionDamper = linearDamper,
            maximumForce = Mathf.Infinity,
        };

        foreach (ConfigurableJoint j in _joints)
        {
            // --- Angular stiffness ---
            if (useSlerpDrive)
            {
                j.rotationDriveMode = RotationDriveMode.Slerp;
                j.slerpDrive = angDrive;
            }
            else
            {
                j.rotationDriveMode = RotationDriveMode.XYAndZ;
                j.angularXDrive = angDrive;
                j.angularYZDrive = angDrive;
            }

            // --- Linear stiffness ---
            if (lockLinearMotion)
            {
                // Locks each axis at its current anchor offset — prevents drift, no spring needed.
                j.xMotion = ConfigurableJointMotion.Locked;
                j.yMotion = ConfigurableJointMotion.Locked;
                j.zMotion = ConfigurableJointMotion.Locked;
            }
            else
            {
                // Free motion with a strong spring resisting displacement.
                j.xMotion = ConfigurableJointMotion.Free;
                j.yMotion = ConfigurableJointMotion.Free;
                j.zMotion = ConfigurableJointMotion.Free;
                j.xDrive = linDrive;
                j.yDrive = linDrive;
                j.zDrive = linDrive;
            }
        }
    }

    private void ConfigureRigidbodies()
    {
        foreach (Rigidbody rb in _bodies)
        {
            if (overrideSolverIterations)
            {
                rb.solverIterations = solverIterations;
                rb.solverVelocityIterations = solverVelocityIterations;
            }
            if (overrideDamping)
            {
                rb.linearDamping = linearDamping;
                rb.angularDamping = angularDamping;
            }
        }
    }

    private void FixedUpdate()
    {
        if (!applyPositionStraightening || _bodies.Count < 3)
            return;

        // Skip first and last — they are directly anchored.
        for (int i = 1; i < _bodies.Count - 1; i++)
        {
            Rigidbody prev = _bodies[i - 1];
            Rigidbody curr = _bodies[i];
            Rigidbody next = _bodies[i + 1];

            Vector3 target = (prev.position + next.position) * 0.5f;
            Vector3 displacement = target - curr.position;
            Vector3 relVel =
                curr.linearVelocity - (prev.linearVelocity + next.linearVelocity) * 0.5f;

            curr.AddForce(displacement * positionSpring - relVel * positionDamper, ForceMode.Force);
        }
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        if (preset != StiffnessPreset.Custom)
            ApplyPreset();
    }
#endif
}
