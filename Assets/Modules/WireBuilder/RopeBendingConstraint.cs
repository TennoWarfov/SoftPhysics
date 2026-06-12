using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

[RequireComponent(typeof(WireController))]
public class RopeBendingConstraint : MonoBehaviour
{
    [Header("Joint Angular Drives")]
    [Tooltip(
        "Strengthens the angular spring on all segment ConfigurableJoints at startup, resisting rotation from the initial straight pose."
    )]
    public bool enhanceJointDrives = true;

    [Min(0f)]
    public float angularSpring = 500f;

    [Min(0f)]
    public float angularDamper = 50f;

    [Header("Position Straightening")]
    [Tooltip(
        "Each FixedUpdate, applies a spring force pulling each segment toward the midpoint of its neighbours, resisting sag and bending."
    )]
    public bool applyPositionConstraint = true;

    [Min(0f)]
    public float positionSpring = 100f;

    [Min(0f)]
    public float positionDamper = 10f;

    [Header("Solver Iterations")]
    [Tooltip(
        "Overrides per-Rigidbody solver iterations for more accurate joint resolution along the chain."
    )]
    public bool overrideSolverIterations = true;

    [Range(1, 255)]
    public int solverIterations = 20;

    [Range(1, 255)]
    public int solverVelocityIterations = 10;

    private WireController _wire;
    private List<Rigidbody> _bodies = new List<Rigidbody>();

    private async void Start()
    {
        _wire = GetComponent<WireController>();

        if (!_wire.usePhysics)
        {
            enabled = false;
            return;
        }

        CacheRigidbodies();

        if (enhanceJointDrives)
            ApplyAngularDrives();

        if (overrideSolverIterations)
            ApplySolverIterations();

        await Task.Delay(10);
        RefreshConstraints();
    }

    // Call this after the rope is rebuilt at runtime to re-apply all constraints.
    public void RefreshConstraints()
    {
        if (_wire == null)
            _wire = GetComponent<WireController>();

        CacheRigidbodies();

        if (enhanceJointDrives)
            ApplyAngularDrives();

        if (overrideSolverIterations)
            ApplySolverIterations();
    }

    private void CacheRigidbodies()
    {
        _bodies.Clear();
        foreach (Transform seg in _wire.segments)
        {
            Rigidbody rb = seg.GetComponent<Rigidbody>();
            if (rb != null)
                _bodies.Add(rb);
        }
    }

    private void ApplyAngularDrives()
    {
        JointDrive drive = new JointDrive
        {
            positionSpring = angularSpring,
            positionDamper = angularDamper,
            maximumForce = Mathf.Infinity,
        };

        foreach (Transform seg in _wire.segments)
        {
            foreach (ConfigurableJoint joint in seg.GetComponents<ConfigurableJoint>())
            {
                // XYAndZ mode lets angularXDrive and angularYZDrive both contribute.
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularXDrive = drive;
                joint.angularYZDrive = drive;
            }
        }
    }

    private void ApplySolverIterations()
    {
        foreach (Rigidbody rb in _bodies)
        {
            rb.solverIterations = solverIterations;
            rb.solverVelocityIterations = solverVelocityIterations;
        }
    }

    private void FixedUpdate()
    {
        if (!applyPositionConstraint || _bodies.Count < 3)
            return;

        // Skip first and last bodies — they are directly constrained by joints to the anchors.
        for (int i = 1; i < _bodies.Count - 1; i++)
        {
            Rigidbody prev = _bodies[i - 1];
            Rigidbody curr = _bodies[i];
            Rigidbody next = _bodies[i + 1];

            // Pull curr toward the midpoint of its neighbours (straightening spring).
            Vector3 target = (prev.position + next.position) * 0.5f;
            Vector3 displacement = target - curr.position;

            // Damp relative to the average neighbour velocity to avoid oscillation.
            Vector3 relativeVelocity =
                curr.linearVelocity - (prev.linearVelocity + next.linearVelocity) * 0.5f;

            curr.AddForce(
                displacement * positionSpring - relativeVelocity * positionDamper,
                ForceMode.Force
            );
        }
    }
}
