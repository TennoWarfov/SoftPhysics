using System.Threading.Tasks;
using UnityEngine;

namespace Modules.SoftPhysics.SoftMesh
{
    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshCollider))]
    public class SoftMeshLight : MonoBehaviour
    {
        [SerializeField]
        private float minImpactImpulse = 0.5f;

        [SerializeField]
        private float strength = 0.05f;

        [SerializeField]
        private float maxDistance = 1.0f;

        [SerializeField]
        private bool updateCollider = true;

        [SerializeField]
        private LayerMask deformLayers;

        [SerializeField]
        private bool reset;

        private MeshFilter _mf;
        private MeshCollider _mc;
        private Mesh _mesh;
        private Vector3[] _originalVertices;
        private Vector3[] _deformedVertices;

        // worker result
        private volatile bool _hasResult;
        private Vector3[] _resultVertices;

        private void Awake()
        {
            _mf = GetComponent<MeshFilter>();
            _mc = GetComponent<MeshCollider>();

            _mesh = Instantiate(_mf.sharedMesh);
            _mf.sharedMesh = _mesh;

            _originalVertices = _mesh.vertices;
            _deformedVertices = (Vector3[])_originalVertices.Clone();

            _mc.sharedMesh = _mesh;
        }

        private void Update()
        {
            if (reset)
            {
                _deformedVertices = (Vector3[])_originalVertices.Clone();
                ApplyToMesh(_deformedVertices);
                reset = false;
            }

            if (_hasResult)
            {
                _hasResult = false;
                _deformedVertices = _resultVertices;
                ApplyToMesh(_deformedVertices);
            }
        }

        private void OnCollisionEnter(Collision c)
        {
            if (c.contactCount == 0)
                return;
            if (c.impulse.magnitude < minImpactImpulse)
                return;
            if ((deformLayers.value & (1 << c.gameObject.layer)) == 0)
                return;

            var cp = c.GetContact(0);

            var impactPointWs = cp.point;
            var impactNormalWs = -cp.normal;

            DeformMeshAsync(impactPointWs, impactNormalWs);
        }

        //Deform mesh in a background thread
        public void DeformMeshAsync(
            Vector3 impactPointWs,
            Vector3 impactNormalWs,
            float strengthValue,
            float maxDistanceValue
        )
        {
            // snapshot everything needed (DON'T touch transform/mesh in worker)
            var l2W = transform.localToWorldMatrix;
            var w2L = transform.worldToLocalMatrix;
            var nWs = impactNormalWs.normalized;

            var baseVerts = (Vector3[])_deformedVertices.Clone();

            Task.Run(() =>
            {
                for (var i = 0; i < baseVerts.Length; i++)
                {
                    var vWs = l2W.MultiplyPoint3x4(baseVerts[i]);
                    var dist = Vector3.Distance(vWs, impactPointWs);
                    if (dist > maxDistanceValue)
                        continue;

                    var t = Mathf.Clamp01(dist / maxDistanceValue);
                    var w = Mathf.SmoothStep(1f, 0f, t);
                    var vWsDef = vWs - nWs * (strengthValue * w);

                    baseVerts[i] = w2L.MultiplyPoint3x4(vWsDef);
                }

                _resultVertices = baseVerts;
                _hasResult = true;
            });
        }

        //Comfort overload
        public void DeformMeshAsync(Vector3 impactPointWs, Vector3 impactNormalWs)
        {
            DeformMeshAsync(impactPointWs, impactNormalWs, strength, maxDistance);
        }

        private void ApplyToMesh(Vector3[] verts)
        {
            _mesh.vertices = verts;
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();

            if (updateCollider)
            {
                _mc.sharedMesh = null;
                _mc.sharedMesh = _mesh;
            }
        }
    }
}
