using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Modules.SoftPhysics.SoftMesh
{
    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshCollider))]
    public class SoftMesh : MonoBehaviour
    {
        [Header("Compute")]
        [SerializeField]
        private ComputeShader compute;

        [Header("Impact")]
        [SerializeField]
        private float minImpactImpulse = 0.5f;

        [SerializeField]
        private float dent = 0.05f;

        [SerializeField]
        private float kick = 1.5f;

        [SerializeField]
        private float maxDistance = 1.0f;

        [SerializeField]
        private LayerMask deformLayers;

        [Header("Spring")]
        [SerializeField]
        private float springStiffness = 50f;

        [SerializeField]
        private float damping = 8f;

        [Header("Collider")]
        [SerializeField]
        private bool updateCollider = true;

        [SerializeField]
        private int updateColliderEveryNFrames = 5;

        [Header("Boundary Mask")]
        [SerializeField]
        private bool useBoundaryMask = true;

        [SerializeField]
        private AnimationCurve falloffCurve = AnimationCurve.EaseInOut(0f, 0f, 1f, 1f);

        [SerializeField]
        private float[] boundaryWeights;

        [Header("Debug")]
        public bool reset;

        private readonly uint[] _activeFlagCPU = new uint[1];
        private static readonly int Vel = Shader.PropertyToID("_Vel");
        private static readonly int ActiveFlag = Shader.PropertyToID("_ActiveFlag");
        private static readonly int VertexCount = Shader.PropertyToID("_VertexCount");
        private static readonly int BoundaryWeights = Shader.PropertyToID("_BoundaryWeights");
        private static readonly int UseBoundaryMask = Shader.PropertyToID("_UseBoundaryMask");
        private static readonly int DT = Shader.PropertyToID("_DT");
        private static readonly int SpringK = Shader.PropertyToID("_SpringK");
        private static readonly int Damping = Shader.PropertyToID("_Damping");
        private static readonly int ImpactPointLs = Shader.PropertyToID("_ImpactPointLS");
        private static readonly int PushDirLs = Shader.PropertyToID("_PushDirLS");
        private static readonly int AbsScale = Shader.PropertyToID("_AbsScale");
        private static readonly int Kick = Shader.PropertyToID("_Kick");
        private static readonly int Dent = Shader.PropertyToID("_Dent");
        private static readonly int Rest = Shader.PropertyToID("_Rest");
        private static readonly int Target = Shader.PropertyToID("_Target");
        private static readonly int Pos = Shader.PropertyToID("_Pos");
        private static readonly int MaxDistanceWs = Shader.PropertyToID("_MaxDistanceWS");

        // Sleep / activity flag
        private bool _inactive;
        private MeshFilter _mf;
        private MeshCollider _mc;
        private Mesh _mesh;
        private Vector3[] _cpuVerts;
        private int _frameCounter;

        //GPU Resources
        private ComputeBuffer _restBuf,
            _targetBuf,
            _posBuf,
            _velBuf,
            _activeFlagBuf,
            _boundaryWeightBuf;
        private int _kSimulate,
            _kImpact,
            _kCheckActivity; // Kernels
        private int _vertexCount;

        private void Awake()
        {
            _mf = GetComponent<MeshFilter>();
            _mc = GetComponent<MeshCollider>();

            //One instance per SoftMesh otherwise they share the same buffer bindings!
            compute = Instantiate(compute);

            _mesh = Instantiate(_mf.sharedMesh);
            _mesh.MarkDynamic();
            _mf.sharedMesh = _mesh;
            _mc.sharedMesh = _mesh;

            var rest = _mesh.vertices;
            _vertexCount = rest.Length;

            _cpuVerts = new Vector3[_vertexCount];

            // Create Buffers
            _restBuf = new ComputeBuffer(_vertexCount, sizeof(float) * 3);
            _posBuf = new ComputeBuffer(_vertexCount, sizeof(float) * 3);
            _velBuf = new ComputeBuffer(_vertexCount, sizeof(float) * 3);
            _targetBuf = new ComputeBuffer(_vertexCount, sizeof(float) * 3);
            _restBuf.SetData(rest);
            _targetBuf.SetData(rest);
            _posBuf.SetData(rest);
            _velBuf.SetData(new Vector3[_vertexCount]);

            // Find Kernels
            _kSimulate = compute.FindKernel("Simulate");
            _kImpact = compute.FindKernel("ApplyImpact");
            _kCheckActivity = compute.FindKernel("CheckActivity");

            // Bind Buffers to Kernels
            BindCommon(_kSimulate);
            BindCommon(_kImpact);

            // 1-uint activity flag buffer
            _activeFlagBuf = new ComputeBuffer(1, sizeof(uint));
            _activeFlagBuf.SetData(new uint[] { 0 });

            // Bind to kernel
            compute.SetBuffer(_kCheckActivity, Vel, _velBuf);
            compute.SetBuffer(_kCheckActivity, ActiveFlag, _activeFlagBuf);
            compute.SetInt(VertexCount, _vertexCount);

            // Boundary mask: reuse serialized weights if valid, otherwise compute from scratch
            if (
                useBoundaryMask
                && (boundaryWeights == null || boundaryWeights.Length != _vertexCount)
            )
                boundaryWeights = ComputeGeodesicBoundaryWeights(_mesh);

            var bw = useBoundaryMask ? boundaryWeights : MakeOnes(_vertexCount);
            _boundaryWeightBuf = new ComputeBuffer(_vertexCount, sizeof(float));
            _boundaryWeightBuf.SetData(bw);
            compute.SetBuffer(_kImpact, BoundaryWeights, _boundaryWeightBuf);
            compute.SetInt(UseBoundaryMask, useBoundaryMask ? 1 : 0);
        }

        private void OnDestroy()
        {
            _restBuf?.Release();
            _targetBuf?.Release();
            _posBuf?.Release();
            _velBuf?.Release();
            _activeFlagBuf?.Release();
            _boundaryWeightBuf?.Release();
        }

        private void FixedUpdate()
        {
            _frameCounter++;

            if (reset)
            {
                ResetState();
                reset = false;
                _inactive = false;
            }

            if (_inactive)
                return;

            compute.SetFloat(DT, Time.fixedDeltaTime);
            compute.SetFloat(SpringK, springStiffness);
            compute.SetFloat(Damping, damping);
            Dispatch(_kSimulate);

            _activeFlagCPU[0] = 0;
            _activeFlagBuf.SetData(_activeFlagCPU);
            Dispatch(_kCheckActivity);
            _activeFlagBuf.GetData(_activeFlagCPU);
            _inactive = (_activeFlagCPU[0] == 0);

            if (_inactive)
                return;

            _posBuf.GetData(_cpuVerts);

            _mesh.vertices = _cpuVerts;
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();

            if (updateCollider && _mc)
            {
                var n = Mathf.Max(1, updateColliderEveryNFrames);

                if (_frameCounter % n == 0)
                {
                    _mc.sharedMesh = null;
                    _mc.sharedMesh = _mesh;
                }
            }
        }

        // OnCollisionEnter applies an impact to the mesh on collision. Very slow on large meshes!
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
            var pushDirWs = -cp.normal.normalized;

            //float kick = kick * c.impulse.magnitude;
            ApplyImpactGPU(impactPointWs, pushDirWs, kick, dent);
        }

        private void ApplyImpactGPU(
            Vector3 impactPointWs,
            Vector3 pushDirWs,
            float kickValue,
            float dentValue
        )
        {
            _inactive = false;

            // Convert impact point from world space to local space
            var impactPointLs = transform.InverseTransformPoint(impactPointWs);

            // Convert direction from world space to local space
            var pushDirLs = transform.InverseTransformDirection(pushDirWs).normalized;

            // Absolute lossy scale used to approximate world-space distance
            // from local-space delta (handles non-uniform scale)
            var s = transform.lossyScale;
            var absScale = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));

            compute.SetVector(ImpactPointLs, impactPointLs);
            compute.SetVector(PushDirLs, pushDirLs);

            // Scale correction so MaxDistance remains in world units
            compute.SetVector(nameID: AbsScale, absScale);
            compute.SetFloat(MaxDistanceWs, maxDistance);

            compute.SetFloat(Kick, kickValue);
            compute.SetFloat(Dent, dentValue);

            Dispatch(_kImpact);
        }

        private void ResetState()
        {
            _restBuf.GetData(_cpuVerts);

            _targetBuf.SetData(_cpuVerts); // Target = Rest
            _posBuf.SetData(_cpuVerts); // Pos = Rest
            _velBuf.SetData(new Vector3[_vertexCount]); // Vel = 0

            _mesh.vertices = _cpuVerts;
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();

            if (_mc)
            {
                _mc.sharedMesh = null;
                _mc.sharedMesh = _mesh;
            }
        }

        private void Dispatch(int kernel)
        {
            var groups = Mathf.CeilToInt(_vertexCount / 256f);
            compute.Dispatch(kernel, groups, 1, 1);
        }

        private void BindCommon(int kernel)
        {
            compute.SetBuffer(kernel, Rest, _restBuf);
            compute.SetBuffer(kernel, Target, _targetBuf);
            compute.SetBuffer(kernel, Pos, _posBuf);
            compute.SetBuffer(kernel, Vel, _velBuf);
            compute.SetInt(VertexCount, _vertexCount);
        }

        // Recomputes geodesic weights and re-uploads to GPU.
        // Available via right-click on the component in the Inspector.
        [ContextMenu("Recompute Boundary Weights")]
        private void RecomputeBoundaryWeights()
        {
            var sourceMesh = Application.isPlaying ? _mesh : GetComponent<MeshFilter>()?.sharedMesh;
            if (!sourceMesh)
                return;

            boundaryWeights = ComputeGeodesicBoundaryWeights(sourceMesh);

            if (Application.isPlaying && _boundaryWeightBuf != null)
                _boundaryWeightBuf.SetData(boundaryWeights);
        }

        // -------------------------------------------------------------------------
        // Boundary Mask: geodesic distance via multi-source Dijkstra
        // Runs once at Awake; zero CPU cost at runtime.
        // -------------------------------------------------------------------------

        private float[] ComputeGeodesicBoundaryWeights(Mesh m)
        {
            var tris = m.triangles;
            var verts = m.vertices;
            var n = verts.Length;

            // Weld co-located vertices so UV-seam duplicates don't appear as false boundaries
            var weld = WeldByPosition(verts);

            // Count undirected edges by welded index; a boundary edge appears exactly once
            var edgeCount = new Dictionary<(int, int), int>();
            for (var t = 0; t < tris.Length; t += 3)
            {
                CountEdge(edgeCount, weld[tris[t]], weld[tris[t + 1]]);
                CountEdge(edgeCount, weld[tris[t + 1]], weld[tris[t + 2]]);
                CountEdge(edgeCount, weld[tris[t + 2]], weld[tris[t]]);
            }

            var boundaryWelded = new HashSet<int>();
            foreach (var kv in edgeCount.Where(kv => kv.Value == 1))
            {
                boundaryWelded.Add(kv.Key.Item1);
                boundaryWelded.Add(kv.Key.Item2);
            }

            if (boundaryWelded.Count == 0)
            {
                Debug.LogWarning(
                    "[SoftMesh] useBoundaryMask: no boundary edges found (closed mesh?). Mask has no effect."
                );
                return MakeOnes(n);
            }

            // Build adjacency list over original (unwelded) vertex indices
            var adj = new List<(int v, float d)>[n];
            for (var i = 0; i < n; i++)
                adj[i] = new List<(int, float)>();
            for (var t = 0; t < tris.Length; t += 3)
            {
                int a = tris[t],
                    b = tris[t + 1],
                    c = tris[t + 2];
                AddAdj(adj, a, b, verts);
                AddAdj(adj, b, c, verts);
                AddAdj(adj, c, a, verts);
            }

            // Zero-cost edges between welded pairs so geodesic can cross UV seams freely
            var weldGroups = new Dictionary<int, List<int>>();
            for (var i = 0; i < n; i++)
            {
                if (!weldGroups.TryGetValue(weld[i], out var g))
                    weldGroups[weld[i]] = g = new List<int>();
                g.Add(i);
            }
            foreach (var g in weldGroups.Values)
            {
                for (var a = 0; a < g.Count; a++)
                for (var b = a + 1; b < g.Count; b++)
                {
                    adj[g[a]].Add((g[b], 0f));
                    adj[g[b]].Add((g[a], 0f));
                }
            }

            // Multi-source Dijkstra from all boundary vertices (O(n²), runs once at startup)
            var dist = new float[n];
            var visited = new bool[n];
            for (var i = 0; i < n; i++)
                dist[i] = float.MaxValue;
            for (var i = 0; i < n; i++)
                if (boundaryWelded.Contains(weld[i]))
                    dist[i] = 0f;

            for (var iter = 0; iter < n; iter++)
            {
                var u = -1;
                var minD = float.MaxValue;
                for (var i = 0; i < n; i++)
                    if (!visited[i] && dist[i] < minD)
                    {
                        minD = dist[i];
                        u = i;
                    }
                if (u < 0)
                    break;
                visited[u] = true;
                foreach (var (v, d) in adj[u])
                {
                    var nd = dist[u] + d;
                    if (nd < dist[v])
                        dist[v] = nd;
                }
            }

            // Normalize to [0..1] and apply smoothstep for a soft falloff
            var maxD = 0f;
            for (var i = 0; i < n; i++)
                if (dist[i] < float.MaxValue)
                    maxD = Mathf.Max(maxD, dist[i]);

            // Degenerate case: every vertex is on the boundary
            if (maxD < 1e-6f)
                return MakeOnes(n);

            var weights = new float[n];
            for (var i = 0; i < n; i++)
            {
                var t = Mathf.Clamp01(dist[i] / maxD);
                weights[i] = Mathf.Clamp01(falloffCurve.Evaluate(t));
            }
            return weights;
        }

        // Group vertices by rounded position; tolerance = 0.00001 world units
        private static int[] WeldByPosition(Vector3[] verts)
        {
            const float invEps = 100000f;
            var n = verts.Length;
            var welded = new int[n];
            var map = new Dictionary<(int, int, int), int>(n);
            var count = 0;
            for (var i = 0; i < n; i++)
            {
                var v = verts[i];
                var key = (
                    Mathf.RoundToInt(v.x * invEps),
                    Mathf.RoundToInt(v.y * invEps),
                    Mathf.RoundToInt(v.z * invEps)
                );
                if (!map.TryGetValue(key, out var id))
                {
                    id = count++;
                    map[key] = id;
                }
                welded[i] = id;
            }
            return welded;
        }

        private static void CountEdge(Dictionary<(int, int), int> d, int a, int b)
        {
            var key = a < b ? (a, b) : (b, a);
            d.TryGetValue(key, out var c);
            d[key] = c + 1;
        }

        private static void AddAdj(List<(int, float)>[] adj, int a, int b, Vector3[] verts)
        {
            var d = Vector3.Distance(verts[a], verts[b]);
            adj[a].Add((b, d));
            adj[b].Add((a, d));
        }

        private static float[] MakeOnes(int n)
        {
            var a = new float[n];
            for (var i = 0; i < n; i++)
                a[i] = 1f;
            return a;
        }
    }
}
