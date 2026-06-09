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

        // ── Fingertip interaction ─────────────────────────────────────────────
        // Assigning any transforms here enables non-physics proximity mode.
        // Leave empty to fall back to the legacy OnCollisionEnter path.

        [Header("Fingertip Interaction")]
        [SerializeField]
        private Transform[] fingertips;

        [SerializeField]
        [Tooltip(
            "Multiplies Time.fixedDeltaTime passed to the spring solver. "
                + "Higher values produce faster response. Instability may appear above ~2 "
                + "depending on springStiffness."
        )]
        private float simulationSpeed = 1f;

        // ── Impact (legacy collision path + shared dent/distance params) ──────

        [Header("Impact")]
        [SerializeField]
        private float minImpactImpulse = 0.5f;

        [SerializeField]
        private float dent = 0.05f;

        [SerializeField]
        private float kick = 1.5f;

        [SerializeField]
        private float maxDistance = 0.08f;

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

        // ── Shader property IDs ───────────────────────────────────────────────

        private readonly uint[] _activeFlagCPU = new uint[1];
        private static readonly int Vel = Shader.PropertyToID("_Vel");
        private static readonly int ActiveFlag = Shader.PropertyToID("_ActiveFlag");
        private static readonly int VertexCount = Shader.PropertyToID("_VertexCount");
        private static readonly int BoundaryWts = Shader.PropertyToID("_BoundaryWeights");
        private static readonly int UseBoundaryMsk = Shader.PropertyToID("_UseBoundaryMask");
        private static readonly int DT = Shader.PropertyToID("_DT");
        private static readonly int SpringK = Shader.PropertyToID("_SpringK");
        private static readonly int Damping = Shader.PropertyToID("_Damping");
        private static readonly int ImpactPointLs = Shader.PropertyToID("_ImpactPointLS");
        private static readonly int PushDirLs = Shader.PropertyToID("_PushDirLS");
        private static readonly int AbsScaleId = Shader.PropertyToID("_AbsScale");
        private static readonly int KickId = Shader.PropertyToID("_Kick");
        private static readonly int DentId = Shader.PropertyToID("_Dent");
        private static readonly int RestId = Shader.PropertyToID("_Rest");
        private static readonly int TargetId = Shader.PropertyToID("_Target");
        private static readonly int PosId = Shader.PropertyToID("_Pos");
        private static readonly int MaxDistWs = Shader.PropertyToID("_MaxDistanceWS");

        // ── State ─────────────────────────────────────────────────────────────

        private MeshFilter _mf;
        private MeshCollider _mc;
        private bool _inactive;
        private Mesh _mesh;
        private Vector3[] _cpuVerts;
        private int _frameCounter;

        // GPU resources
        private ComputeBuffer _restBuf,
            _targetBuf,
            _posBuf,
            _velBuf,
            _activeFlagBuf,
            _boundaryWeightBuf;

        // Kernel handles
        private int _kSimulate,
            _kImpact,
            _kCheckActivity,
            _kResetTargets,
            _kApplyFingertip;

        private int _vertexCount;
        private Bounds _localBounds; // cached for per-finger coarse culling
        private Vector3 _absScale; // cached per-frame to avoid redundant work

        // ── Lifecycle ─────────────────────────────────────────────────────────

        private void Awake()
        {
            _mf = GetComponent<MeshFilter>();
            _mc = GetComponent<MeshCollider>();

            // Per-instance shader so multiple SoftMesh components never share buffers.
            compute = Instantiate(compute);

            _mesh = Instantiate(_mf.sharedMesh);
            _mesh.MarkDynamic();
            _mf.sharedMesh = _mesh;
            _mc.sharedMesh = _mesh;

            var rest = _mesh.vertices;
            _vertexCount = rest.Length;
            _cpuVerts = new Vector3[_vertexCount];
            _localBounds = _mesh.bounds;

            // Allocate GPU buffers
            _restBuf = new ComputeBuffer(_vertexCount, sizeof(float) * 3);
            _posBuf = new ComputeBuffer(_vertexCount, sizeof(float) * 3);
            _velBuf = new ComputeBuffer(_vertexCount, sizeof(float) * 3);
            _targetBuf = new ComputeBuffer(_vertexCount, sizeof(float) * 3);

            _restBuf.SetData(rest);
            _targetBuf.SetData(rest);
            _posBuf.SetData(rest);
            _velBuf.SetData(new Vector3[_vertexCount]);

            // Find kernels
            _kSimulate = compute.FindKernel("Simulate");
            _kImpact = compute.FindKernel("ApplyImpact");
            _kCheckActivity = compute.FindKernel("CheckActivity");
            _kResetTargets = compute.FindKernel("ResetTargets");
            _kApplyFingertip = compute.FindKernel("ApplyFingertip");

            // Bind common buffers to every kernel that needs them
            BindCommon(_kSimulate);
            BindCommon(_kImpact);
            BindCommon(_kResetTargets);
            BindCommon(_kApplyFingertip);

            // Activity flag (1-element uint)
            _activeFlagBuf = new ComputeBuffer(1, sizeof(uint));
            _activeFlagBuf.SetData(new uint[] { 0 });
            compute.SetBuffer(_kCheckActivity, Vel, _velBuf);
            compute.SetBuffer(_kCheckActivity, ActiveFlag, _activeFlagBuf);
            compute.SetInt(VertexCount, _vertexCount);

            // Boundary mask
            if (
                useBoundaryMask
                && (boundaryWeights == null || boundaryWeights.Length != _vertexCount)
            )
                boundaryWeights = ComputeGeodesicBoundaryWeights(_mesh);

            var bw = useBoundaryMask ? boundaryWeights : MakeOnes(_vertexCount);
            _boundaryWeightBuf = new ComputeBuffer(_vertexCount, sizeof(float));
            _boundaryWeightBuf.SetData(bw);
            compute.SetBuffer(_kImpact, BoundaryWts, _boundaryWeightBuf);
            compute.SetBuffer(_kApplyFingertip, BoundaryWts, _boundaryWeightBuf);
            compute.SetInt(UseBoundaryMsk, useBoundaryMask ? 1 : 0);
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

            // Cache per-frame values used by multiple dispatches
            var s = transform.lossyScale;
            _absScale = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));

            compute.SetVector(AbsScaleId, _absScale);
            compute.SetFloat(MaxDistWs, maxDistance);
            compute.SetFloat(DentId, dent);

            // ── Fingertip mode ────────────────────────────────────────────────
            bool fingertipMode = fingertips != null && fingertips.Length > 0;

            if (fingertipMode)
            {
                bool anyActive = ApplyFingertips();
                if (anyActive)
                    _inactive = false;
            }

            if (_inactive)
                return;

            // ── Spring simulation ─────────────────────────────────────────────
            var dt = Time.fixedDeltaTime * Mathf.Max(0f, simulationSpeed);
            compute.SetFloat(DT, dt);
            compute.SetFloat(SpringK, springStiffness);
            compute.SetFloat(Damping, damping);
            Dispatch(_kSimulate);

            // ── Activity check (GPU → CPU readback, 1 uint) ───────────────────
            _activeFlagCPU[0] = 0;
            _activeFlagBuf.SetData(_activeFlagCPU);
            Dispatch(_kCheckActivity);
            _activeFlagBuf.GetData(_activeFlagCPU);
            _inactive = (_activeFlagCPU[0] == 0);

            if (_inactive)
                return;

            // ── Upload deformed mesh to CPU ───────────────────────────────────
            _posBuf.GetData(_cpuVerts);
            _mesh.vertices = _cpuVerts;
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();
            _localBounds = _mesh.bounds;

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

        // ── Fingertip interaction ─────────────────────────────────────────────

        // Returns true if at least one fingertip is close enough to dispatch a
        // deformation kernel. Called once per FixedUpdate when fingertip mode is active.
        private bool ApplyFingertips()
        {
            // Reset _Target → _Rest on GPU first. This is the key step that enables
            // automatic elastic recovery: when no finger is active, _Target stays at
            // rest and the spring pulls _Pos back without any additional logic.
            Dispatch(_kResetTargets);

            // Expand bounds by maxDistance for coarse CPU-side culling.
            // Fingers outside this box are definitively too far; those inside go to GPU.
            var expanded = _localBounds;
            expanded.Expand(maxDistance * 2f);

            // Geometric center of the mesh in world space.
            // Used to derive an inward push direction that is independent of pivot placement.
            var meshCenterWs = transform.TransformPoint(_localBounds.center);

            bool anyActive = false;

            foreach (var tip in fingertips)
            {
                if (tip == null)
                    continue;

                var fingerPosWs = tip.position;
                var fingerPosLs = transform.InverseTransformPoint(fingerPosWs);

                // Coarse AABB reject — avoids a GPU dispatch for clearly distant fingers.
                if (!expanded.Contains(fingerPosLs))
                    continue;

                // Closest point on the collider surface gives us a stable contact anchor.
                // Even when the finger has penetrated the mesh, ClosestPoint returns the
                // nearest exit point on the surface, which is a good push-direction reference.
                var surfacePtWs = _mc.ClosestPoint(fingerPosWs);

                // Push direction: from the surface contact point toward the mesh geometric center.
                // This correctly tracks the inward surface normal for a convex mesh regardless
                // of where the object pivot is placed.
                // var pushDirWs = meshCenterWs - surfacePtWs;
                // if (pushDirWs.sqrMagnitude < 1e-6f)
                // pushDirWs = -transform.up; // degenerate guard (finger exactly at center)
                // pushDirWs.Normalize();


                Vector3 pushDirWsw = Vector3.up.normalized;
                var pushDirLs = transform.InverseTransformDirection(pushDirWsw).normalized;

                compute.SetVector(ImpactPointLs, fingerPosLs);
                compute.SetVector(PushDirLs, pushDirLs);
                Dispatch(_kApplyFingertip);

                anyActive = true;
            }

            return anyActive;
        }

        // ── Legacy collision path (active when fingertips array is empty) ─────

        private void OnCollisionEnter(Collision c)
        {
            // Disabled when fingertip mode is active to prevent conflict with
            // ResetTargets (which would overwrite collision-driven target offsets).
            if (fingertips != null && fingertips.Length > 0)
                return;

            if (c.contactCount == 0)
                return;
            if (c.impulse.magnitude < minImpactImpulse)
                return;
            if ((deformLayers.value & (1 << c.gameObject.layer)) == 0)
                return;

            var cp = c.GetContact(0);
            ApplyImpactGPU(cp.point, -cp.normal.normalized, kick, dent);
        }

        private void ApplyImpactGPU(
            Vector3 impactPointWs,
            Vector3 pushDirWs,
            float kickValue,
            float dentValue
        )
        {
            _inactive = false;

            var impactPointLs = transform.InverseTransformPoint(impactPointWs);
            var pushDirLs = transform.InverseTransformDirection(pushDirWs).normalized;
            var s = transform.lossyScale;
            var absScale = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));

            compute.SetVector(ImpactPointLs, impactPointLs);
            compute.SetVector(PushDirLs, pushDirLs);
            compute.SetVector(AbsScaleId, absScale);
            compute.SetFloat(MaxDistWs, maxDistance);
            compute.SetFloat(KickId, kickValue);
            compute.SetFloat(DentId, dentValue);

            Dispatch(_kImpact);
        }

        // ── State reset ───────────────────────────────────────────────────────

        private void ResetState()
        {
            _restBuf.GetData(_cpuVerts);

            _targetBuf.SetData(_cpuVerts);
            _posBuf.SetData(_cpuVerts);
            _velBuf.SetData(new Vector3[_vertexCount]);

            _mesh.vertices = _cpuVerts;
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();
            _localBounds = _mesh.bounds;

            if (_mc)
            {
                _mc.sharedMesh = null;
                _mc.sharedMesh = _mesh;
            }
        }

        // ── GPU helpers ───────────────────────────────────────────────────────

        private void Dispatch(int kernel)
        {
            var groups = Mathf.CeilToInt(_vertexCount / 256f);
            compute.Dispatch(kernel, groups, 1, 1);
        }

        private void BindCommon(int kernel)
        {
            compute.SetBuffer(kernel, RestId, _restBuf);
            compute.SetBuffer(kernel, TargetId, _targetBuf);
            compute.SetBuffer(kernel, PosId, _posBuf);
            compute.SetBuffer(kernel, Vel, _velBuf);
            compute.SetInt(VertexCount, _vertexCount);
        }

        // ── Inspector helpers ─────────────────────────────────────────────────

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

        // ── Boundary mask (geodesic Dijkstra, runs once at Awake) ─────────────

        private float[] ComputeGeodesicBoundaryWeights(Mesh m)
        {
            var tris = m.triangles;
            var verts = m.vertices;
            var n = verts.Length;

            var weld = WeldByPosition(verts);

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

            var maxD = 0f;
            for (var i = 0; i < n; i++)
                if (dist[i] < float.MaxValue)
                    maxD = Mathf.Max(maxD, dist[i]);

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
            var dist = Vector3.Distance(verts[a], verts[b]);
            adj[a].Add((b, dist));
            adj[b].Add((a, dist));
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
