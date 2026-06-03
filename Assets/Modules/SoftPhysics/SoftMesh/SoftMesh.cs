using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshCollider))]
public class SoftMesh : MonoBehaviour
{
    [Header("Compute")]
    public ComputeShader compute;

    [Header("Impact")]
    public float minImpactImpulse = 0.5f;
    public float dent = 0.05f;
    public float kick = 1.5f;
    public float maxDistance = 1.0f;
    public LayerMask deformLayers;

    [Header("Spring")]
    public float springStiffness = 50f;
    public float damping = 8f;

    [Header("Collider")]
    public bool updateCollider = true;
    public int updateColliderEveryNFrames = 5;

    [Header("Boundary Mask")]
    [SerializeField] bool useBoundaryMask = true;
    [SerializeField] AnimationCurve falloffCurve = AnimationCurve.EaseInOut(0f, 0f, 1f, 1f);
    [SerializeField] float[] boundaryWeights;

    [Header("Debug")]
    public bool reset = false;

    // Sleep / activity flag
    bool inactive = false;

    MeshFilter mf;
    MeshCollider mc;
    Mesh mesh;

    Vector3[] cpuVerts;
    int frameCounter;

    //GPU Resources
    ComputeBuffer restBuf, targetBuf, posBuf, velBuf, activeFlagBuf;
    ComputeBuffer boundaryWeightBuf;
    int kSimulate, kImpact, kCheckActivity; // Kernels
    int vertexCount;
    uint[] activeFlagCPU = new uint[1];


    void Awake()
    {
        mf = GetComponent<MeshFilter>();
        mc = GetComponent<MeshCollider>();

        //One instance per SoftMesh otherwise they share the same buffer bindings!
        compute = Instantiate(compute);

        mesh = Instantiate(mf.sharedMesh);
        mesh.MarkDynamic();
        mf.sharedMesh = mesh;
        mc.sharedMesh = mesh;

        var rest = mesh.vertices;
        vertexCount = rest.Length;

        cpuVerts = new Vector3[vertexCount];

        // Create Buffers
        restBuf = new ComputeBuffer(vertexCount, sizeof(float) * 3);
        posBuf = new ComputeBuffer(vertexCount, sizeof(float) * 3);
        velBuf = new ComputeBuffer(vertexCount, sizeof(float) * 3);
        targetBuf = new ComputeBuffer(vertexCount, sizeof(float) * 3);
        restBuf.SetData(rest);
        targetBuf.SetData(rest);
        posBuf.SetData(rest);
        velBuf.SetData(new Vector3[vertexCount]);

        // Find Kernels
        kSimulate = compute.FindKernel("Simulate");
        kImpact = compute.FindKernel("ApplyImpact");
        kCheckActivity = compute.FindKernel("CheckActivity");

        // Bind Buffers to Kernels
        BindCommon(kSimulate);
        BindCommon(kImpact);

        // 1-uint activity flag buffer
        activeFlagBuf = new ComputeBuffer(1, sizeof(uint));
        activeFlagBuf.SetData(new uint[] { 0 });

        // Bind to kernel
        compute.SetBuffer(kCheckActivity, "_Vel", velBuf);
        compute.SetBuffer(kCheckActivity, "_ActiveFlag", activeFlagBuf);
        compute.SetInt("_VertexCount", vertexCount);

        // Boundary mask: reuse serialized weights if valid, otherwise compute from scratch
        if (useBoundaryMask && (boundaryWeights == null || boundaryWeights.Length != vertexCount))
            boundaryWeights = ComputeGeodesicBoundaryWeights(mesh);

        float[] bw = useBoundaryMask ? boundaryWeights : MakeOnes(vertexCount);
        boundaryWeightBuf = new ComputeBuffer(vertexCount, sizeof(float));
        boundaryWeightBuf.SetData(bw);
        compute.SetBuffer(kImpact, "_BoundaryWeights", boundaryWeightBuf);
        compute.SetInt("_UseBoundaryMask", useBoundaryMask ? 1 : 0);
    }

    void OnDestroy()
    {
        restBuf?.Release();
        targetBuf?.Release();
        posBuf?.Release();
        velBuf?.Release();
        activeFlagBuf?.Release();
        boundaryWeightBuf?.Release();
    }


    void FixedUpdate()
    {
        frameCounter++;

        if (reset)
        {
            ResetState();
            reset = false;
            inactive = false;
        }

        if (inactive)
            return;

        compute.SetFloat("_DT", Time.fixedDeltaTime);
        compute.SetFloat("_SpringK", springStiffness);
        compute.SetFloat("_Damping", damping);
        Dispatch(kSimulate);

        activeFlagCPU[0] = 0;
        activeFlagBuf.SetData(activeFlagCPU);
        Dispatch(kCheckActivity);
        activeFlagBuf.GetData(activeFlagCPU);
        inactive = (activeFlagCPU[0] == 0);

        if (inactive)
            return;

        posBuf.GetData(cpuVerts);

        mesh.vertices = cpuVerts;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        if (updateCollider && mc != null)
        {
            int n = Mathf.Max(1, updateColliderEveryNFrames);

            if (frameCounter % n == 0)
            {
                mc.sharedMesh = null;
                mc.sharedMesh = mesh;
            }
        }

    }


    // OnCollisionEnter applies an impact to the mesh on collision. Very slow on large meshes!
    void OnCollisionEnter(Collision c)
    {
        if (c.contactCount == 0) return;
        if (c.impulse.magnitude < minImpactImpulse) return;
        if ((deformLayers.value & (1 << c.gameObject.layer)) == 0) return;

        var cp = c.GetContact(0);

        Vector3 impactPointWS = cp.point;
        Vector3 pushDirWS = -cp.normal.normalized;

        //float kick = kick * c.impulse.magnitude;
        ApplyImpactGPU(impactPointWS, pushDirWS, kick, dent);
    }


    void ApplyImpactGPU(Vector3 impactPointWS, Vector3 pushDirWS, float kick, float dent)
    {
        inactive = false;

        // Convert impact point from world space to local space
        Vector3 impactPointLS = transform.InverseTransformPoint(impactPointWS);

        // Convert direction from world space to local space
        Vector3 pushDirLS = transform.InverseTransformDirection(pushDirWS).normalized;

        // Absolute lossy scale used to approximate world-space distance
        // from local-space delta (handles non-uniform scale)
        Vector3 s = transform.lossyScale;
        Vector3 absScale = new Vector3(
            Mathf.Abs(s.x),
            Mathf.Abs(s.y),
            Mathf.Abs(s.z)
        );

        compute.SetVector("_ImpactPointLS", impactPointLS);
        compute.SetVector("_PushDirLS", pushDirLS);

        // Scale correction so MaxDistance remains in world units
        compute.SetVector("_AbsScale", absScale);
        compute.SetFloat("_MaxDistanceWS", maxDistance);

        compute.SetFloat("_Kick", kick);
        compute.SetFloat("_Dent", dent);

        Dispatch(kImpact);
    }

    void ResetState()
    {
        restBuf.GetData(cpuVerts);

        targetBuf.SetData(cpuVerts);             // Target = Rest
        posBuf.SetData(cpuVerts);                // Pos = Rest
        velBuf.SetData(new Vector3[vertexCount]); // Vel = 0

        mesh.vertices = cpuVerts;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        if (mc != null)
        {
            mc.sharedMesh = null;
            mc.sharedMesh = mesh;
        }
    }

    void Dispatch(int kernel)
    {
        int groups = Mathf.CeilToInt(vertexCount / 256f);
        compute.Dispatch(kernel, groups, 1, 1);
    }

    void BindCommon(int kernel)
    {
        compute.SetBuffer(kernel, "_Rest", restBuf);
        compute.SetBuffer(kernel, "_Target", targetBuf);
        compute.SetBuffer(kernel, "_Pos", posBuf);
        compute.SetBuffer(kernel, "_Vel", velBuf);
        compute.SetInt("_VertexCount", vertexCount);
    }

    // Recomputes geodesic weights and re-uploads to GPU.
    // Available via right-click on the component in the Inspector.
    [ContextMenu("Recompute Boundary Weights")]
    void RecomputeBoundaryWeights()
    {
        var sourceMesh = Application.isPlaying ? mesh : GetComponent<MeshFilter>()?.sharedMesh;
        if (sourceMesh == null) return;

        boundaryWeights = ComputeGeodesicBoundaryWeights(sourceMesh);

        if (Application.isPlaying && boundaryWeightBuf != null)
            boundaryWeightBuf.SetData(boundaryWeights);
    }


    // -------------------------------------------------------------------------
    // Boundary Mask: geodesic distance via multi-source Dijkstra
    // Runs once at Awake; zero CPU cost at runtime.
    // -------------------------------------------------------------------------

    float[] ComputeGeodesicBoundaryWeights(Mesh m)
    {
        int[] tris = m.triangles;
        Vector3[] verts = m.vertices;
        int n = verts.Length;

        // Weld co-located vertices so UV-seam duplicates don't appear as false boundaries
        int[] weld = WeldByPosition(verts);

        // Count undirected edges by welded index; a boundary edge appears exactly once
        var edgeCount = new Dictionary<(int, int), int>();
        for (int t = 0; t < tris.Length; t += 3)
        {
            CountEdge(edgeCount, weld[tris[t]], weld[tris[t + 1]]);
            CountEdge(edgeCount, weld[tris[t + 1]], weld[tris[t + 2]]);
            CountEdge(edgeCount, weld[tris[t + 2]], weld[tris[t]]);
        }

        var boundaryWelded = new HashSet<int>();
        foreach (var kv in edgeCount)
            if (kv.Value == 1) { boundaryWelded.Add(kv.Key.Item1); boundaryWelded.Add(kv.Key.Item2); }

        if (boundaryWelded.Count == 0)
        {
            Debug.LogWarning("[SoftMesh] useBoundaryMask: no boundary edges found (closed mesh?). Mask has no effect.");
            return MakeOnes(n);
        }

        // Build adjacency list over original (unwelded) vertex indices
        var adj = new List<(int v, float d)>[n];
        for (int i = 0; i < n; i++) adj[i] = new List<(int, float)>();
        for (int t = 0; t < tris.Length; t += 3)
        {
            int a = tris[t], b = tris[t + 1], c = tris[t + 2];
            AddAdj(adj, a, b, verts);
            AddAdj(adj, b, c, verts);
            AddAdj(adj, c, a, verts);
        }

        // Zero-cost edges between welded pairs so geodesic can cross UV seams freely
        var weldGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            if (!weldGroups.TryGetValue(weld[i], out var g)) weldGroups[weld[i]] = g = new List<int>();
            g.Add(i);
        }
        foreach (var g in weldGroups.Values)
        {
            for (int a = 0; a < g.Count; a++)
                for (int b = a + 1; b < g.Count; b++)
                {
                    adj[g[a]].Add((g[b], 0f));
                    adj[g[b]].Add((g[a], 0f));
                }
        }

        // Multi-source Dijkstra from all boundary vertices (O(n²), runs once at startup)
        float[] dist = new float[n];
        bool[] visited = new bool[n];
        for (int i = 0; i < n; i++) dist[i] = float.MaxValue;
        for (int i = 0; i < n; i++)
            if (boundaryWelded.Contains(weld[i])) dist[i] = 0f;

        for (int iter = 0; iter < n; iter++)
        {
            int u = -1;
            float minD = float.MaxValue;
            for (int i = 0; i < n; i++)
                if (!visited[i] && dist[i] < minD) { minD = dist[i]; u = i; }
            if (u < 0) break;
            visited[u] = true;
            foreach (var (v, d) in adj[u])
            {
                float nd = dist[u] + d;
                if (nd < dist[v]) dist[v] = nd;
            }
        }

        // Normalize to [0..1] and apply smoothstep for a soft falloff
        float maxD = 0f;
        for (int i = 0; i < n; i++)
            if (dist[i] < float.MaxValue) maxD = Mathf.Max(maxD, dist[i]);

        // Degenerate case: every vertex is on the boundary
        if (maxD < 1e-6f)
            return MakeOnes(n);

        float[] weights = new float[n];
        for (int i = 0; i < n; i++)
        {
            float t = Mathf.Clamp01(dist[i] / maxD);
            weights[i] = Mathf.Clamp01(falloffCurve.Evaluate(t));
        }
        return weights;
    }

    // Group vertices by rounded position; tolerance = 0.00001 world units
    static int[] WeldByPosition(Vector3[] verts)
    {
        const float invEps = 100000f;
        int n = verts.Length;
        int[] welded = new int[n];
        var map = new Dictionary<(int, int, int), int>(n);
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            var v = verts[i];
            var key = (Mathf.RoundToInt(v.x * invEps),
                       Mathf.RoundToInt(v.y * invEps),
                       Mathf.RoundToInt(v.z * invEps));
            if (!map.TryGetValue(key, out int id)) { id = count++; map[key] = id; }
            welded[i] = id;
        }
        return welded;
    }

    static void CountEdge(Dictionary<(int, int), int> d, int a, int b)
    {
        var key = a < b ? (a, b) : (b, a);
        d.TryGetValue(key, out int c);
        d[key] = c + 1;
    }

    static void AddAdj(List<(int, float)>[] adj, int a, int b, Vector3[] verts)
    {
        float d = Vector3.Distance(verts[a], verts[b]);
        adj[a].Add((b, d));
        adj[b].Add((a, d));
    }

    static float[] MakeOnes(int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1f;
        return a;
    }
}
