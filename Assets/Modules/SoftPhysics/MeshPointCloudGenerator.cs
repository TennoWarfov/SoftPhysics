using System.Collections.Generic;
using UnityEngine;

namespace Modules.SoftPhysics
{
    [ExecuteAlways]
    public class MeshPointCloudGenerator : MonoBehaviour
    {
        [SerializeField]
        private MeshFilter[] sourceMeshFilters;

        [Header("Point Cloud Offset")]
        [SerializeField]
        private Vector3 offset;

        [SerializeField]
        private float normalOffset = 0.02f;

        [Header("Smoothing")]
        [SerializeField]
        private int smoothingIterations = 3;

        [Range(0f, 1f)]
        [SerializeField]
        private float smoothingStrength = 0.5f;

        [Header("Spatial Hash")]
        [Tooltip("Hash cell size in local units. ~2× average point spacing is optimal.")]
        [SerializeField]
        private float hashCellSize = 0.06f;

        [Header("Debug")]
        [SerializeField]
        private bool drawPoints = true;

        [SerializeField]
        private float pointSize = 0.005f;

        private Vector3[] smoothedPoints;
        private Vector3[] normals;
        private PointCloudSpatialHash _hash;

        public IReadOnlyList<Vector3> Points => smoothedPoints;
        public IReadOnlyList<Vector3> Normals => normals;

        private void Start()
        {
            Generate();
        }

        [ContextMenu("Generate Point Cloud")]
        public void Generate()
        {
            if (sourceMeshFilters == null || sourceMeshFilters.Length == 0)
                return;

            var uniqueVertices = new List<Vector3>();
            var uniqueNormals = new List<Vector3>();
            var lookup = new Dictionary<Vector3, int>();
            var globalTriangles = new List<int>();

            foreach (var mf in sourceMeshFilters)
            {
                if (mf == null || mf.sharedMesh == null)
                    continue;

                var mesh = mf.sharedMesh;
                var vertices = mesh.vertices;
                var meshNormals = mesh.normals;
                var triangles = mesh.triangles;

                // Map each vertex index in this mesh to a global unique-vertex index,
                // converting positions and normals into this transform's local space.
                var indexRemap = new int[vertices.Length];

                for (var i = 0; i < vertices.Length; i++)
                {
                    var localPos = transform.InverseTransformPoint(
                        mf.transform.TransformPoint(vertices[i])
                    );

                    if (!lookup.TryGetValue(localPos, out var globalIdx))
                    {
                        globalIdx = uniqueVertices.Count;
                        lookup[localPos] = globalIdx;

                        var localNormal = transform.InverseTransformDirection(
                            mf.transform.TransformDirection(meshNormals[i])
                        );

                        uniqueVertices.Add(localPos);
                        uniqueNormals.Add(localNormal);
                    }

                    indexRemap[i] = globalIdx;
                }

                for (var i = 0; i < triangles.Length; i += 3)
                {
                    globalTriangles.Add(indexRemap[triangles[i]]);
                    globalTriangles.Add(indexRemap[triangles[i + 1]]);
                    globalTriangles.Add(indexRemap[triangles[i + 2]]);
                }
            }

            smoothedPoints = uniqueVertices.ToArray();
            normals = uniqueNormals.ToArray();

            var neighbours = BuildNeighbourMap(globalTriangles.ToArray(), smoothedPoints.Length);

            for (var iteration = 0; iteration < smoothingIterations; iteration++)
            {
                var next = new Vector3[smoothedPoints.Length];

                for (var i = 0; i < smoothedPoints.Length; i++)
                {
                    if (neighbours[i].Count == 0)
                    {
                        next[i] = smoothedPoints[i];
                        continue;
                    }

                    var average = Vector3.zero;
                    foreach (var nb in neighbours[i])
                        average += smoothedPoints[nb];
                    average /= neighbours[i].Count;

                    next[i] = Vector3.Lerp(smoothedPoints[i], average, smoothingStrength);
                }

                smoothedPoints = next;
            }

            for (var i = 0; i < smoothedPoints.Length; i++)
            {
                normals[i] = normals[i].normalized;
                smoothedPoints[i] += offset;
                smoothedPoints[i] += normals[i] * normalOffset;
            }

            _hash = new PointCloudSpatialHash();
            _hash.Build(smoothedPoints, hashCellSize);
        }

        // Query K nearest points in local space. Results sorted by distance (closest first).
        public int QueryKNearest(Vector3 localPos, int k, float radius, List<KNearestResult> results) =>
            _hash?.QueryKNearest(localPos, k, radius, results) ?? 0;

        private List<int>[] BuildNeighbourMap(int[] triangles, int pointCount)
        {
            var neighbours = new List<int>[pointCount];
            for (var i = 0; i < pointCount; i++)
                neighbours[i] = new List<int>();

            for (var i = 0; i < triangles.Length; i += 3)
            {
                int a = triangles[i], b = triangles[i + 1], c = triangles[i + 2];
                AddNeighbour(neighbours[a], b);
                AddNeighbour(neighbours[a], c);
                AddNeighbour(neighbours[b], a);
                AddNeighbour(neighbours[b], c);
                AddNeighbour(neighbours[c], a);
                AddNeighbour(neighbours[c], b);
            }

            return neighbours;
        }

        private void AddNeighbour(List<int> neighbours, int index)
        {
            if (!neighbours.Contains(index))
                neighbours.Add(index);
        }

        private void OnDrawGizmosSelected()
        {
            if (!drawPoints || smoothedPoints == null)
                return;

            Gizmos.color = Color.green;

            foreach (var point in smoothedPoints)
                Gizmos.DrawSphere(transform.TransformPoint(point), pointSize);
        }
    }
}
