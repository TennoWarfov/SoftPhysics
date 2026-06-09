using System.Collections.Generic;
using UnityEngine;

namespace Modules.SoftPhysics
{
    [ExecuteAlways]
    public class MeshPointCloudGenerator : MonoBehaviour
    {
        [SerializeField]
        private MeshFilter sourceMeshFilter;

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
            if (sourceMeshFilter == null)
                return;

            Mesh mesh = sourceMeshFilter.sharedMesh;

            Vector3[] vertices = mesh.vertices;
            Vector3[] meshNormals = mesh.normals;
            int[] triangles = mesh.triangles;

            var uniqueVertices = new List<Vector3>();
            var uniqueNormals = new List<Vector3>();

            Dictionary<Vector3, int> lookup = new();

            for (int i = 0; i < vertices.Length; i++)
            {
                if (!lookup.ContainsKey(vertices[i]))
                {
                    lookup.Add(vertices[i], uniqueVertices.Count);

                    uniqueVertices.Add(vertices[i]);
                    uniqueNormals.Add(meshNormals[i]);
                }
            }

            smoothedPoints = uniqueVertices.ToArray();
            normals = uniqueNormals.ToArray();

            List<int>[] neighbours = BuildNeighbourMap(
                triangles,
                lookup,
                vertices,
                smoothedPoints.Length
            );

            for (int iteration = 0; iteration < smoothingIterations; iteration++)
            {
                Vector3[] next = new Vector3[smoothedPoints.Length];

                for (int i = 0; i < smoothedPoints.Length; i++)
                {
                    if (neighbours[i].Count == 0)
                    {
                        next[i] = smoothedPoints[i];
                        continue;
                    }

                    Vector3 average = Vector3.zero;

                    foreach (int neighbour in neighbours[i])
                        average += smoothedPoints[neighbour];

                    average /= neighbours[i].Count;

                    next[i] = Vector3.Lerp(smoothedPoints[i], average, smoothingStrength);
                }

                smoothedPoints = next;
            }

            for (int i = 0; i < smoothedPoints.Length; i++)
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

        private List<int>[] BuildNeighbourMap(
            int[] triangles,
            Dictionary<Vector3, int> lookup,
            Vector3[] originalVertices,
            int pointCount
        )
        {
            List<int>[] neighbours = new List<int>[pointCount];

            for (int i = 0; i < pointCount; i++)
                neighbours[i] = new List<int>();

            for (int i = 0; i < triangles.Length; i += 3)
            {
                int a = lookup[originalVertices[triangles[i]]];
                int b = lookup[originalVertices[triangles[i + 1]]];
                int c = lookup[originalVertices[triangles[i + 2]]];

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

            foreach (Vector3 point in smoothedPoints)
            {
                Gizmos.DrawSphere(transform.TransformPoint(point), pointSize);
            }
        }
    }
}
