using System;
using System.Collections.Generic;
using UnityEngine;

namespace Modules.SoftPhysics
{
    public struct KNearestResult : IComparable<KNearestResult>
    {
        public int Index;
        public float DistanceSqr;

        public int CompareTo(KNearestResult other) => DistanceSqr.CompareTo(other.DistanceSqr);
    }

    // Uniform-grid spatial hash for O(1) average K-nearest queries on a static point cloud.
    // All positions (Build + QueryKNearest) must be in the same local coordinate space.
    // Non-uniform scale on the source transform will skew distance rankings slightly;
    // acceptable for uniform-scaled medical meshes.
    public class PointCloudSpatialHash
    {
        private float _invCellSize;
        private readonly Dictionary<long, List<int>> _cells = new();
        private Vector3[] _points;

        public void Build(Vector3[] points, float cellSize)
        {
            _invCellSize = 1f / Mathf.Max(cellSize, 1e-4f);
            _points = points;
            _cells.Clear();

            for (int i = 0; i < points.Length; i++)
            {
                long key = PointHash(points[i]);
                if (!_cells.TryGetValue(key, out var bucket))
                    _cells[key] = bucket = new List<int>(4);
                bucket.Add(i);
            }
        }

        // Fills results with up to k nearest points within searchRadius, sorted by distance.
        // Returns the count written into results.
        public int QueryKNearest(
            Vector3 queryPos,
            int k,
            float radius,
            List<KNearestResult> results
        )
        {
            results.Clear();
            if (_points == null)
                return 0;

            float rSqr = radius * radius;
            int x0 = Mathf.FloorToInt((queryPos.x - radius) * _invCellSize);
            int x1 = Mathf.FloorToInt((queryPos.x + radius) * _invCellSize);
            int y0 = Mathf.FloorToInt((queryPos.y - radius) * _invCellSize);
            int y1 = Mathf.FloorToInt((queryPos.y + radius) * _invCellSize);
            int z0 = Mathf.FloorToInt((queryPos.z - radius) * _invCellSize);
            int z1 = Mathf.FloorToInt((queryPos.z + radius) * _invCellSize);

            for (int cx = x0; cx <= x1; cx++)
            for (int cy = y0; cy <= y1; cy++)
            for (int cz = z0; cz <= z1; cz++)
            {
                if (!_cells.TryGetValue(CellKey(cx, cy, cz), out var bucket))
                    continue;

                foreach (int idx in bucket)
                {
                    float dSqr = (queryPos - _points[idx]).sqrMagnitude;
                    if (dSqr <= rSqr)
                        results.Add(new KNearestResult { Index = idx, DistanceSqr = dSqr });
                }
            }

            results.Sort();
            if (results.Count > k)
                results.RemoveRange(k, results.Count - k);
            return results.Count;
        }

        private long PointHash(Vector3 p) =>
            CellKey(
                Mathf.FloorToInt(p.x * _invCellSize),
                Mathf.FloorToInt(p.y * _invCellSize),
                Mathf.FloorToInt(p.z * _invCellSize)
            );

        // Packs 3 × 21-bit signed integers into a 63-bit long.
        // Covers ±2 097 151 cells per axis (~125 km at 6 cm cell size).
        private static long CellKey(int x, int y, int z)
        {
            const long M = 0x1FFFFF;
            return ((long)x & M) | (((long)y & M) << 21) | (((long)z & M) << 42);
        }
    }
}
