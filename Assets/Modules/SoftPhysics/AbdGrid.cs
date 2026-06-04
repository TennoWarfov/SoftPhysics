using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace Modules.SoftPhysics
{
    public class AbdGrid : MonoBehaviour
    {
        [SerializeField]
        private float gridSpacing = 0.1f;

        [SerializeField]
        private Cell cellPrefab;

        [SerializeField]
        private DecalProjector decalPrefab;

        private const float fadeDuration = 1f;
        private const float targetOpacity = 1f;
        private bool _isShown;
        private List<DecalProjector> _projectors;
        private CancellationTokenSource _cancellationTokenSource;

        private void Start()
        {
            CreateGrid();
            foreach (var projector in _projectors)
                projector.fadeFactor = 0f;
        }

        public void CreateGrid()
        {
            _projectors = new List<DecalProjector>();

            var center = transform.position;
            var cCell = Instantiate(cellPrefab, center, Quaternion.identity, transform);

            var up = center + transform.up * gridSpacing;
            var uCell = Instantiate(cellPrefab, up, Quaternion.identity, transform);

            var leftUp = up + -transform.right * gridSpacing;
            var luCell = Instantiate(cellPrefab, leftUp, Quaternion.identity, transform);

            var rightUp = up + transform.right * gridSpacing;
            var ruCell = Instantiate(cellPrefab, rightUp, Quaternion.identity, transform);

            var left = center + -transform.right * gridSpacing;
            var lCell = Instantiate(cellPrefab, left, Quaternion.identity, transform);

            var right = center + transform.right * gridSpacing;
            var rCell = Instantiate(cellPrefab, right, Quaternion.identity, transform);

            var leftDown = left + -transform.up * gridSpacing;
            var ldCell = Instantiate(cellPrefab, leftDown, Quaternion.identity, transform);

            var rightDown = right + -transform.up * gridSpacing;
            var rdCell = Instantiate(cellPrefab, rightDown, Quaternion.identity, transform);

            var down = center + -transform.up * gridSpacing;
            var dCell = Instantiate(cellPrefab, down, Quaternion.identity, transform);

            if (decalPrefab)
            {
                var rotation = Vector3.zero;
                var vlu = up + -transform.right * gridSpacing / 2;
                var vluProjector = CreateDecal(vlu, rotation, "vlu");
                var vru = up + transform.right * gridSpacing / 2;
                var vruProjector = CreateDecal(vru, rotation, "vru");
                var vld = down + -transform.right * gridSpacing / 2;
                var vldProjector = CreateDecal(vld, rotation, "vld");
                var vrd = down + transform.right * gridSpacing / 2;
                var vrdProjector = CreateDecal(vrd, rotation, "vrd");
                var l = center + -transform.right * gridSpacing / 2;
                var lProjector = CreateDecal(l, rotation, "l");
                var r = center + transform.right * gridSpacing / 2;
                var rProjector = CreateDecal(r, rotation, "r");

                rotation = new Vector3(0, 0, 90);
                var hld = left + -transform.up * gridSpacing / 2;
                var hldProjector = CreateDecal(hld, rotation, "hld");
                var hrd = right + -transform.up * gridSpacing / 2;
                var hrdProjector = CreateDecal(hrd, rotation, "hrd");
                var hlu = left + transform.up * gridSpacing / 2;
                var hluProjector = CreateDecal(hlu, rotation, "hlu");
                var hru = right + transform.up * gridSpacing / 2;
                var hruProjector = CreateDecal(hru, rotation, "hru");
                var u = center + transform.up * gridSpacing / 2;
                var uProjector = CreateDecal(u, rotation, "d");
                var d = center + -transform.up * gridSpacing / 2;
                var dProjector = CreateDecal(d, rotation, "u");

                _projectors.Add(vluProjector);
                _projectors.Add(vruProjector);
                _projectors.Add(vldProjector);
                _projectors.Add(vrdProjector);
                _projectors.Add(lProjector);
                _projectors.Add(rProjector);
                _projectors.Add(hluProjector);
                _projectors.Add(hrdProjector);
                _projectors.Add(hldProjector);
                _projectors.Add(hruProjector);
                _projectors.Add(uProjector);
                _projectors.Add(dProjector);

                cCell.Initialize(lProjector, rProjector, uProjector, dProjector);
                uCell.Initialize(vluProjector, vruProjector, uProjector);
                lCell.Initialize(hluProjector, hldProjector, lProjector);
                rCell.Initialize(hruProjector, hrdProjector, rProjector);
                dCell.Initialize(vldProjector, vrdProjector, dProjector);
                ldCell.Initialize(hldProjector, vldProjector);
                rdCell.Initialize(hrdProjector, vrdProjector);
                luCell.Initialize(hluProjector, vluProjector);
                ruCell.Initialize(hruProjector, vruProjector);
            }
        }

        private DecalProjector CreateDecal(Vector3 lu, Vector3 rotation, string n)
        {
            var projector = Instantiate(decalPrefab, lu, Quaternion.identity, transform);
            projector.size = new Vector3(0.005f, gridSpacing, gridSpacing);
            projector.transform.localEulerAngles = rotation;
            projector.name = n;
            return projector;
        }

        private async void OnTriggerEnter(Collider other)
        {
            try
            {
                if (other.TryGetComponent(out PhysicallyHand _))
                {
                    if (!_isShown)
                    {
                        _isShown = true;
                        _cancellationTokenSource?.Cancel();
                        _cancellationTokenSource = new CancellationTokenSource();
                        await ShowDecals(_cancellationTokenSource.Token);
                    }
                }
            }
            catch (Exception)
            {
                // ignored
            }
        }

        private async void OnTriggerExit(Collider other)
        {
            try
            {
                if (other.TryGetComponent(out PhysicallyHand _))
                {
                    if (_isShown)
                    {
                        _isShown = false;
                        _cancellationTokenSource?.Cancel();
                        _cancellationTokenSource = new CancellationTokenSource();
                        await HideDecals(_cancellationTokenSource.Token);
                    }
                }
            }
            catch (Exception)
            {
                // ignored
            }
        }

        private void OnDestroy()
        {
            _cancellationTokenSource?.Cancel();
            _cancellationTokenSource?.Dispose();
        }

        private async Task ShowDecals(CancellationToken cancellationToken)
        {
            if (_projectors == null || _projectors.Count == 0)
                return;

            var elapsedTime = 0f;
            var startOpacities = new float[_projectors.Count];

            for (var i = 0; i < _projectors.Count; i++)
            {
                if (_projectors[i])
                {
                    startOpacities[i] = _projectors[i].fadeFactor;
                }
            }

            while (elapsedTime < fadeDuration)
            {
                elapsedTime += Time.deltaTime;
                var t = Mathf.Clamp01(elapsedTime / fadeDuration);

                for (var i = 0; i < _projectors.Count; i++)
                {
                    if (_projectors[i])
                    {
                        _projectors[i].fadeFactor = Mathf.Lerp(startOpacities[i], targetOpacity, t);
                    }
                }

                if (cancellationToken.IsCancellationRequested)
                    return;

                await Task.Yield();
            }

            foreach (var projector in _projectors)
            {
                if (projector)
                {
                    projector.fadeFactor = targetOpacity;
                }
            }
        }

        private async Task HideDecals(CancellationToken cancellationToken)
        {
            if (_projectors == null || _projectors.Count == 0)
                return;

            var elapsedTime = 0f;
            var startOpacities = new float[_projectors.Count];

            for (var i = 0; i < _projectors.Count; i++)
            {
                if (_projectors[i])
                {
                    startOpacities[i] = _projectors[i].fadeFactor;
                }
            }

            while (elapsedTime < fadeDuration)
            {
                elapsedTime += Time.deltaTime;
                var t = Mathf.Clamp01(elapsedTime / fadeDuration);

                for (var i = 0; i < _projectors.Count; i++)
                {
                    if (_projectors[i])
                    {
                        _projectors[i].fadeFactor = Mathf.Lerp(startOpacities[i], 0f, t);
                    }
                }

                if (cancellationToken.IsCancellationRequested)
                    return;

                await Task.Yield();
            }

            foreach (var projector in _projectors)
            {
                if (projector)
                {
                    projector.fadeFactor = 0f;
                }
            }
        }

        private void OnDrawGizmos()
        {
            var center = transform.position;
            Gizmos.DrawSphere(center, .01f);

            var up = center + transform.up * gridSpacing;
            Gizmos.DrawSphere(up, .01f);

            var leftUp = up + -transform.right * gridSpacing;
            Gizmos.DrawSphere(leftUp, .01f);

            var rightUp = up + transform.right * gridSpacing;
            Gizmos.DrawSphere(rightUp, .01f);

            var left = center + -transform.right * gridSpacing;
            Gizmos.DrawSphere(left, .01f);

            var right = center + transform.right * gridSpacing;
            Gizmos.DrawSphere(right, .01f);

            var leftDown = left + -transform.up * gridSpacing;
            Gizmos.DrawSphere(leftDown, .01f);

            var rightDown = right + -transform.up * gridSpacing;
            Gizmos.DrawSphere(rightDown, .01f);

            var down = center + -transform.up * gridSpacing;
            Gizmos.DrawSphere(down, .01f);
        }
    }
}
