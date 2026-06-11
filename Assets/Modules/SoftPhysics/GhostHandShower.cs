using UnityEngine;

namespace Modules.SoftPhysics
{
    public class GhostHandShower : MonoBehaviour
    {
        [SerializeField]
        private GameObject ghostMesh;

        [SerializeField]
        private Transform ghost;

        [SerializeField]
        private Transform target;

        private bool _isShown;

        private void Start()
        {
            var mesh = ghostMesh.GetComponent<SkinnedMeshRenderer>();
            mesh.materials[1].SetColor("_MainColor", new Color(.45f, .2f, .52f, .1f));
            mesh.materials[1].SetColor("_EdgeColor", new Color(.45f, .2f, .52f, 1f));
            ghostMesh.SetActive(false);
        }

        private void Update()
        {
            const float threshold = 0.03f;
            if (Vector3.Distance(ghost.position, target.position) > threshold)
            {
                if (!_isShown)
                {
                    ghostMesh.SetActive(true);
                    _isShown = true;
                }
            }
            else
            {
                if (_isShown)
                {
                    ghostMesh.SetActive(false);
                    _isShown = false;
                }
            }
        }
    }
}
