using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace Modules.SoftPhysics
{
    public class Cell : MonoBehaviour
    {
        [SerializeField]
        private Material greenMaterial;

        [SerializeField]
        private Material defaultMaterial;

        private DecalProjector[] _decalProjectors;

        private void Awake()
        {
            var col = gameObject.AddComponent<BoxCollider>();
            col.isTrigger = true;
            col.size = new Vector3(0.06f, 0.06f, 0.06f);
        }

        public void Initialize(params DecalProjector[] decalProjector)
        {
            _decalProjectors = decalProjector;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.transform.GetComponent<Finger>())
            {
                ToggleMaterial(true);
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (other.transform.GetComponent<Finger>())
            {
                ToggleMaterial(false);
            }
        }

        private void ToggleMaterial(bool isGreen)
        {
            if (_decalProjectors != null)
            {
                foreach (var projector in _decalProjectors)
                {
                    projector.material = isGreen ? greenMaterial : defaultMaterial;
                }
            }
        }
    }
}
