# Localización del LCA en RM de rodilla (CI-0148)

Proyecto del curso CI-0148 Aprendizaje Automático (UCR) que sigue CRISP-ML(Q):
localizar el ligamento cruzado anterior (LCA) en volúmenes de RM del conjunto
KneeMRI (Štajduhar et al., 2017), con un modelo de aprendizaje profundo 2.5D.

Este README cubre la **fase 2 (preparación de datos)**: cómo reconstruir, desde los
datos crudos, el conjunto listo para entrenar.

## Estructura

```
configs/prep.yaml          toda decisión del pipeline (semilla, rutas, percentiles, partición)
src/data/                  funciones puras: io, manifest, qa, split, transforms, slices
scripts/build_dataset.py   único punto de entrada; encadena las etapas y escribe artefactos
tests/                     pruebas unitarias (pytest) sobre datos sintéticos
notebooks/                 solo exploración; nada del pipeline depende de ellos
docs/                      entregables (fase 1 en PDF; fase 2 en docs/entregable2/)
datasets/                  ignorado por git
  raw/                     datos crudos, de solo lectura
  interim/                 manifest.csv, qa_report.csv, splits.csv
  processed/               volumes/*.npy y slices.csv (listos para entrenar)
```

## 1. Ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Probado con Python 3.14.4 en macOS (Apple M1, 16 GB). No requiere GPU.

## 2. Datos crudos

1. Descargar los 917 volúmenes `.pck` de Zenodo (registro 14789903,
   licencia CC BY-NC-ND 4.0) en `datasets/raw/volumetric_data/`.
2. Colocar `metadata.csv` (redistribución de Kaggle, `sohaibanwaar1203/kneemridataset`)
   en `datasets/raw/kneemridataset/metadata.csv`.

La descarga de Kaggle trae solo 736 de los 917 volúmenes; por eso los volúmenes se
toman de Zenodo. Los 736 comunes son idénticos byte a byte en ambas fuentes.

## 3. Reconstruir el conjunto

Desde la raíz del proyecto:

```bash
python -m scripts.build_dataset
```

Etapas, en orden (~1 min):

1. **Manifiesto**: una fila por volumen, con forma, tipo e intensidades leídas del archivo.
2. **Puerta de calidad**: etiquetas, archivo presente, caja dentro del volumen, `uint16`,
   intensidad en 12 bits. Si `qa_report.csv` no queda vacío, el pipeline se detiene.
3. **Partición**: `StratifiedGroupKFold` agrupado por `examId` y estratificado por
   `aclDiagnosis`; prueba = 1 de 6 pliegues, luego 5 pliegues de validación cruzada.
   Verifica que no haya fuga y que las proporciones estén a ±2 p.p. del total.
4. **Transformaciones** (por volumen, sin fuga): recorte a percentiles 0.5–99.5,
   puntaje z, padding a 320×320 al final de cada eje (la caja no cambia), `float16`.
5. **Índice de cortes 2.5D**: `slices.csv`, una fila por corte con `contains_acl`,
   `dist_to_roi` y la caja 2D (extremo excluido).

### Resultado esperado

La salida es determinista. Tras una reconstrucción, estos hashes deben coincidir:

```bash
shasum datasets/interim/*.csv datasets/processed/slices.csv
```

| Archivo | SHA-1 |
|---|---|
| `datasets/interim/manifest.csv` | `2d7604fb10138b4c3c940135e836303c08aa2984` |
| `datasets/interim/qa_report.csv` | `654e8df57f80dc55462cbb2ea65afb1c27ea6046` |
| `datasets/interim/splits.csv` | `0877f2eeebace2b8e069e4f91f961a93d9ec4d58` |
| `datasets/processed/slices.csv` | `d4636c0ec821739e54167283ddad06f254813d7a` |

Además: 917 archivos en `datasets/processed/volumes/` (~5.4 GB) y 28 359 cortes,
3 081 de ellos positivos.

## 4. Pruebas

```bash
pytest
```

31 pruebas sobre datos sintéticos (no necesitan el conjunto descargado): clave
duplicada, cada verificación de calidad puede fallar, ausencia de fuga y
determinismo de la partición, normalización y padding, y el índice de cortes.

## 5. Figuras del entregable 2

```bash
python -m docs.entregable2.make_figures
```

Regenera `docs/entregable2/figs/` (normalización e inspección visual de 30 cajas).
