# Entregable
Este entregable corresponde a la **fase 1 de CRISP-ML(Q): comprensión del negocio y de los datos**. Recuerden que en CRISP-ML(Q) esta fase fusiona lo que en CRISP-DM eran dos fases separadas, porque la disponibilidad y la calidad de los datos son las que determinan si el proyecto es viable.

Deben entregar un informe con estructura típica que comience con una introducción. La introducción debería indicar qué se encontrará en el documento. Asimismo, la introducción debería tener una explicación del contexto donde se hará el estudio de creación del modelo de aprendizaje automático. Por ejemplo, financiero, médico o biológico. Expliquen bien el contexto. Este contexto podría estar separado de la introducción como una sección de marco teórico. Pueden añadir cualquier revisión bibliográfica que hayan hecho donde describan trabajos similares acá en la introducción; también podría ser una sección separada o formar parte del estudio de factibilidad.

## Objetivos de negocio y criterios de éxito

Antes del estudio de factibilidad deben dejar claro **qué problema del dominio se quiere resolver y cómo se sabrá que se resolvió.** En CRISP-ML(Q) esto significa:

Enunciar el **objetivo de negocio** (o del dominio: clínico, biológico, financiero, etc.) en el lenguaje de la parte interesada.
**Traducirlo a un objetivo de aprendizaje automático:** ¿qué tarea es (clasificación, regresión, agrupamiento, detección de anomalías…)?, ¿cuál es la entrada?, ¿cuál es la salida?, ¿cuál es la unidad de predicción?
Definir **criterios de éxito medibles** para ambos niveles: una métrica de negocio (por ejemplo, reducción de costos, tiempo de diagnóstico, tasa de falsos negativos aceptable) y una métrica de aprendizaje automático que la refleje, con un valor umbral justificado y una línea base contra la cual comparar (el proceso actual, un modelo trivial o un trabajo previo de la literatura).

Un objetivo mal definido es el primer riesgo de la fase 1: si el criterio de éxito no es verificable, no hay forma de decidir después si el modelo sirve. Pueden apoyarse en un ML Canvas para ordenar estas ideas.

## Inventario y descripción de los datos

Describan los datos con los que trabajarían: origen, cantidad, formato, variables, etiquetas, período que cubren y cómo fueron generados o recolectados. Incluyan una descripción estadística preliminar (distribuciones, datos faltantes, desbalance de clases, posibles sesgos de muestreo). Documentar **el proceso que genera los datos**, y no solo el archivo, es parte del aseguramiento de calidad de esta fase.

## Estudio de factibilidad

Seguiría un estudio de factibilidad. Listen los recursos disponibles, por ejemplo expertos o poder computacional. Listen requerimientos, suposiciones y restricciones. Por último, analicen la razón costo-beneficio.

La sección de factibilidad debería tener información sobre:

- **Disponibilidad y calidad de los datos:** ¿hay suficientes datos disponibles para entrenar el modelo? ¿Son de calidad suficiente y representativos del fenómeno? ¿Se pueden utilizar datos sintéticos o aumentación de datos para reducir el costo?

- **Aplicabilidad:** ¿resolverá esta solución el problema o mejorará el proceso actual? ¿Se puede utilizar aprendizaje automático para resolver este problema? ¿Existe una solución más simple y no basada en aprendizaje automático que sea suficiente?

- **Limitaciones legales y éticas:** ¿existe permiso jurídico para implementar esta solución? ¿Se obtienen los datos de forma ética (consentimiento, anonimización, protección de datos personales)? ¿Cuál será el impacto de la aplicación en la sociedad? ¿Podría el modelo discriminar a algún grupo?

- **Robustez y escalabilidad:** ¿es la aplicación lo suficientemente robusta ante datos ruidosos, faltantes o fuera de distribución? ¿Es escalable en volumen de datos y en cantidad de solicitudes?

- **Explicabilidad:** ¿es posible explicar cómo el modelo de aprendizaje automático obtiene los resultados? Por ejemplo, ¿es posible explicar el funcionamiento interno de las redes neuronales profundas (véanse saliency maps, por ejemplo)? ¿El dominio exige explicaciones (por ejemplo, uso clínico o crediticio)?

- **Disponibilidad de recursos:** ¿hay suficientes recursos informáticos, de almacenamiento, de red y humanos?

- **Reproducibilidad:** ¿cómo se versionarán los datos, el código, el ambiente y las semillas aleatorias para que el resultado sea reproducible?

- **Mantenibilidad y monitoreo:** una vez desplegado, ¿cómo se detectaría la degradación del modelo por cambios en los datos o en el concepto? ¿Con qué frecuencia habría que reentrenarlo? ¿Existe la infraestructura y el personal para sostenerlo?

## Riesgos, contingencias y aseguramiento de calidad (Q)
CRISP-ML(Q) está orientado a riesgos: en cada tarea se identifica qué puede hacer fracasar el proyecto y c**ómo se comprobará que la tarea quedó bien hecha.** Por eso, además de listar riesgos y contingencias, deben indicar el método de calidad asociado a cada uno. Para cada riesgo indiquen:

- La fase de CRISP-ML(Q) en la que aparece (no todos los riesgos son de la fase 1: la fuga de información es de la fase 2, el sobreajuste de la 3, la degradación del modelo de la 6, etc.).
- Su **impacto** y su **probabilidad.**
- El **método de aseguramiento de calidad (Q):** la acción preventiva y el criterio verificable que permitirá decir que el riesgo está controlado.
- La **contingencia:** qué se hará si el riesgo se materializa de todos modos.

Se recomienda presentar esto como una tabla. Recuerden que en CRISP-ML(Q) los riesgos se gestionan de forma preventiva y no como reacción, y que si no se cumplen los criterios de calidad de una tarea, la tarea se repite.

## Objetivos
Lo siguiente en el documento es definir objetivos. Pueden definir objetivo general y objetivos específicos. Los objetivos específicos deberían ser coherentes con los criterios de éxito y con las fases del proceso que el proyecto va a cubrir.

## Plan
Por último, propongan un plan (usen las fechas de la carta al estudiante). Pueden usar un diagrama de Gantt. El plan debe organizarse según **las fases de CRISP-ML(Q)** e incluir el monitoreo y mantenimiento, aunque en el curso no se llegue a ejecutar esa fase: al menos describan qué implicaría. Tengan presente que el proceso es **iterativo y no lineal**, así que el plan debería contemplar regresos a fases anteriores. Consideren también la distribución típica del esfuerzo: las fases de datos (1 y 2) suelen consumir cerca de la mitad del tiempo, el modelado y la evaluación (3 y 4) alrededor de una cuarta parte, y el despliegue y la gestión posterior (5 y 6) cerca de una quinta parte.

No olviden unas conclusiones.

**El documento debe contar una historia (tener hilación).**

## Posible estructura del documento
De acuerdo a la explicación anterior del entregable, una posible estructura del informe podría ser:

1. Introducción
2. Revisión de literatura (opcional)
3. Marco teórico (opcional)
4. Comprensión del negocio y de los datos
	- Objetivos de negocio y criterios de éxito
	- Inventario y descripción de los datos
	- Estudio de factibilidad (con todas las subsecciones que consideren necesarias)
	- Riesgos, contingencias y aseguramiento de calidad (Q)
5. Objetivos (general y específicos)
6. Plan
7. Conclusiones