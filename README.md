Este proyecto se centra en aplicar segmentación semántica a imágenes multiespectrales de sensado remoto mediante técnicas de aprendizaje profundo con PyTorch.

Las imágenes multiespectrales contienen información más allá del espectro visible (RGB), abarcando bandas adicionales que permiten identificar con mayor precisión las características de los objetos observados. En este trabajo se ha adaptado una serie de modelos de redes neuronales profundas —inicialmente diseñados para imágenes RGB— para que funcionen eficazmente con este tipo de imágenes más complejas.

Tras evaluar distintas arquitecturas, se ha demostrado que modelos más ligeros como ResNet18dilated + C1 deepsup no solo son más eficientes computacionalmente, sino que también alcanzan una precisión competitiva del 87,74% en la segmentación semántica. Estos resultados validan la hipótesis de que, con la configuración adecuada, los modelos existentes pueden ser reutilizados en contextos multiespectrales, superando en algunos casos a modelos más pesados como ResNet101.

![image](https://github.com/user-attachments/assets/72122957-feb2-4371-ad93-b749ff7628cb)

En la imagen de ejemplo anterior se muestra (de izquierda a derecha):
Imagen original, máscara real y segmentación generada por el modelo.

Precisión obtenida: 87,74%

Para más detalles sobre el desarrollo técnico y los resultados, consulta el documento GID-semantic-segmentation.pdf.
Si deseas aprender a instalar, ejecutar o entrenar el sistema, dirígete directamente al Apéndice B - Manual de usuario incluido en ese mismo PDF.
