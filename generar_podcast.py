"""
Generador de podcast educativo sobre análisis acústico para monitoreo de ranas.
Voz: es-MX-JorgeNeural (masculina, mexicana)
Genera chunks separados y los combina en un MP3 final.
"""

import asyncio
import os
import sys
import subprocess

# ── Verificar/instalar dependencias ──────────────────────────────────────────
def check_and_install(package, import_name=None):
    import importlib
    name = import_name or package
    try:
        importlib.import_module(name)
        print(f"[OK] {package} ya está instalado.")
    except ImportError:
        print(f"[...] Instalando {package}...")
        os.system(f"{sys.executable} -m pip install {package} -q")

check_and_install("edge-tts", "edge_tts")

import edge_tts

# ── Directorio de trabajo ─────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local"
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "podcast_pipeline_acustico.mp3")
VOICE = "es-MX-JorgeNeural"
RATE = "+5%"

# ── Script del podcast dividido en secciones ─────────────────────────────────
SECCIONES = {
    "00_intro": """
Bienvenidos a Sonidos del Chaco, el podcast donde exploramos la ciencia detrás de escuchar la naturaleza.
Hoy vamos a hacer un viaje fascinante, desde los sonidos de las ranas en una reserva natural de Paraguay hasta los algoritmos de machine learning que nos ayudan a entenderlos.
Si alguna vez te preguntaste cómo es posible que una computadora aprenda a reconocer el canto de una rana entre miles de horas de grabación, este episodio es para ti.

Hablo desde el contexto de un proyecto real: el monitoreo acústico pasivo en la Reserva Natural Tapytá, en el departamento de Canindeyú, Paraguay.
La reserva alberga una biodiversidad extraordinaria, y las ranas son uno de sus componentes más sensibles a los cambios ambientales.
El problema es que monitorear manualmente la presencia de decenas de especies de anfibios escuchando grabaciones hora tras hora es una tarea humanamente imposible cuando hablamos de meses de registros continuos.

Por eso nuestro equipo desarrolló un pipeline de análisis acústico automatizado.
La idea es dejar que la computadora haga el trabajo pesado: analizar, agrupar, clasificar, y luego presentarle al experto humano esos grupos para que los interprete.
Hoy voy a explicarte exactamente cómo funciona ese proceso, paso a paso, con analogías concretas y sin pretender que ya sabes estadística avanzada.

Hay cuatro grandes bloques en este pipeline: primero, los MFCCs, que son la forma en que transformamos el sonido en números; segundo, el PCA, que reduce esos números a lo más esencial; tercero, UMAP, que nos permite visualizar todo en tres dimensiones; y cuarto, HDBSCAN, el algoritmo que encuentra los grupos automáticamente.
Vamos uno por uno.
""",

    "01_mfccs_parte1": """
Bloque uno: los MFCCs, o cómo convertir el sonido en una huella digital.

Para entender los MFCCs, primero necesitamos hablar de lo básico: qué es un sonido.
Cuando una rana canta, hace vibrar el aire a su alrededor.
Esas vibraciones llegan a nuestro micrófono y se convierten en una señal digital: una lista enorme de números que representan la presión del aire en cada milisegundo.
Si graficamos esos números en el tiempo, obtenemos lo que se llama una forma de onda.
La forma de onda nos dice cuánto sube y baja el sonido a lo largo del tiempo, pero no nos dice mucho sobre qué tipo de sonido es.

¿Cómo distinguimos el canto de una rana del ruido de la lluvia? Para eso necesitamos mirar las frecuencias.
Una frecuencia es básicamente la velocidad a la que vibra el aire.
Los sonidos graves, como el de un tambor, tienen frecuencias bajas: el aire vibra lentamente.
Los sonidos agudos, como el silbido de un grillo, tienen frecuencias altas: el aire vibra muy rápido.
La mayoría de los sonidos naturales son en realidad una mezcla de muchas frecuencias al mismo tiempo.

Para descomponer un sonido en sus frecuencias, usamos algo llamado la Transformada de Fourier.
Sin entrar en la matemática, la idea es esta: la Transformada de Fourier toma la señal de audio y nos dice exactamente cuánta energía hay en cada frecuencia.
Es como pasar la luz blanca a través de un prisma y ver el arco iris: el prisma no crea los colores, los separa.
La Transformada de Fourier hace lo mismo con el sonido: separa las frecuencias que ya están ahí.

Cuando hacemos esto en pequeñas ventanas de tiempo, milisegundo a milisegundo, obtenemos lo que se llama un espectrograma.
Un espectrograma es como una fotografía del sonido: en el eje horizontal está el tiempo, en el eje vertical están las frecuencias, y el brillo de cada punto indica cuánta energía hay en esa frecuencia en ese momento.
Si has visto alguna vez esas imágenes coloridas del canto de los pájaros, estabas mirando un espectrograma.
Los espectrogramas de las ranas son preciosos: cada especie tiene un patrón visual único, como una firma.
""",

    "02_mfccs_parte2": """
Pero hay un problema con usar el espectrograma directamente para comparar sonidos.
Tiene demasiados datos.
Un espectrograma típico puede tener cientos de filas de frecuencias para cada fracción de segundo de audio.
Necesitamos una forma más compacta de representar el sonido, y aquí es donde entra la escala Mel y los MFCCs.

El oído humano no percibe las frecuencias de manera lineal.
Eso significa que la diferencia entre cien y doscientos hertz, que es cien hertz, nos parece mucho más grande que la diferencia entre cinco mil y cinco mil cien hertz, que también es cien hertz pero en una zona alta.
En las frecuencias bajas somos muy sensibles a los cambios; en las altas, mucho menos.
La escala Mel captura exactamente eso: es una escala de frecuencias diseñada para imitar la percepción auditiva humana.
En lugar de distribuir las frecuencias uniformemente, las comprime en las zonas altas y las expande en las bajas.

Los MFCCs, que en inglés significan Mel-Frequency Cepstral Coefficients, o coeficientes cepstrales en frecuencia Mel, aprovechan esta escala.
El proceso tiene varios pasos que ocurren automáticamente.
Primero, se divide el audio en pequeñas ventanas de tiempo, en nuestro caso de cuatro segundos.
Luego se calcula el espectrograma de esa ventana.
Después se aplica el filtro de escala Mel para comprimir las frecuencias de manera que imiten la percepción auditiva.
Finalmente, se aplica una transformación matemática llamada la transformada coseno discreta, que convierte ese espectrograma en un conjunto pequeño de coeficientes.

El resultado es que cada segmento de audio, de cuatro segundos, queda representado por solo veinte números.
Esos veinte números, más sus veinte desviaciones estándar, nos dan cuarenta números en total por segmento.
¿Qué capturan esos cuarenta números? Capturan el timbre o la textura del sonido: no la melodía, no el ritmo, sino la calidad tonal intrínseca de ese fragmento de audio.
Piensa en la diferencia entre un violín y una flauta tocando la misma nota.
La nota es la misma, la frecuencia fundamental es la misma, pero suenan completamente distinto.
Esa diferencia, ese timbre, es lo que los MFCCs codifican.

Para nuestras ranas, eso es perfecto.
El canto de una rana Physalaemus tiene un timbre completamente distinto al del Elachistocleis, aunque ambas canten en frecuencias similares.
Los MFCCs capturan esa diferencia de manera compacta y matemáticamente manejable.

La analogía que más me gusta es esta: los MFCCs son como la huella digital del sonido.
Así como una huella digital no te dice qué hizo la persona, sino quién es, los MFCCs no te dicen qué dijo el sonido, sino cómo suena.
Y así como dos personas distintas tienen huellas distintas, dos especies de ranas distintas tienen MFCCs distintos.
Eso es exactamente lo que necesitamos para que la computadora pueda agruparlas.

Entonces en este punto del pipeline, tenemos miles de segmentos de audio, cada uno representado por cuarenta números.
El siguiente problema es que cuarenta dimensiones son demasiadas para trabajar directamente con los algoritmos de agrupamiento.
Necesitamos reducir esa dimensionalidad, y para eso usamos el PCA.
""",

    "03_pca": """
Bloque dos: el PCA, o cómo encontrar las dimensiones que realmente importan.

Imagina que tienes una escultura tridimensional y quieres hacer una fotografía de ella en dos dimensiones.
La fotografía siempre va a perder algo de la escultura, eso es inevitable.
Pero hay ángulos mejores que otros.
Si fotografías una esfera desde cualquier lado, obtienes un círculo y no pierdes mucho.
Pero si fotografías una figura alargada desde el costado, ves toda su extensión.
Si la fotografías desde el frente, solo ves un punto.
El truco está en encontrar el ángulo que conserva la mayor cantidad de información posible.

Eso es exactamente lo que hace el Análisis de Componentes Principales, el PCA.
Tenemos nuestros datos en cuarenta dimensiones, lo que es imposible de visualizar directamente.
El PCA busca las direcciones, llamadas componentes principales, en ese espacio de cuarenta dimensiones donde los datos varían más.
Dicho de otra manera, busca los ejes donde hay más diferencias entre los puntos.
Los ejes donde todos los puntos están muy juntos no nos dicen mucho.
Los ejes donde los puntos están muy separados son los que contienen la información más interesante.

El primer componente principal es la dirección de máxima varianza: el eje donde los datos se dispersan más.
El segundo componente principal es la dirección de máxima varianza que sea perpendicular al primero.
El tercero es perpendicular a los dos anteriores.
Y así sucesivamente.
Lo notable es que en la mayoría de los conjuntos de datos reales, los primeros pocos componentes principales capturan la gran mayoría de la varianza total.
Los últimos componentes suelen representar ruido o variaciones mínimas.

En nuestro proyecto, reducimos de cuarenta dimensiones a veinte usando PCA.
Esas veinte dimensiones capturan la mayor parte de la varianza de los MFCCs, descartando las dimensiones que aportan poco.

Hay un detalle técnico muy importante que vale la pena mencionar: el whitening, o blanqueamiento.
Cuando el PCA transforma los datos, los componentes principales que resultan pueden tener escalas muy distintas.
El primer componente puede tener varianza cien, el segundo varianza diez, el tercero varianza uno.
Esa diferencia de escala puede ser un problema para los algoritmos que vienen después, porque los que tienen escala grande dominan el cálculo de distancias.
El whitening normaliza las varianzas de todos los componentes para que queden iguales, en uno.
Así, todas las dimensiones tienen el mismo peso.

Ahora, ¿por qué hacer PCA antes de UMAP?
UMAP es el algoritmo que viene después, y es el que va a hacer la reducción final a tres dimensiones.
UMAP funciona calculando relaciones de vecindad entre puntos: quién está cerca de quién.
El problema es que en espacios de muchas dimensiones, el concepto de distancia se vuelve raro.
Este fenómeno tiene un nombre elegante: la maldición de la dimensionalidad.
En cuarenta dimensiones, todos los puntos tienden a estar aproximadamente a la misma distancia de todos los demás.
Las diferencias se aplanan.
UMAP tiene mucha dificultad para encontrar estructura en ese contexto.

Al hacer PCA primero, con whitening, le estamos dando a UMAP datos más limpios.
Los datos tienen menos dimensiones redundantes, las escalas están normalizadas, y las relaciones de vecindad son más significativas.
Es como limpiar el terreno antes de construir.
El PCA no tiene que encontrar los clusters, solo tiene que preparar los datos para que UMAP pueda hacerlo bien.
""",

    "04_umap_parte1": """
Bloque tres: UMAP, o cómo ver veinte dimensiones en tres.

Tenemos nuestros datos en veinte dimensiones después del PCA.
Todavía es demasiado para visualizar o para que HDBSCAN trabaje de manera eficiente.
Necesitamos comprimirlos a tres dimensiones.
Para eso usamos UMAP, que significa Uniform Manifold Approximation and Projection.
El nombre suena intimidante, pero la idea detrás es bastante intuitiva.

Imagina que vives en un planeta esférico.
Ese planeta es una superficie bidimensional, pero vive en un espacio tridimensional.
Si quieres hacer un mapa plano del planeta, tienes un problema: no puedes aplanar una esfera sin distorsionarla.
Por eso los mapas del mundo siempre mienten un poco.
Los mapas de Mercator, por ejemplo, hacen que Groenlandia parezca del tamaño de África, cuando en realidad es mucho más pequeña.
Cada proyección cartográfica elige qué distorsionar y qué conservar.
Algunos conservan las áreas, otros conservan las formas, otros conservan las distancias locales.

UMAP hace algo parecido con nuestros datos.
Nuestros datos en veinte dimensiones forman una especie de superficie, técnicamente llamada una variedad o manifold, en ese espacio de alta dimensión.
UMAP intenta encontrar una representación en tres dimensiones que conserve las relaciones de vecindad de los datos originales.
No puede conservar todo perfectamente, pero hace el mejor trabajo posible para que los puntos que eran vecinos en veinte dimensiones sigan siendo vecinos en tres.

¿Cómo funciona UMAP internamente? Sin entrar en las matemáticas, el proceso tiene dos grandes pasos.
Primero, construye un grafo de vecindad: para cada punto, identifica sus vecinos más cercanos y crea conexiones entre ellos.
El resultado es una red donde los puntos similares están conectados entre sí.
Segundo, optimiza una representación en baja dimensión que preserva ese grafo lo mejor posible.
Es un proceso iterativo: empieza con una distribución aleatoria de los puntos en tres dimensiones y los va ajustando hasta que la estructura local del grafo original queda reflejada.
""",

    "05_umap_parte2": """
Ahora voy a explicarte los tres parámetros clave que usamos en nuestro proyecto y por qué los elegimos así.

El primero es n_neighbors, que en nuestro caso vale sesenta.
Este parámetro controla cuántos vecinos considera UMAP al construir el grafo de vecindad.
Si n_neighbors es pequeño, digamos cinco, cada punto solo mira sus cinco vecinos más inmediatos.
UMAP captura detalles muy locales pero puede perder la estructura global de los datos.
Los grupos grandes pueden fragmentarse en grupos pequeños que no reflejan la realidad biológica.
Si n_neighbors es grande, digamos doscientos, cada punto mira muchos vecinos y UMAP captura la estructura global a expensas de los detalles finos.
Con sesenta vecinos, en un dataset de bioacústica como el nuestro donde esperamos relativamente pocos clusters bien definidos, estamos priorizando que la estructura global sea correcta.
Queremos que el cluster de la Physalaemus aparezca junto y coherente, aunque eso signifique perder algunos detalles internos del cluster.

El segundo parámetro es min_dist, que vale cero punto tres.
Este parámetro controla qué tan compactos quedan los grupos en el espacio tridimensional resultante.
Con min_dist igual a cero, UMAP aprieta todos los puntos de un mismo grupo en prácticamente el mismo lugar.
Los clusters quedan como puntos perfectamente compactos, pero pierdes toda la estructura interna.
Con min_dist igual a uno, los puntos quedan muy dispersos y los grupos se desdibujan.
Cero punto tres es un balance razonable: los grupos quedan compactos y separados entre sí, pero con suficiente estructura interna como para ver subgrupos o gradientes dentro de cada cluster.
En bioacústica esto es útil: si dentro del cluster de una especie hay variación entre macho y hembra, o entre distintos contextos de canto, quieres que eso sea visible.

El tercer parámetro es la métrica de distancia, que en nuestro proyecto es coseno.
Este es quizás el parámetro más interesante desde el punto de vista conceptual.
La distancia coseno entre dos vectores no mide qué tan lejos están en el espacio, sino el ángulo entre ellos.
Dos vectores con el mismo ángulo tienen distancia coseno cero, sin importar cuánto de largos sean.
¿Por qué es esto importante para audio?

Imagina que grabas el mismo canto de una rana dos veces.
En la primera grabación, el micrófono estaba cerca y el sonido quedó fuerte.
En la segunda, el micrófono estaba lejos y el sonido quedó suave.
Los vectores de MFCCs de ambas grabaciones van a apuntar en la misma dirección en el espacio de cuarenta dimensiones, pero el de la grabación fuerte va a ser más largo.
Si usas distancia euclidiana, que mide la longitud del vector diferencia, esas dos grabaciones van a parecer distintas, aunque suenen igual.
Si usas distancia coseno, solo te importa el ángulo, y dos grabaciones del mismo sonido pero a distinto volumen van a tener distancia cero.
En campo, el volumen de las grabaciones varía enormemente dependiendo de la distancia al micrófono, del viento, de la lluvia, de muchos factores.
La distancia coseno hace que nuestra comparación sea robusta a esas variaciones de volumen.

El resultado final de UMAP es que cada segmento de audio, que antes era un vector de veinte números, ahora es un punto en el espacio tridimensional.
Puntos que suenan parecido están cerca.
Puntos que suenan distinto están lejos.
Y podemos visualizar todo esto en una nube de puntos tridimensional que de repente hace visible la estructura acústica de meses de grabación.
""",

    "06_hdbscan_parte1": """
Bloque cuatro: HDBSCAN, o cómo encontrar los grupos automáticamente.

Llegamos al último gran paso del pipeline.
Tenemos nuestros datos en tres dimensiones gracias a UMAP.
Podemos verlos, podemos intuir que hay grupos.
Pero necesitamos que la computadora los identifique formalmente, sin que nosotros le digamos cuántos grupos hay ni dónde están.
Para eso usamos HDBSCAN, que significa Hierarchical Density-Based Spatial Clustering of Applications with Noise.
Es un nombre largo para una idea bastante elegante.

Primero, entendamos por qué no usamos los métodos de clustering más clásicos, como K-means.
K-means es el algoritmo de agrupamiento más conocido, el que se enseña en todos los cursos introductorios.
Tiene dos limitaciones que son problemáticas para nuestro caso.
La primera: necesitas decirle de antemano cuántos grupos quieres.
En bioacústica, no sabemos cuántas especies van a aparecer en las grabaciones.
Puede haber cinco, puede haber veinte.
Forzar un número fijo de clusters es arbitrario y potencialmente muy engañoso.
La segunda limitación: K-means asigna todos los puntos a algún cluster, sin excepción.
No tiene el concepto de ruido o de puntos que no pertenecen a ningún grupo.
En un dataset de campo, donde hay lluvia, viento, insectos, camiones en la distancia, eso es un problema serio.
Queremos poder decir que ciertos segmentos son ruido ambiental y no asignarlos artificialmente a un cluster de ranas.

HDBSCAN resuelve ambos problemas.
No necesita que le digas cuántos clusters hay.
Y puede etiquetar puntos como ruido, es decir, asignarles el cluster menos uno, que es la convención para indicar que ese punto no pertenece a ningún grupo bien definido.

¿Cómo funciona HDBSCAN? La intuición detrás es la noción de densidad.
Un cluster es una región del espacio donde los puntos están más densamente concentrados que en las regiones circundantes.
Imagina un mapa de ciudades del mundo.
Las ciudades son regiones de alta densidad de población.
Entre ellas hay campo, océano, desierto, que son regiones de baja densidad.
Si alguien te da ese mapa y te dice que encuentres las ciudades, lo harías buscando regiones densas.
HDBSCAN hace exactamente eso, pero en nuestro espacio tridimensional de sonidos.
""",

    "07_hdbscan_parte2": """
La parte jerárquica de HDBSCAN, la H al inicio del nombre, es lo que lo hace especialmente poderoso.
En lugar de buscar clusters a una densidad fija, HDBSCAN construye una jerarquía de clusters a distintas escalas de densidad.

La analogía que me parece más clara es esta: imagina que eres un aviador y vuelas sobre el mapa del mundo a distintas altitudes.
A veinte mil metros de altura, solo ves los continentes, grandes manchas de tierra.
A diez mil metros, empiezan a aparecer los países.
A cinco mil, las ciudades grandes.
A mil, los barrios.
A cien metros, las manzanas.
Cada altitud te revela una escala distinta de organización.

HDBSCAN hace algo análogo: construye un árbol jerárquico donde los clusters se fusionan y se dividen a distintas escalas de densidad.
A densidades muy altas, solo existen grupos muy densos y pequeños.
A densidades más bajas, esos grupos se fusionan en grupos más grandes.
El árbol completo captura toda esa jerarquía.

Luego viene el paso de selección de clusters, y aquí es donde entra el parámetro cluster_selection_method.
Usamos el método llamado eom, que significa Excess of Mass, o exceso de masa.
Para entenderlo, piensa en el árbol jerárquico que construyó HDBSCAN.
Hay muchas formas distintas de cortar ese árbol para obtener clusters.
Si cortas muy arriba, obtienes pocos clusters grandes.
Si cortas muy abajo, obtienes muchos clusters pequeños.
El método eom no hace un corte a una altura fija.
En cambio, evalúa cada posible cluster en el árbol y calcula cuánta masa, es decir, cuántos puntos, gana o pierde al fusionarse con otros clusters.
Elige los clusters que maximizan la masa total estable.
El resultado es que puede encontrar clusters de tamaños y densidades muy distintas en el mismo dataset.

¿Por qué es esto importante para nosotros? Porque biológicamente, algunas especies de ranas son muy abundantes en Tapytá y van a generar miles de segmentos de audio.
Otras especies son raras o cantan solo en ciertos períodos del año, y van a generar muchos menos segmentos.
Un algoritmo que exige que todos los clusters sean del mismo tamaño va a perder las especies raras.
El método eom de HDBSCAN puede encontrar tanto el cluster enorme de la especie abundante como el cluster pequeño de la especie rara, en la misma corrida.

El parámetro min_cluster_size vale ochocientos en nuestro proyecto.
Eso significa que un cluster tiene que tener al menos ochocientos segmentos de cuatro segundos para ser reconocido como tal.
Ochocientos segmentos son aproximadamente cincuenta y tres minutos de audio de una misma especie.
Este umbral evita que ruidos esporádicos o sonidos únicos que aparecen solo una o dos veces formen clusters ficticios.
Si hay un evento de audio que aparece cien veces, probablemente es un ruido raro, no una especie.
Si aparece ochocientas veces o más, hay buenas razones para pensar que es un sonido recurrente que merece ser investigado.

El cluster menos uno, que HDBSCAN etiqueta como ruido, es completamente esperado en bioacústica de campo.
La lluvia, el viento, los insectos que no agrupan en ninguna especie específica, los camiones en la distancia, los aviones, todos esos sonidos van a terminar en el cluster de ruido.
Eso está bien.
No son errores del algoritmo, son parte de la realidad acústica del campo.
El hecho de que HDBSCAN tenga ese concepto de ruido es una de sus grandes ventajas sobre métodos que obligan a clasificar todo.
""",

    "08_cierre": """
Y con eso llegamos al final del pipeline.

Repasemos el flujo completo de principio a fin.
Empezamos con señales de audio: horas y horas de grabación de la Reserva Natural Tapytá.
Dividimos esas grabaciones en segmentos de cuatro segundos.
A cada segmento le calculamos veinte coeficientes cepstrales en frecuencia Mel más sus veinte desviaciones estándar, obteniendo cuarenta números que capturan el timbre del sonido, su huella digital acústica.
Luego aplicamos PCA para reducir esos cuarenta números a veinte componentes principales, estandarizados y sin redundancias, que conservan la mayor parte de la varianza.
Después UMAP toma esos veinte números y los comprime a tres coordenadas en el espacio, preservando las relaciones de vecindad y usando distancia coseno para ser robusto a las variaciones de volumen.
Finalmente, HDBSCAN analiza esa nube de puntos tridimensional y encuentra automáticamente los grupos de sonidos similares, asignando a cada segmento una etiqueta de cluster o la etiqueta de ruido si no pertenece a ningún grupo.

El resultado es que miles de horas de audio quedan organizadas en grupos coherentes.
Cada grupo, idealmente, corresponde a un tipo de sonido recurrente: una especie de rana, un insecto, lluvia fuerte, viento.
Un experto en anfibios puede entonces escuchar una muestra de cada grupo y decir: este es el canto de Leptodactylus latinasus, este es Physalaemus albonotatus, este es ruido de lluvia.
Y esa identificación, hecha por el experto humano en unos pocos minutos, se puede propagar automáticamente a todos los miles de segmentos de ese cluster.

Eso es lo verdaderamente poderoso de este enfoque.
La máquina no reemplaza al experto, sino que multiplica su eficiencia.
El experto ya no tiene que escuchar cincuenta mil segmentos uno por uno.
Solo tiene que identificar quince o veinte grupos, y automáticamente se etiquetan todos los segmentos.

¿Por qué importa esto para la conservación de la biodiversidad?
Las ranas son consideradas bioindicadores ambientales.
Son muy sensibles a los cambios en temperatura, humedad, calidad del agua y fragmentación del hábitat.
Si una especie de rana desaparece o reduce su actividad en una reserva, es una señal temprana de que algo está cambiando en ese ecosistema.
Pero detectar esos cambios requiere monitoreo sistemático a largo plazo, y el monitoreo manual es costoso y lento.

El pipeline que hemos descrito hoy permite detectar automáticamente qué especies están presentes en cada sitio de grabación, en qué momentos del día o del año son más activas, y cómo varía eso entre distintos sitios de la reserva.
Con esos datos, los conservacionistas pueden tomar decisiones informadas sobre dónde enfocar los esfuerzos de protección, qué áreas están en mejor estado ecológico y cuáles necesitan intervención.

La acústica pasiva, combinada con machine learning, está transformando la forma en que monitoreamos la biodiversidad.
No solo para ranas, sino para aves, murciélagos, cetáceos, insectos.
Y lo que hoy requiere servidores y algoritmos sofisticados, mañana va a ser accesible para cualquier grupo de investigación con un grabador económico y un presupuesto modesto.

Eso es todo por hoy en Sonidos del Chaco.
Si tienes preguntas sobre alguno de los conceptos que discutimos, ya sea MFCCs, PCA, UMAP o HDBSCAN, puedes escribirnos a través de nuestras redes sociales.
La próxima semana vamos a hablar sobre cómo validar los clusters que encontramos y cómo integrar las identidades de las especies con los datos de variables ambientales como temperatura y precipitación.
Hasta entonces, sigan escuchando la naturaleza.
"""
}

# ── Función para generar cada sección ────────────────────────────────────────
async def generar_seccion(nombre, texto, directorio):
    ruta = os.path.join(directorio, f"chunk_{nombre}.mp3")
    if os.path.exists(ruta):
        print(f"  [SKIP] {nombre} ya existe, omitiendo.")
        return ruta
    print(f"  [GEN]  Generando {nombre}...")
    communicate = edge_tts.Communicate(texto.strip(), VOICE, rate=RATE)
    await communicate.save(ruta)
    size_kb = os.path.getsize(ruta) / 1024
    print(f"  [OK]   {nombre} -> {size_kb:.0f} KB")
    return ruta

async def main():
    print("=" * 60)
    print("PODCAST: Pipeline Acústico para Monitoreo de Ranas")
    print("=" * 60)
    print(f"Voz: {VOICE}  |  Rate: {RATE}")
    print(f"Directorio: {OUTPUT_DIR}\n")

    # Generar cada sección
    chunks_dir = os.path.join(OUTPUT_DIR, "podcast_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    rutas = []
    for nombre, texto in SECCIONES.items():
        ruta = await generar_seccion(nombre, texto, chunks_dir)
        rutas.append(ruta)

    # Combinar con ffmpeg (concat demuxer)
    print(f"\nCombinando {len(rutas)} chunks con ffmpeg...")
    lista_ffmpeg = os.path.join(chunks_dir, "lista_chunks.txt")
    with open(lista_ffmpeg, "w", encoding="utf-8") as f:
        for ruta in rutas:
            ruta_posix = ruta.replace("\\", "/")
            f.write(f"file '{ruta_posix}'\n")
            # Pausa de 0.8s: generamos un chunk de silencio si no existe
            silencio_path = os.path.join(chunks_dir, "silencio.mp3").replace("\\", "/")
            if not os.path.exists(silencio_path):
                subprocess.run([
                    "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
                    "-t", "0.8", "-q:a", "9", "-acodec", "libmp3lame",
                    silencio_path, "-y"
                ], capture_output=True)
            f.write(f"file '{silencio_path}'\n")

    result = subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", lista_ffmpeg,
        "-c", "copy", FINAL_OUTPUT, "-y"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR ffmpeg] {result.stderr}")
        sys.exit(1)

    # Calcular duración con ffprobe
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", FINAL_OUTPUT
    ], capture_output=True, text=True)
    duracion_seg = float(probe.stdout.strip()) if probe.stdout.strip() else 0
    minutos = int(duracion_seg // 60)
    segundos = int(duracion_seg % 60)

    size_mb = os.path.getsize(FINAL_OUTPUT) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("=" * 60)
    print(f"Archivo: {FINAL_OUTPUT}")
    print(f"Duracion: {minutos} min {segundos} seg")
    print(f"Tamanio:  {size_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
