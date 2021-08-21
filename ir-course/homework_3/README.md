# Index Construction

## Run the program
* Type
```
# install nltk
pip3 install nltk

# execute program
python3 main.py --path PATH_TO_CRANFIELD_DIRECOTRY
```
* The only argument is **--path** that accepts the path to the CRANFIELD directory/folder.

## Code flow
- Each text file is processed by either lemmatizer or stemmer
- Stop-word are removed and not stored in dictionary. However, the stop-word occurences are counted to document length
- Each text file is then parsed into a **Document** object that holds document length, maximum frequency, and document id, and word frequency dictionary
- All **Document** objects are then merged together into a single **Index** object for either "lemma" or "stem"
- Then **Index** objects are inverted (alphebetically sorted). The returns of "invert" is 2 objects: **Posting** and **Dictionary** for the correponding posting list and dictionary
- Then **Posting** and **Dictionary** objects are compressed based on the given requirements. The results of compression are stored into separate variables within the objects. This is done purposely because the retrieval mechanisms for compressed posting lists and dictionarys are not implemented yet.
- Then, 2. Query class objects for w_1 and w_2 weighting functions are created to compute the cosine similarity score between query and document vectors. 
- In both weighting functions, collection_size is the number of documents in the collection and df is the  document frequency for each token. If the token does not exist in the document, then df = 1.
- For query vector:
	- W_1 weighting function: tf is the term frequency in the query; maxtf is the maximum frequency in the query.
	- w_2 weighting function: tf is the term frequency in the query; doclen is the length of query; avgdoclen is the average length of all documents in the collection.
- For document vector:
	- w_1 weighting function: tf is the term frequency in the document; maxtf is the maximum frequency in the query.
	- w_2 weighting function: tf is the term frequency in the document; doclen is the length of the document; avgdoclen is the average length of all documents in the collection.
- Finally, documents are sorted by the cosine similarity and displayed with corresponding query-document vectors. For each displayed document, its headline is printed as well.

## Design decision to build ranking system
- The query vector is computed only once because the query vector should be independent from documents
- For each document in the collection, a corresponding document vector and its cosine similar are computed. This design may slow the ranking time because it linearly iterates through the collection.
- To accelerate the ranking process, the collection should be partitioned into smaller chunks where each chunk is processed a dedicated processor. By this, multi-processing could be applied for faster ranking. Then, documents in each chunk are sorted separately and merged into a global rank at the end.

## Why the top-ranked non-relevant document for each query did not get a lower score?
* The implemented weighting functions focus the term-frequency and document-frequency. This means the weighting functions based on the frequency distribution and ignores the word dependency (aka context). 
* Given that, more words in a document may be ranked as a top candidate despite the fact that the document is not relevantant to the query.

## Discuss the different effects you notice with the two weighting schemes, either on a query-by-query basis or overall, whichever is most illuminating
- Overall, the w_1 function bases on the frequency of the most-indexed word. In other words, if documents have many distinct words, then the document vectors among documents are equal. If documents have a term's frequency than other terms, then document vectors are greatly different among documents.
- W_2 function bases on the document length. Hence, the longer a document is, the higher the document is ranked.

## Results
```
For weighting 1 function

Query Q1:
what similarity laws must be obeyed when constructing aeroelastic models
of heated high speed aircraft  has vector = [1.1931568569324176, 0.5488290713472348, 0.5488290713472348, 0.5488290713472348, 1.1931568569324176, 1.1931568569324176, 1.1931568569324176, 0.8980460128737068, 0.7364990230545483, 0.3415548857006682, 1.1931568569324176, 0.6629935298167022, 0.336033547534164, 0.28507034259726477, 0.4777125651702375]

Top 5 documents is:

Document 1 with rank 1 has cosine similarity score = 1.0000000000000002
Document 1 has the headline: inviscid-incompressible-flow theory of static two-dimensionalsolid jets, in proximity to the ground .

Document 13 with rank 2 has cosine similarity score = 1.0000000000000002
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 22 with rank 3 has cosine similarity score = 1.0000000000000002
Document 22 has the headline: concerning the effect of compressibility on laminarboundary layers and their separation .

Document 35 with rank 4 has cosine similarity score = 1.0000000000000002
Document 35 has the headline: similar solutions for the compressible boundary layeron a yawed cylinder with transpiration cooling .

Document 38 with rank 5 has cosine similarity score = 1.0000000000000002
Document 38 has the headline: formulae for use with the fatigue load meter in theassessment of wing fatigue life .

Query Q2:
what are the structural and aeroelastic problems associated with flight
of high speed aircraft  has vector = [1.1931568569324176, 1.1931568569324176, 1.1931568569324176, 0.5984224345234445, 1.1931568569324176, 0.7364990230545483, 0.2415515309571643, 0.5006250694763165, 1.1931568569324176, 0.3979118662588444, 1.1931568569324176, 0.336033547534164, 0.28507034259726477, 0.4777125651702375]

Top 5 documents is:

Document 53 with rank 1 has cosine similarity score = 1.0000000000000004
Document 53 has the headline: measurement of two dimensional derivatives on a wing-aileron-tabsystem .

Document 1052 with rank 2 has cosine similarity score = 1.0000000000000004
Document 1052 has the headline: theories of plastic buckling .

Document 5 with rank 3 has cosine similarity score = 1.0000000000000002
Document 5 has the headline: buckling of sandwich under normal pressure .

Document 13 with rank 4 has cosine similarity score = 1.0000000000000002
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 21 with rank 5 has cosine similarity score = 1.0000000000000002
Document 21 has the headline: transonic flow in two dimensional and axially symmetricalnozzles .

Query Q3:
what problems of heat conduction in composite slabs have been solved so
far  has vector = [1.1931568569324176, 0.2415515309571643, 1.1931568569324176, 0.2891041237551904, 0.5940300542731123, 1.1931568569324176, 0.8139107349050925, 0.7471288077853163, 1.1931568569324176, 1.1931568569324176, 0.5488290713472348, 1.1931568569324176, 0.5897517748226879]

Top 5 documents is:

Document 13 with rank 1 has cosine similarity score = 1.0000000000000004
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 53 with rank 2 has cosine similarity score = 1.0000000000000004
Document 53 has the headline: measurement of two dimensional derivatives on a wing-aileron-tabsystem .

Document 139 with rank 3 has cosine similarity score = 1.0000000000000004
Document 139 has the headline: the effects of a small jet of air exhausting from the nose of a bodyof revolution in supersonic flow .

Document 419 with rank 4 has cosine similarity score = 1.0000000000000004
Document 419 has the headline: pressure loads produced on a flat-plate wing by rocket jets exhaustingin a spanwise direction below the wing and perpendicular to a freestreamflow of mach number 2.0 .

Document 539 with rank 5 has cosine similarity score = 1.0000000000000004
Document 539 has the headline: atmosphere entries with spacecraft lift-drag ratiosmodulated to limit decelerations .

Query Q4:
can a criterion be developed to show empirically the validity of flow
solutions for chemically reacting gas mixtures based on the simplifying
assumption of instantaneous local chemical equilibrium  has vector = [0.9004262602878819, 0.9004262602878819, 0.46211484436840433, 0.9004262602878819, 0.2577573825500192, 0.9004262602878819, 0.2751645059581678, 0.7638734874130577, 1.0841883973539788, 0.4954589017850283, 1.0841883973539788, 0.08404248184997314, 0.15325165681475583, 0.9004262602878819, 0.6273207145382332, 0.6585580149026898, 0.27845729958930754, 0.528069649166831, 0.25088846975211393, 0.9004262602878819, 0.5724028075632779, 0.31731193647536693, 0.6273207145382332, 0.31505195257229723, 0.4516047254471216, 0.35732272290648925]

Top 5 documents is:

Document 967 with rank 1 has cosine similarity score = 0.9973982309089124
Document 967 has the headline: the use of models for the determination of critical flutter speeds .

Document 576 with rank 2 has cosine similarity score = 0.9973982309089122
Document 576 has the headline: response of plates to a decaying and convecting randonpressure field .

Document 789 with rank 3 has cosine similarity score = 0.9973982309089122
Document 789 has the headline: wing-flow study of pressure drag reduction at transonic speed by projectinga jet of air from the nose of a prolate spheroid of finenessratio 6 .

Document 891 with rank 4 has cosine similarity score = 0.9973982309089122
Document 891 has the headline: stability of rectangular plates under shear and bendingforces .

Document 1334 with rank 5 has cosine similarity score = 0.9973982309089122
Document 1334 has the headline: an investigation of the noise produced by a subsonic air jet .

Query Q5:
what chemical kinetic system is applicable to hypersonic aerodynamic
problems  has vector = [1.1931568569324176, 0.5984224345234445, 0.6840482611090702, 0.45950582123042816, 1.1931568569324176, 0.4864988881468708, 1.1931568569324176, 0.3472677531196771, 0.35932468303705756, 0.2415515309571643]

Top 5 documents is:

Document 12 with rank 1 has cosine similarity score = 1.0000000000000004
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 37 with rank 2 has cosine similarity score = 1.0000000000000004
Document 37 has the headline: note on creep buckling of columns .

Document 39 with rank 3 has cosine similarity score = 1.0000000000000004
Document 39 has the headline: some current and proposed investigations into the flowfor slender delta and other wings in unsteady motion .

Document 67 with rank 4 has cosine similarity score = 1.0000000000000004
Document 67 has the headline: the transition to tubulence in a boundary layer ona blunt cone in supersonic flow .

Document 77 with rank 5 has cosine similarity score = 1.0000000000000004
Document 77 has the headline: investigation of a retrocket exhausting from the nose of a blunt bodyinto a supersonic free stream .

Query Q6:
what theoretical and experimental guides do we have as to turbulent
couette flow behaviour  has vector = [1.1931568569324176, 0.3063063002032549, 1.1931568569324176, 0.2446398307758126, 0.7584922440546503, 1.1931568569324176, 1.1931568569324176, 1.1931568569324176, 1.1931568569324176, 1.1931568569324176, 0.4005684480139413, 0.7982127195785376, 0.1113648811851109, 0.6443277855851828]

Top 5 documents is:

Document 884 with rank 1 has cosine similarity score = 1.0000000000000004
Document 884 has the headline: oscillatory aerodynamic coefficients for a unified supersonichypersonic strip theory .

Document 12 with rank 2 has cosine similarity score = 1.0000000000000002
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 69 with rank 3 has cosine similarity score = 1.0000000000000002
Document 69 has the headline: an approximate analytical method for studying entryinto planetary atospheres .

Document 90 with rank 4 has cosine similarity score = 1.0000000000000002
Document 90 has the headline: further comments on the inversion of large structural matrices .

Document 119 with rank 5 has cosine similarity score = 1.0000000000000002
Document 119 has the headline: recent developments in rocket nozzle configurations .

Query Q7:
is it possible to relate the available pressure distributions for an
ogive forebody at zero angle of attack to the lower surface pressures of
an equivalent ogive forebody at angle of attack  has vector = [0.7965784284662087, 0.7965784284662087, 0.2757839740126393, 0.9422064766172813, 0.6196039311524101, 0.9422064766172813, 0.3133628759364394, 0.12831501377716079, 0.15857495228082094, 0.7965784284662087, 0.9422064766172813, 0.6564287007412802, 0.6564287007412802, 0.9422064766172813, 0.269231200920391, 0.2517968789980713, 1.0509775004326938, 0.3356652673139371, 0.33099422392183686, 0.16160621295541924, 0.4025331174615726]

Top 5 documents is:

Document 1 with rank 1 has cosine similarity score = 0.995358760234525
Document 1 has the headline: inviscid-incompressible-flow theory of static two-dimensionalsolid jets, in proximity to the ground .

Document 37 with rank 2 has cosine similarity score = 0.995358760234525
Document 37 has the headline: note on creep buckling of columns .

Document 38 with rank 3 has cosine similarity score = 0.995358760234525
Document 38 has the headline: formulae for use with the fatigue load meter in theassessment of wing fatigue life .

Document 46 with rank 4 has cosine similarity score = 0.995358760234525
Document 46 has the headline: convergence rates of iterative treatments of partial differential equations.

Document 72 with rank 5 has cosine similarity score = 0.995358760234525
Document 72 has the headline: on the analogues relating flexure and extension offlat plates .

Query Q8:
what methods -dash exact or approximate -dash are presently available
for predicting body pressures at angle of attack has vector = [0.9004262602878819, 0.13997314527395896, 0.772974208179334, 0.3231484871062042, 0.9004262602878819, 0.259177920187506, 0.9004262602878819, 0.7003800638456551, 0.35421516878858467, 0.9004262602878819, 0.48188590040174245, 0.2022858190232139, 0.12262514731268172, 0.9004262602878819, 0.24063146214233447, 0.9004262602878819, 0.32078087856191784]

Top 5 documents is:

Document 258 with rank 1 has cosine similarity score = 0.9992092212705771
Document 258 has the headline: methods of boundary-layer control for postponing and alleviatingbuffeting and other effects of shock-induced separation .

Document 1149 with rank 2 has cosine similarity score = 0.9991724234794357
Document 1149 has the headline: qualitiative solutions of the stability equation for a boundary layer in contact with various forms of flexible surface .

Document 706 with rank 3 has cosine similarity score = 0.9990066685405081
Document 706 has the headline: an investigation of fluid flow in two dimensions .

Document 870 with rank 4 has cosine similarity score = 0.9988024129921039
Document 870 has the headline: viscosity effects in sound waves of finite amplitude:in survey in mechanics .

Document 891 with rank 5 has cosine similarity score = 0.9988024129921039
Document 891 has the headline: stability of rectangular plates under shear and bendingforces .

Query Q9:
papers on internal /slip flow/ heat transfer studies  has vector = [0.31483408468759716, 1.1931568569324176, 0.5855818179661578, 1.1931568569324176, 1.0789923984629501, 0.2891041237551904, 0.34062171324360335, 0.33069131172116484]

Top 5 documents is:

Document 9 with rank 1 has cosine similarity score = 1.0000000000000004
Document 9 has the headline: stability of cylindrical and conical shells of circularcross section, with simultaneous action of axial compressionand external normal pressure .

Document 11 with rank 2 has cosine similarity score = 1.0000000000000004
Document 11 has the headline: the behaviour of non-linear systems .

Document 13 with rank 3 has cosine similarity score = 1.0000000000000004
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 21 with rank 4 has cosine similarity score = 1.0000000000000004
Document 21 has the headline: transonic flow in two dimensional and axially symmetricalnozzles .

Document 25 with rank 5 has cosine similarity score = 1.0000000000000004
Document 25 has the headline: note on creep buckling of columns .

Query Q10:
are real-gas transport properties for air available over a wide range of
enthalpies and densities  has vector = [1.1931568569324176, 0.8139107349050925, 0.6767268558660842, 0.42047075867272227, 1.1931568569324176, 0.34062171324360335, 0.4693713145754434, 1.1931568569324176, 1.1931568569324176, 0.5555526374653047, 0.26567094682717973, 1.1931568569324176, 0.6275637180056669, 1.1931568569324176, 0.4693713145754434]

Top 5 documents is:

Document 1 with rank 1 has cosine similarity score = 1.0000000000000004
Document 1 has the headline: inviscid-incompressible-flow theory of static two-dimensionalsolid jets, in proximity to the ground .

Document 6 with rank 2 has cosine similarity score = 1.0000000000000004
Document 6 has the headline: hypersonic strong viscous interaction on a flat platewith surface mass transfer .

Document 22 with rank 3 has cosine similarity score = 1.0000000000000004
Document 22 has the headline: concerning the effect of compressibility on laminarboundary layers and their separation .

Document 24 with rank 4 has cosine similarity score = 1.0000000000000004
Document 24 has the headline: influence of the leading-edge shock wave on the laminar boundary layerat hypersonic speeds .

Document 35 with rank 5 has cosine similarity score = 1.0000000000000004
Document 35 has the headline: similar solutions for the compressible boundary layeron a yawed cylinder with transpiration cooling .

Query Q11:
is it possible to find an analytical,  similar solution of the strong
blast wave problem in the newtonian approximation  has vector = [0.9004262602878819, 0.9004262602878819, 0.31173720439012015, 0.9004262602878819, 0.5411655071988212, 0.9004262602878819, 0.37780108834462606, 0.3196137743026895, 0.15325165681475583, 0.9004262602878819, 1.0841883973539788, 0.4516047254471216, 0.6023782119076675, 0.24501513980738934, 0.1822889760242911, 0.9004262602878819, 0.4329252952679785, 0.2809854346835769]

Top 5 documents is:

Document 139 with rank 1 has cosine similarity score = 0.9980309928653238
Document 139 has the headline: the effects of a small jet of air exhausting from the nose of a bodyof revolution in supersonic flow .

Document 317 with rank 2 has cosine similarity score = 0.9980309928653238
Document 317 has the headline: effect of distributed three-dimensional roughness andsurface cooling on boundary layer transition and lateralspread of turbulence at supersonic speeds .

Document 419 with rank 3 has cosine similarity score = 0.9980309928653238
Document 419 has the headline: pressure loads produced on a flat-plate wing by rocket jets exhaustingin a spanwise direction below the wing and perpendicular to a freestreamflow of mach number 2.0 .

Document 462 with rank 4 has cosine similarity score = 0.9980309928653238
Document 462 has the headline: effect of mach number on boundary layer transitionin the presence of pressure rise and surface roughnesson an ogive-cylinder body with cold wall conditions .

Document 789 with rank 5 has cosine similarity score = 0.9980309928653238
Document 789 has the headline: wing-flow study of pressure drag reduction at transonic speed by projectinga jet of air from the nose of a prolate spheroid of finenessratio 6 .

Query Q12:
how can the aerodynamic performance of channel flow ground effect
machines be calculated  has vector = [1.1931568569324176, 1.1931568569324176, 1.1931568569324176, 0.35932468303705756, 0.562562397396617, 1.1931568569324176, 0.7265138751470022, 0.1113648811851109, 0.632964349315849, 0.16567673070721148, 0.6997462764356253, 1.1931568569324176, 0.3562267014577842]

Top 5 documents is:

Document 576 with rank 1 has cosine similarity score = 1.0000000000000004
Document 576 has the headline: response of plates to a decaying and convecting randonpressure field .

Document 12 with rank 2 has cosine similarity score = 1.0000000000000002
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 14 with rank 3 has cosine similarity score = 1.0000000000000002
Document 14 has the headline: generalized heat transfer formulas and graphs .

Document 16 with rank 4 has cosine similarity score = 1.0000000000000002
Document 16 has the headline: principles of creep buckling weight-strength analysis

Document 20 with rank 5 has cosine similarity score = 1.0000000000000002
Document 20 has the headline: note on creep buckling of columns .

Query Q13:
what is the basic mechanism of the transonic aileron buzz  has vector = [0.9004262602878819, 0.9004262602878819, 1.0841883973539788, 0.4482899772649444, 0.4776720836314187, 0.9004262602878819, 0.3854524691530413, 0.6142248565062429, 0.8142710529484697]

Top 5 documents is:

Document 12 with rank 1 has cosine similarity score = 0.9973042957885019
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 15 with rank 2 has cosine similarity score = 0.9973042957885019
Document 15 has the headline: some low speed problems of high speed aircraft .

Document 55 with rank 3 has cosine similarity score = 0.9973042957885019
Document 55 has the headline: transition in the viscous wakes of blunt bodies athypersonic speeds .

Document 65 with rank 4 has cosine similarity score = 0.9973042957885019
Document 65 has the headline: an approximate solution of the supersonic blunt body                   problem for prescribed arbitrary axisymmetric shapes .                 

Document 77 with rank 5 has cosine similarity score = 0.9973042957885019
Document 77 has the headline: investigation of a retrocket exhausting from the nose of a blunt bodyinto a supersonic free stream .

Query Q14:
papers on shock-sound wave interaction  has vector = [0.31483408468759716, 1.1931568569324176, 1.1931568569324176, 0.32467011126483014, 0.46535393253119484]

Top 5 documents is:

Document 2 with rank 1 has cosine similarity score = 1.0000000000000002
Document 2 has the headline: free-flight measurements of the static and dynamic

Document 8 with rank 2 has cosine similarity score = 1.0000000000000002
Document 8 has the headline: new test techniques for a hypervelocity wind tunnel .

Document 12 with rank 3 has cosine similarity score = 1.0000000000000002
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 13 with rank 4 has cosine similarity score = 1.0000000000000002
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 14 with rank 5 has cosine similarity score = 1.0000000000000002
Document 14 has the headline: generalized heat transfer formulas and graphs .

Query Q15:
material properties of photoelastic materials  has vector = [0.45715465905126806, 0.31731193647536693, 0.9004262602878819, 0.8142710529484697]

Top 5 documents is:

Document 1052 with rank 1 has cosine similarity score = 0.9999877430439748
Document 1052 has the headline: theories of plastic buckling .

Document 1139 with rank 2 has cosine similarity score = 0.9999819196626009
Document 1139 has the headline: some experimental studies of panel flutter at mach1 .3.

Document 261 with rank 3 has cosine similarity score = 0.9999088055755668
Document 261 has the headline: the solution of small displacement, stability or vibrationproblems concerning a flat rectangular panel when theedges are either clamped or simply supported .

Document 1038 with rank 4 has cosine similarity score = 0.9999088055755668
Document 1038 has the headline: on some fourier transforms in the theory of non-stationaryflows .

Document 1152 with rank 5 has cosine similarity score = 0.9999088055755668
Document 1152 has the headline: review of panel flutter and effects of aerodynamic noisepart i..  panel flutter .

Query Q16:
can the transverse potential flow about a body of revolution be
calculated efficiently by an electronic computer  has vector = [1.1931568569324176, 1.1931568569324176, 0.5243836288992468, 0.5301633271157155, 0.1113648811851109, 1.1931568569324176, 1.1931568569324176, 0.2680493924628222, 1.1931568569324176, 0.4777125651702375, 1.1931568569324176, 0.3562267014577842, 1.1931568569324176, 1.1931568569324176, 1.1931568569324176, 0.8139107349050925, 0.632964349315849]

Top 5 documents is:

Document 13 with rank 1 has cosine similarity score = 1.0000000000000002
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 538 with rank 2 has cosine similarity score = 1.0000000000000002
Document 538 has the headline: some possibilities of using gas mixtures other than inaerodynamic research .

Document 539 with rank 3 has cosine similarity score = 1.0000000000000002
Document 539 has the headline: atmosphere entries with spacecraft lift-drag ratiosmodulated to limit decelerations .

Document 576 with rank 4 has cosine similarity score = 1.0000000000000002
Document 576 has the headline: response of plates to a decaying and convecting randonpressure field .

Document 922 with rank 5 has cosine similarity score = 1.0000000000000002
Document 922 has the headline: theoretical investigation of the ablation of a glass-typeheat protection shield of varied material propertiesat the stagnation point of a re-entering irbm .

Query Q17:
can the three-dimensional problem of a transverse potential flow about
a body of revolution be reduced to a two-dimensional problem  has vector = [0.7965784284662087, 0.7965784284662087, 0.3620979396832128, 0.19074727315383572, 0.9422064766172813, 1.0509775004326938, 0.3500903377414271, 0.3539489946275973, 0.07434970643242629, 0.7965784284662087, 0.17895582006570257, 0.3189316829643157, 0.7965784284662087, 0.31472886167290715, 0.7965784284662087, 0.24129405155077047]

Top 5 documents is:

Document 1085 with rank 1 has cosine similarity score = 0.9943357185267147
Document 1085 has the headline: some experiments relating to the problem of simulation of hot jetengines in studies of jet effects on adjacent surfaces at a free-streammach number of 1.80 .

Document 653 with rank 2 has cosine similarity score = 0.9943295780702834
Document 653 has the headline: near noise field of a jet engine exhaust .

Document 937 with rank 3 has cosine similarity score = 0.9943295780702834
Document 937 has the headline: investigation of full scale split trailing edge wingflaps with various chords and hinge locations .

Document 7 with rank 4 has cosine similarity score = 0.9943295780702833
Document 7 has the headline: the generation of sound by aerodynamic means .

Document 52 with rank 5 has cosine similarity score = 0.9943295780702833
Document 52 has the headline: a graphical approximation for temperatures and sublimation rates atsurfaces subjected to small net and large gross heat transfer rates .

Query Q18:
are experimental pressure distributions on bodies of revolution at angle
of attack available  has vector = [0.9004262602878819, 0.18461958850009128, 0.12262514731268172, 0.1792479511810003, 0.9004262602878819, 0.2022858190232139, 1.0841883973539788, 0.3605099665224756, 0.9004262602878819, 0.24063146214233447, 0.32078087856191784, 0.35421516878858467]

Top 5 documents is:

Document 891 with rank 1 has cosine similarity score = 0.9967957278559786
Document 891 has the headline: stability of rectangular plates under shear and bendingforces .

Document 1334 with rank 2 has cosine similarity score = 0.9967957278559786
Document 1334 has the headline: an investigation of the noise produced by a subsonic air jet .

Document 90 with rank 3 has cosine similarity score = 0.9967957278559785
Document 90 has the headline: further comments on the inversion of large structural matrices .

Document 109 with rank 4 has cosine similarity score = 0.9967957278559785
Document 109 has the headline: note on creep buckling of columns .

Document 137 with rank 5 has cosine similarity score = 0.9967957278559785
Document 137 has the headline: on the theory of thin elastic shells .

Query Q19:
does there exist a good basic treatment of the dynamics of re-entry
combining consideration of realistic effects with relative simplicity of
results  has vector = [0.34745312945885515, 0.7415940651559643, 0.39341231807087373, 0.7415940651559643, 0.23793392369631267, 0.36921311746538266, 0.3589675393514601, 1.035531726598951, 0.7415940651559643, 0.3009854395585222, 0.4401708330086434, 0.5996786340694428, 0.28094699979796667, 0.6291287468830573, 0.10297462526661758, 0.7415940651559643, 0.3665539979782177, 0.48721331579653576, 0.07949351956089666]

Top 5 documents is:

Document 12 with rank 1 has cosine similarity score = 0.9928502234549681
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 90 with rank 2 has cosine similarity score = 0.9928502234549681
Document 90 has the headline: further comments on the inversion of large structural matrices .

Document 91 with rank 3 has cosine similarity score = 0.9928502234549681
Document 91 has the headline: local heat transfer and recovery temperature on a yawedcylinder at a mach number of 4. 15 and high reynoldsnumbers .

Document 101 with rank 4 has cosine similarity score = 0.9928502234549681
Document 101 has the headline: the law of the wake in the turbulent boundary layer .

Document 137 with rank 5 has cosine similarity score = 0.9928502234549681
Document 137 has the headline: on the theory of thin elastic shells .

Query Q20:
has anyone formally determined the influence of joule heating,  produced
by the induced current,  in magnetohydrodynamic free convection flows
under general conditions 
 has vector = [0.17266605352051925, 0.9004262602878819, 0.7281158456090577, 0.28620140378163894, 1.0841883973539788, 0.3330974507159971, 0.9004262602878819, 0.8142710529484697, 0.3654495181077095, 0.4658254390328432, 0.9004262602878819, 0.41417866006401616, 0.5220052420278657, 0.9004262602878819, 0.5162230045682554, 0.3231484871062042, 0.5106978420511133, 0.08404248184997314, 0.9004262602878819, 0.23230231964517273, 0.17812647869746076]

Top 5 documents is:

Document 301 with rank 1 has cosine similarity score = 0.9981451805594265
Document 301 has the headline: effect of rheological behaviour on thermal stresses .

Document 710 with rank 2 has cosine similarity score = 0.9981451805594265
Document 710 has the headline: static aerodynamic characteristics of short blunt coneswith various nose and base cone angles at mach numbersof 0. 6 to 5. 5 and angles of attack to 180 .

Document 902 with rank 3 has cosine similarity score = 0.9981451805594265
Document 902 has the headline: the rolling up of the trailing vortex sheet and itseffect on the downwash behind wings .

Document 959 with rank 4 has cosine similarity score = 0.9981451805594265
Document 959 has the headline: some effects of bluntness on boundary layer transitionand heat transfer at supersonic speeds .

Document 1002 with rank 5 has cosine similarity score = 0.9981451805594265
Document 1002 has the headline: thermal effects on a transpiration cooled hemisphere .
***********************


For weighting 2 function

Query Q1:
what similarity laws must be obeyed when constructing aeroelastic models
of heated high speed aircraft  has vector = [0.8552016302631957, 0.3933762051694568, 0.3933762051694568, 0.3933762051694568, 0.8552016302631957, 0.8552016302631957, 0.8552016302631957, 0.643679336709757, 0.5278896580478494, 0.24481131158777383, 0.8552016302631957, 0.4752041982232935, 0.24085386259538516, 0.2043257098280058, 0.34240306474133664]

Top 5 documents is:

Document 434 with rank 1 has cosine similarity score = 1.0000000000000007
Document 434 has the headline: on the propagation and structure of the blast wave .

Document 872 with rank 2 has cosine similarity score = 1.0000000000000007
Document 872 has the headline: breathing vibrations of a circular shell with an internalliquid .

Document 888 with rank 3 has cosine similarity score = 1.0000000000000007
Document 888 has the headline: investigation of the flow through a single stage twodimensional nozzle in the langley 11in . hypersonictunnel .

Document 1236 with rank 4 has cosine similarity score = 1.0000000000000007
Document 1236 has the headline: effects of free stream vorticity on the behaviour ofa viscous boundary layer .

Document 13 with rank 5 has cosine similarity score = 1.0000000000000004
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Query Q2:
what are the structural and aeroelastic problems associated with flight
of high speed aircraft  has vector = [0.8567748624751591, 0.8567748624751591, 0.8567748624751591, 0.4297115639590336, 0.8567748624751591, 0.5288607658954194, 0.17345186302542523, 0.3594858233098877, 0.8567748624751591, 0.2857301473065797, 0.8567748624751591, 0.24129693828841703, 0.20470158819055612, 0.3430329507376487]

Top 5 documents is:

Document 48 with rank 1 has cosine similarity score = 1.0000000000000004
Document 48 has the headline: control system and analysis and design via the second method oflyapunov .

Document 67 with rank 2 has cosine similarity score = 1.0000000000000004
Document 67 has the headline: the transition to tubulence in a boundary layer ona blunt cone in supersonic flow .

Document 97 with rank 3 has cosine similarity score = 1.0000000000000004
Document 97 has the headline: note on creep buckling of columns .

Document 121 with rank 4 has cosine similarity score = 1.0000000000000004
Document 121 has the headline: free-flight measurements of the static and dynamic

Document 152 with rank 5 has cosine similarity score = 1.0000000000000004
Document 152 has the headline: transition in a separated laminar boundary layer .

Query Q3:
what problems of heat conduction in composite slabs have been solved so
far  has vector = [0.8583590069651166, 0.17377256899512514, 0.8583590069651166, 0.2079819825316407, 0.42734619889309666, 0.8583590069651166, 0.5855287224913085, 0.537485694189796, 0.8583590069651166, 0.8583590069651166, 0.3948285373696537, 0.8583590069651166, 0.4242683976138692]

Top 5 documents is:

Document 64 with rank 1 has cosine similarity score = 1.0000000000000004
Document 64 has the headline: a study of laminar compressible viscous pipe flow accelerated by anaxial body force, with application to magnetogasdynamics .

Document 70 with rank 2 has cosine similarity score = 1.0000000000000004
Document 70 has the headline: air scooping vehicle .

Document 114 with rank 3 has cosine similarity score = 1.0000000000000004
Document 114 has the headline: note on creep buckling of columns .

Document 352 with rank 4 has cosine similarity score = 1.0000000000000004
Document 352 has the headline: an approximate solution of the turbulent boundary layerequations in incompressible and compressible .

Document 369 with rank 5 has cosine similarity score = 1.0000000000000004
Document 369 has the headline: some results on buckling and postbuckling of cylindricalshells .

Query Q4:
can a criterion be developed to show empirically the validity of flow
solutions for chemically reacting gas mixtures based on the simplifying
assumption of instantaneous local chemical equilibrium  has vector = [0.8356935249031621, 0.8356935249031621, 0.4288928480127151, 0.8356935249031621, 0.23922689185471915, 0.8356935249031621, 0.2553825960594468, 0.7089576964050264, 0.8794592219664876, 0.4598397607206618, 0.8794592219664876, 0.07800056595013025, 0.1422342094285726, 0.8356935249031621, 0.5822218679068905, 0.6112134808810449, 0.2584386667647881, 0.490106081941138, 0.23285179352462534, 0.8356935249031621, 0.5312520758380417, 0.2944999967039682, 0.5822218679068905, 0.29240248578333283, 0.4191383142814687, 0.33163435919588613]

Top 5 documents is:

Document 206 with rank 1 has cosine similarity score = 0.9998197244977751
Document 206 has the headline: design of thin walled torispherical and toriconicalpressure - vessel heads .

Document 212 with rank 2 has cosine similarity score = 0.9998197244977751
Document 212 has the headline: low speed tests on 45 sweptback wings .

Document 324 with rank 3 has cosine similarity score = 0.9998197244977751
Document 324 has the headline: scale height in the upper atmosphere, derived from changes in satelliteorbits .

Document 921 with rank 4 has cosine similarity score = 0.9998197244977751
Document 921 has the headline: preliminary results of density measurements from an air force satellite.

Document 1004 with rank 5 has cosine similarity score = 0.9998197244977751
Document 1004 has the headline: boundary layer displacement effects in air at machnumbers of 6. 8 and 9. 6.

Query Q5:
what chemical kinetic system is applicable to hypersonic aerodynamic
problems  has vector = [0.863178061388394, 0.4329230594636963, 0.49486825515823496, 0.3324251473407315, 0.863178061388394, 0.351953026710897, 0.863178061388394, 0.25122757680931207, 0.25995004890668333, 0.17474817414456148]

Top 5 documents is:

Document 47 with rank 1 has cosine similarity score = 1.0000000000000004
Document 47 has the headline: on the use of side-jets as control devices .

Document 269 with rank 2 has cosine similarity score = 1.0000000000000004
Document 269 has the headline: the effect of end plates on swept wings .

Document 509 with rank 3 has cosine similarity score = 1.0000000000000004
Document 509 has the headline: practical calculation of second-order supersonic flowpast non-lifting bodies of revolution .

Document 1375 with rank 4 has cosine similarity score = 1.0000000000000004
Document 1375 has the headline: viscous compressible and incompressible flow in slenderchannels .

Document 31 with rank 5 has cosine similarity score = 1.0000000000000002
Document 31 has the headline: the effect of slip particularly for highly cooled walls .

Query Q6:
what theoretical and experimental guides do we have as to turbulent
couette flow behaviour  has vector = [0.8567748624751591, 0.2199505762441286, 0.8567748624751591, 0.17566949068857024, 0.544653525068923, 0.8567748624751591, 0.8567748624751591, 0.8567748624751591, 0.8567748624751591, 0.8567748624751591, 0.2876377694726451, 0.57317576399895, 0.07996822061371288, 0.46267500092398783]

Top 5 documents is:

Document 12 with rank 1 has cosine similarity score = 1.0000000000000004
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 37 with rank 2 has cosine similarity score = 1.0000000000000004
Document 37 has the headline: note on creep buckling of columns .

Document 234 with rank 3 has cosine similarity score = 1.0000000000000004
Document 234 has the headline: compressive buckling of simply supplorted plates withlongitudinal stiffeners .

Document 343 with rank 4 has cosine similarity score = 1.0000000000000004
Document 343 has the headline: flutter model testing at transonic speeds .

Document 375 with rank 5 has cosine similarity score = 1.0000000000000004
Document 375 has the headline: structural loads surveys on two tilt-wing vtol configurations .

Query Q7:
is it possible to relate the available pressure distributions for an
ogive forebody at zero angle of attack to the lower surface pressures of
an equivalent ogive forebody at angle of attack  has vector = [0.8300230470974215, 0.8300230470974215, 0.28736285878511875, 0.8748653622742897, 0.6456182147926199, 0.8748653622742897, 0.32651952379979393, 0.11914411946775963, 0.16523277606059178, 0.8300230470974215, 0.8748653622742897, 0.6095126146267548, 0.6095126146267548, 0.8748653622742897, 0.2805349652662866, 0.23380052380351338, 0.9009865147882452, 0.311674694431959, 0.3448911299798066, 0.16839130525466145, 0.4194336085103414]

Top 5 documents is:

Document 102 with rank 1 has cosine similarity score = 0.9995657702711793
Document 102 has the headline: a note on the use of end plates to prevent three dimensionalflow at the ends of bluff cylinders .

Document 179 with rank 2 has cosine similarity score = 0.9995657702711793
Document 179 has the headline: on blunt-body heat transfer at hypersonic speed and low reynoldsnumber .

Document 382 with rank 3 has cosine similarity score = 0.9995657702711793
Document 382 has the headline: simplified formulas for boundary value problems ofthe thin-- walled circular cylinder .

Document 439 with rank 4 has cosine similarity score = 0.9995657702711793
Document 439 has the headline: a survey of buckling theory and experiment for circularconical shells of constant thickness .

Document 722 with rank 5 has cosine similarity score = 0.9995657702711793
Document 722 has the headline: pitch-yaw stability of a missile oscillating in rollvia the second method of lyapunov .

Query Q8:
what methods -dash exact or approximate -dash are presently available
for predicting body pressures at angle of attack has vector = [0.8505462912985426, 0.1322191997777746, 0.635484260214769, 0.3052473693504249, 0.8505462912985426, 0.24482051282186457, 0.8505462912985426, 0.6615818441511258, 0.33459308265675713, 0.8505462912985426, 0.4551913726780021, 0.19108000370566083, 0.11583221066141818, 0.8505462912985426, 0.22730145345769046, 0.8505462912985426, 0.3030109167330327]

Top 5 documents is:

Document 7 with rank 1 has cosine similarity score = 0.9999324297157606
Document 7 has the headline: the generation of sound by aerodynamic means .

Document 225 with rank 2 has cosine similarity score = 0.9999324297157606
Document 225 has the headline: analysis of partly wrinkled membrane .

Document 280 with rank 3 has cosine similarity score = 0.9999324297157606
Document 280 has the headline: a simple model study of transient temperature and thermalstress distribution due to aerodynamic heating .

Document 318 with rank 4 has cosine similarity score = 0.9999324297157606
Document 318 has the headline: on periodically oscillating wakes in the oseen approximation .

Document 798 with rank 5 has cosine similarity score = 0.9999324297157606
Document 798 has the headline: on squire's test of the compressibility transformation .

Query Q9:
papers on internal /slip flow/ heat transfer studies  has vector = [0.22862642963011864, 0.8664474574277228, 0.42523820262587625, 0.8664474574277228, 0.7835434333719102, 0.2099418291100202, 0.2473529072643227, 0.2401416415364666]

Top 5 documents is:

Document 201 with rank 1 has cosine similarity score = 1.0000000000000004
Document 201 has the headline: theory for supersonic two-dimensional, laminar, base-typeflows using the crocco-lees mixing concepts .

Document 220 with rank 2 has cosine similarity score = 1.0000000000000004
Document 220 has the headline: growth of the turbulent wake behind a supersonic sphere .

Document 413 with rank 3 has cosine similarity score = 1.0000000000000004
Document 413 has the headline: on the propagation and structure of the blast wave . part 1.

Document 1061 with rank 4 has cosine similarity score = 1.0000000000000004
Document 1061 has the headline: the theoretical wave drag of some bodies of revolution .

Document 1378 with rank 5 has cosine similarity score = 1.0000000000000004
Document 1378 has the headline: on the response of the laminar boundary layer to smallfluctuations of the free-stream velocity .

Query Q10:
are real-gas transport properties for air available over a wide range of
enthalpies and densities  has vector = [0.8552016302631957, 0.5833749211894079, 0.48504763394436295, 0.30137469034826364, 0.8552016302631957, 0.24414245518221256, 0.3364244282647867, 0.8552016302631957, 0.8552016302631957, 0.3981953575482471, 0.19042108799031107, 0.8552016302631957, 0.4498096889895156, 0.8552016302631957, 0.3364244282647867]

Top 5 documents is:

Document 35 with rank 1 has cosine similarity score = 1.0000000000000004
Document 35 has the headline: similar solutions for the compressible boundary layeron a yawed cylinder with transpiration cooling .

Document 64 with rank 2 has cosine similarity score = 1.0000000000000004
Document 64 has the headline: a study of laminar compressible viscous pipe flow accelerated by anaxial body force, with application to magnetogasdynamics .

Document 138 with rank 3 has cosine similarity score = 1.0000000000000004
Document 138 has the headline: pressure distributions . axially symmetric bodies inoblique flow .

Document 149 with rank 4 has cosine similarity score = 1.0000000000000004
Document 149 has the headline: optimum nose shapes for missiles in the super-aerodynamicregion .

Document 190 with rank 5 has cosine similarity score = 1.0000000000000004
Document 190 has the headline: the flow field over blunted flat plates and its effecton turbulent boundary growth and heat transfer at amach number of 4. 7.

Query Q11:
is it possible to find an analytical,  similar solution of the strong
blast wave problem in the newtonian approximation  has vector = [0.8490155997848305, 0.8490155997848305, 0.2939382837145428, 0.8490155997848305, 0.510267167830459, 0.8490155997848305, 0.35623018981890003, 0.30136513366717427, 0.14450161336587355, 0.8490155997848305, 0.8901276129335317, 0.425819940789561, 0.5679848772031477, 0.2310257764066395, 0.17188102028915136, 0.8490155997848305, 0.40820702975327733, 0.2649423144127386]

Top 5 documents is:

Document 439 with rank 1 has cosine similarity score = 0.9998849634096577
Document 439 has the headline: a survey of buckling theory and experiment for circularconical shells of constant thickness .

Document 444 with rank 2 has cosine similarity score = 0.9998849634096577
Document 444 has the headline: contributions of the wing panels to the forces andmoments of supersonic wing-body combinations at combinedangles .

Document 715 with rank 3 has cosine similarity score = 0.9998849634096577
Document 715 has the headline: numerical technique to lifting surface theory for calculationof unsteady aerodynamic forces due to continuous sinusoidalgusts on several wing planforms at sobsonic speeds .

Document 1021 with rank 4 has cosine similarity score = 0.9998849634096577
Document 1021 has the headline: intensity, scale and spectra of turbulence in mixingregion of free subsonic jet .

Document 1177 with rank 5 has cosine similarity score = 0.9998849634096577
Document 1177 has the headline: normal-shock relations in magnetohydrodynamics .

Query Q12:
how can the aerodynamic performance of channel flow ground effect
machines be calculated  has vector = [0.8583590069651166, 0.8583590069651166, 0.8583590069651166, 0.25849876847098735, 0.4047083147363805, 0.8583590069651166, 0.5226552777149918, 0.0801160788453269, 0.4553555948376429, 0.11918811279573789, 0.5033986231391554, 0.8583590069651166, 0.2562700754231864]

Top 5 documents is:

Document 16 with rank 1 has cosine similarity score = 1.0000000000000002
Document 16 has the headline: principles of creep buckling weight-strength analysis

Document 27 with rank 2 has cosine similarity score = 1.0000000000000002
Document 27 has the headline: the properties of crossed flexure pivots, and the influence of thepoint at which the strips cross .

Document 51 with rank 3 has cosine similarity score = 1.0000000000000002
Document 51 has the headline: a unified theory of creep buckling under normal loads .

Document 70 with rank 4 has cosine similarity score = 1.0000000000000002
Document 70 has the headline: air scooping vehicle .

Document 93 with rank 5 has cosine similarity score = 1.0000000000000002
Document 93 has the headline: note on creep buckling of columns .

Query Q13:
what is the basic mechanism of the transonic aileron buzz  has vector = [0.863178061388394, 0.863178061388394, 0.9012815704189442, 0.4297454334480272, 0.4579120815473532, 0.863178061388394, 0.36950734308272787, 0.5888160355586628, 0.7805868619425089]

Top 5 documents is:

Document 146 with rank 1 has cosine similarity score = 0.9998665988927309
Document 146 has the headline: heat transfer in separated flows .

Document 388 with rank 2 has cosine similarity score = 0.9998665988927309
Document 388 has the headline: application of similar solutions to calculations oflaminar heat transfer on bodies with yaw and largepressure gradients in high speed flow .

Document 529 with rank 3 has cosine similarity score = 0.9998665988927309
Document 529 has the headline: some experimental techniques in mass transfer cooling .

Document 603 with rank 4 has cosine similarity score = 0.9998665988927309
Document 603 has the headline: slender-body theory-review and extension .

Document 688 with rank 5 has cosine similarity score = 0.9998665988927309
Document 688 has the headline: note on creep buckling of columns .

Query Q14:
papers on shock-sound wave interaction  has vector = [0.22994353504868564, 0.871439017865105, 0.871439017865105, 0.2371274164389298, 0.33987783883441114]

Top 5 documents is:

Document 14 with rank 1 has cosine similarity score = 1.0000000000000004
Document 14 has the headline: generalized heat transfer formulas and graphs .

Document 64 with rank 2 has cosine similarity score = 1.0000000000000004
Document 64 has the headline: a study of laminar compressible viscous pipe flow accelerated by anaxial body force, with application to magnetogasdynamics .

Document 187 with rank 3 has cosine similarity score = 1.0000000000000004
Document 187 has the headline: methods for calculating the lift distribution of wings /subsonic liftingsurface theory/ .

Document 281 with rank 4 has cosine similarity score = 1.0000000000000004
Document 281 has the headline: the lift of twisted and cambered wings in supersonic flow .

Document 373 with rank 5 has cosine similarity score = 1.0000000000000004
Document 373 has the headline: the calculation of aerodynamic loading on surfaces of any shape .

Query Q15:
material properties of photoelastic materials  has vector = [0.38273741062164035, 0.30709677679831415, 0.871439017865105, 0.7880573878759732]

Top 5 documents is:

Document 330 with rank 1 has cosine similarity score = 0.9999339461414993
Document 330 has the headline: effect of ground proximity on the aerodynamic characteristicsof a four- engined vertical take-off and landing transportairplane model with tilting wing and propellers .

Document 65 with rank 2 has cosine similarity score = 0.9999339461414992
Document 65 has the headline: an approximate solution of the supersonic blunt body                   problem for prescribed arbitrary axisymmetric shapes .                 

Document 96 with rank 3 has cosine similarity score = 0.9999339461414992
Document 96 has the headline: two dimensional transonic unsteady flow with shockwaves .

Document 120 with rank 4 has cosine similarity score = 0.9999339461414992
Document 120 has the headline: first-order approach to a strong interaction problemin hypersonic flow over an insulated flat plate .

Document 171 with rank 5 has cosine similarity score = 0.9999339461414992
Document 171 has the headline: numerical determination of indical lift and moment functions for a twodimensional sinking and pitching airfoil at mach numbers 0.5 and 0.6 .

Query Q16:
can the transverse potential flow about a body of revolution be
calculated efficiently by an electronic computer  has vector = [0.8520874547475841, 0.8520874547475841, 0.37448614493892113, 0.3786136897071266, 0.07953071518296034, 0.8520874547475841, 0.8520874547475841, 0.19142623473454964, 0.8520874547475841, 0.34115622048501815, 0.8520874547475841, 0.2543976524081474, 0.8520874547475841, 0.8520874547475841, 0.8520874547475841, 0.5812505895327542, 0.4520285645771147]

Top 5 documents is:

Document 31 with rank 1 has cosine similarity score = 1.0000000000000004
Document 31 has the headline: the effect of slip particularly for highly cooled walls .

Document 90 with rank 2 has cosine similarity score = 1.0000000000000004
Document 90 has the headline: further comments on the inversion of large structural matrices .

Document 97 with rank 3 has cosine similarity score = 1.0000000000000004
Document 97 has the headline: note on creep buckling of columns .

Document 191 with rank 4 has cosine similarity score = 1.0000000000000004
Document 191 has the headline: use of freon-12 as a fluid for aerodynamic testing .

Document 213 with rank 5 has cosine similarity score = 1.0000000000000004
Document 213 has the headline: the motion of rolling symmetrical missiles referredto a body-axis system .

Query Q17:
can the three-dimensional problem of a transverse potential flow about
a body of revolution be reduced to a two-dimensional problem  has vector = [0.8474952738353885, 0.8474952738353885, 0.38524303644267244, 0.17995933024944125, 0.888918848931509, 0.912646063438806, 0.37246791533456713, 0.37657321539984545, 0.07910209787359666, 0.8474952738353885, 0.190394575488373, 0.3393176168592293, 0.8474952738353885, 0.334846153593394, 0.8474952738353885, 0.2567174316879526]

Top 5 documents is:

Document 543 with rank 1 has cosine similarity score = 0.99961859928488
Document 543 has the headline: a study of the acoustic fatigue characteristics ofsome flat and curved aluminium panels exposed to randomand discrete noise .

Document 1237 with rank 2 has cosine similarity score = 0.99961859928488
Document 1237 has the headline: transtability flutter of supersonic aircraft panels .

Document 352 with rank 3 has cosine similarity score = 0.9996185992848798
Document 352 has the headline: an approximate solution of the turbulent boundary layerequations in incompressible and compressible .

Document 411 with rank 4 has cosine similarity score = 0.9996185992848798
Document 411 has the headline: temperature measurements of shock-waves by spectrum-line reversal, ii adouble beam method .

Document 515 with rank 5 has cosine similarity score = 0.9996185992848798
Document 515 has the headline: boundary layer induced noise in the interior of aircraft .

Query Q18:
are experimental pressure distributions on bodies of revolution at angle
of attack available  has vector = [0.8583590069651166, 0.175994296968395, 0.1168961905249324, 0.17087361859822223, 0.8583590069651166, 0.19283517418114554, 0.8975076018353152, 0.3436672057591079, 0.8583590069651166, 0.22938933702691125, 0.3057942315984574, 0.33766649635045365]

Top 5 documents is:

Document 31 with rank 1 has cosine similarity score = 0.9998282355034254
Document 31 has the headline: the effect of slip particularly for highly cooled walls .

Document 213 with rank 2 has cosine similarity score = 0.9998282355034254
Document 213 has the headline: the motion of rolling symmetrical missiles referredto a body-axis system .

Document 299 with rank 3 has cosine similarity score = 0.9998282355034254
Document 299 has the headline: the flow field in the diffuser of a radial compressor .

Document 1021 with rank 4 has cosine similarity score = 0.9998282355034254
Document 1021 has the headline: intensity, scale and spectra of turbulence in mixingregion of free subsonic jet .

Document 253 with rank 5 has cosine similarity score = 0.9998282355034253
Document 253 has the headline: a description of the r. a. e.  high speed supersonictunnel .

Query Q19:
does there exist a good basic treatment of the dynamics of re-entry
combining consideration of realistic effects with relative simplicity of
results  has vector = [0.39565993601834637, 0.8444853002998796, 0.4479956558145894, 0.8444853002998796, 0.2709456702056419, 0.42043897736942504, 0.40877189356062715, 0.926338706123175, 0.8444853002998796, 0.34274516376828595, 0.5012416031382028, 0.682880048762324, 0.3199265240112716, 0.7164161684965619, 0.1172616683808796, 0.8444853002998796, 0.4174109227716856, 0.5548109169590063, 0.09052271571801192]

Top 5 documents is:

Document 209 with rank 1 has cosine similarity score = 0.9995386325273912
Document 209 has the headline: laminar mixing of a compressible fluid .

Document 461 with rank 2 has cosine similarity score = 0.9995386325273912
Document 461 has the headline: on the ground level disturbance from large aircraftflying at supersonic speeds .

Document 547 with rank 3 has cosine similarity score = 0.9995386325273912
Document 547 has the headline: means and examples of aeronautical research in france at onera .

Document 792 with rank 4 has cosine similarity score = 0.9995386325273912
Document 792 has the headline: oscillatory aerodynamic coefficients for a unifiedsupersonic hypersonic strip theory .

Document 38 with rank 5 has cosine similarity score = 0.9995386325273911
Document 38 has the headline: formulae for use with the fatigue load meter in theassessment of wing fatigue life .

Query Q20:
has anyone formally determined the influence of joule heating,  produced
by the induced current,  in magnetohydrodynamic free convection flows
under general conditions 
 has vector = [0.161938795534741, 0.8444853002998796, 0.682880048762324, 0.2684205126819725, 0.8865191196259924, 0.3124030396526729, 0.8444853002998796, 0.7636826745311017, 0.34274516376828595, 0.43688501004313507, 0.8444853002998796, 0.38844690070471244, 0.489574519329405, 0.8444853002998796, 0.4841515160776751, 0.3030721772686701, 0.47896961642272434, 0.07882115799280506, 0.8444853002998796, 0.21787002758359297, 0.16705997979900286]

Top 5 documents is:

Document 884 with rank 1 has cosine similarity score = 0.9998858210328677
Document 884 has the headline: oscillatory aerodynamic coefficients for a unified supersonichypersonic strip theory .

Document 1260 with rank 2 has cosine similarity score = 0.9998858210328677
Document 1260 has the headline: free-flight measurements of the static and dynamicstability and drag of a 10 blunted cone at mach numbers3 .5 and 8 .5 .

Document 1276 with rank 3 has cosine similarity score = 0.9998858210328677
Document 1276 has the headline: surface pressure distributions with a sonic jet normal to adjacent flatsurfaces at mach 2.92 to 6.4 .

Document 83 with rank 4 has cosine similarity score = 0.9998858210328676
Document 83 has the headline: the bending of uniformly loaded sectorial plates withclamped edges .

Document 99 with rank 5 has cosine similarity score = 0.9998858210328676
Document 99 has the headline: note on creep buckling of columns .
```

## Non-relevant Documents
```
For weighting 1 function

Query Q1:
what similarity laws must be obeyed when constructing aeroelastic models
of heated high speed aircraft

Document 1 with rank 1 has cosine similarity score = 1.0000000000000002
Document 1 has the headline: inviscid-incompressible-flow theory of static two-dimensionalsolid jets, in proximity to the ground .

Document 13 with rank 2 has cosine similarity score = 1.0000000000000002
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 22 with rank 3 has cosine similarity score = 1.0000000000000002
Document 22 has the headline: concerning the effect of compressibility on laminarboundary layers and their separation .

Document 35 with rank 4 has cosine similarity score = 1.0000000000000002
Document 35 has the headline: similar solutions for the compressible boundary layeron a yawed cylinder with transpiration cooling .

Document 38 with rank 5 has cosine similarity score = 1.0000000000000002
Document 38 has the headline: formulae for use with the fatigue load meter in theassessment of wing fatigue life .

Non-relevant documents are:
* Document = 1
* Document = 38
* Document = 22
* Document = 13

Relevant documents are:
* Document = 38

Query Q2:
what are the structural and aeroelastic problems associated with flight

Top 5 documents is:

Document 53 with rank 1 has cosine similarity score = 1.0000000000000004
Document 53 has the headline: measurement of two dimensional derivatives on a wing-aileron-tabsystem .

Document 1052 with rank 2 has cosine similarity score = 1.0000000000000004
Document 1052 has the headline: theories of plastic buckling .

Document 5 with rank 3 has cosine similarity score = 1.0000000000000002
Document 5 has the headline: buckling of sandwich under normal pressure .

Document 13 with rank 4 has cosine similarity score = 1.0000000000000002
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 21 with rank 5 has cosine similarity score = 1.0000000000000002
Document 21 has the headline: transonic flow in two dimensional and axially symmetricalnozzles .

Non-relevant documents are:
* Document = 1052
* Document = 21
* Document = 13
* Document = 5
Relevant-documents are:
* Document = 53

Query Q3:
what problems of heat conduction in composite slabs have been solved so
far

Top 5 documents is:

Document 13 with rank 1 has cosine similarity score = 1.0000000000000004
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 53 with rank 2 has cosine similarity score = 1.0000000000000004
Document 53 has the headline: measurement of two dimensional derivatives on a wing-aileron-tabsystem .

Document 139 with rank 3 has cosine similarity score = 1.0000000000000004
Document 139 has the headline: the effects of a small jet of air exhausting from the nose of a bodyof revolution in supersonic flow .

Document 419 with rank 4 has cosine similarity score = 1.0000000000000004
Document 419 has the headline: pressure loads produced on a flat-plate wing by rocket jets exhaustingin a spanwise direction below the wing and perpendicular to a freestreamflow of mach number 2.0 .

Document 539 with rank 5 has cosine similarity score = 1.0000000000000004
Document 539 has the headline: atmosphere entries with spacecraft lift-drag ratiosmodulated to limit decelerations .

Non-relevant documents are:
* Document = 539
* Document = 53
* Document = 13
Relevant documents are:
* Document = 139
* Document = 419

Query Q4:
can a criterion be developed to show empirically the validity of flow
solutions for chemically reacting gas mixtures based on the simplifying
assumption of instantaneous local chemical equilibrium

Top 5 documents is:

Document 967 with rank 1 has cosine similarity score = 0.9973982309089124
Document 967 has the headline: the use of models for the determination of critical flutter speeds .

Document 576 with rank 2 has cosine similarity score = 0.9973982309089122
Document 576 has the headline: response of plates to a decaying and convecting randonpressure field .

Document 789 with rank 3 has cosine similarity score = 0.9973982309089122
Document 789 has the headline: wing-flow study of pressure drag reduction at transonic speed by projectinga jet of air from the nose of a prolate spheroid of finenessratio 6 .

Document 891 with rank 4 has cosine similarity score = 0.9973982309089122
Document 891 has the headline: stability of rectangular plates under shear and bendingforces .

Document 1334 with rank 5 has cosine similarity score = 0.9973982309089122
Document 1334 has the headline: an investigation of the noise produced by a subsonic air jet .

Non-relevant documents are:
* Document = 967
* Document = 891

Relevant documents are:
* Document = 576
* Document = 789
* Document = 1334

Query Q5:
what chemical kinetic system is applicable to hypersonic aerodynamic
problems

Top 5 documents is:

Document 12 with rank 1 has cosine similarity score = 1.0000000000000004
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 37 with rank 2 has cosine similarity score = 1.0000000000000004
Document 37 has the headline: note on creep buckling of columns .

Document 39 with rank 3 has cosine similarity score = 1.0000000000000004
Document 39 has the headline: some current and proposed investigations into the flowfor slender delta and other wings in unsteady motion .

Document 67 with rank 4 has cosine similarity score = 1.0000000000000004
Document 67 has the headline: the transition to tubulence in a boundary layer ona blunt cone in supersonic flow .

Document 77 with rank 5 has cosine similarity score = 1.0000000000000004
Document 77 has the headline: investigation of a retrocket exhausting from the nose of a blunt bodyinto a supersonic free stream .

Non-relevant documents are:
* Document = 37

Relevant documents are:
* Document = 12
* Document = 39
* Document = 67
* Document = 77

Query Q6:
what theoretical and experimental guides do we have as to turbulent
couette flow behaviour

Top 5 documents is:

Document 884 with rank 1 has cosine similarity score = 1.0000000000000004
Document 884 has the headline: oscillatory aerodynamic coefficients for a unified supersonichypersonic strip theory .

Document 12 with rank 2 has cosine similarity score = 1.0000000000000002
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 69 with rank 3 has cosine similarity score = 1.0000000000000002
Document 69 has the headline: an approximate analytical method for studying entryinto planetary atospheres .

Document 90 with rank 4 has cosine similarity score = 1.0000000000000002
Document 90 has the headline: further comments on the inversion of large structural matrices .

Document 119 with rank 5 has cosine similarity score = 1.0000000000000002
Document 119 has the headline: recent developments in rocket nozzle configurations .

Non-relevant documents are:
* Document = 69
* Document = 90

Relevant documents are:
* Document = 884
* Document = 12
* Document = 119

Query Q7:
is it possible to relate the available pressure distributions for an
ogive forebody at zero angle of attack to the lower surface pressures of
an equivalent ogive forebody at angle of attack

Top 5 documents is:

Document 1 with rank 1 has cosine similarity score = 0.995358760234525
Document 1 has the headline: inviscid-incompressible-flow theory of static two-dimensionalsolid jets, in proximity to the ground .

Document 37 with rank 2 has cosine similarity score = 0.995358760234525
Document 37 has the headline: note on creep buckling of columns .

Document 38 with rank 3 has cosine similarity score = 0.995358760234525
Document 38 has the headline: formulae for use with the fatigue load meter in theassessment of wing fatigue life .

Document 46 with rank 4 has cosine similarity score = 0.995358760234525
Document 46 has the headline: convergence rates of iterative treatments of partial differential equations.

Document 72 with rank 5 has cosine similarity score = 0.995358760234525
Document 72 has the headline: on the analogues relating flexure and extension offlat plates .

Non-relevant documents are:
* Document = 37
* Document = 38
* Document = 46
* Document = 72

Relevant documents are:
* Document = 1

Query Q8:
what methods -dash exact or approximate -dash are presently available
for predicting body pressures at angle of attack

Top 5 documents is:

Document 258 with rank 1 has cosine similarity score = 0.9992092212705771
Document 258 has the headline: methods of boundary-layer control for postponing and alleviatingbuffeting and other effects of shock-induced separation .

Document 1149 with rank 2 has cosine similarity score = 0.9991724234794357
Document 1149 has the headline: qualitiative solutions of the stability equation for a boundary layer in contact with various forms of flexible surface .

Document 706 with rank 3 has cosine similarity score = 0.9990066685405081
Document 706 has the headline: an investigation of fluid flow in two dimensions .

Document 870 with rank 4 has cosine similarity score = 0.9988024129921039
Document 870 has the headline: viscosity effects in sound waves of finite amplitude:in survey in mechanics .

Document 891 with rank 5 has cosine similarity score = 0.9988024129921039
Document 891 has the headline: stability of rectangular plates under shear and bendingforces .

Non-relevant documents are:
* Document = 258
* Document = 1149
* Document = 706
* Document = 891

Relevant documents are:
* Document = 870

Query Q9:
papers on internal /slip flow/ heat transfer studies

Top 5 documents is:

Document 9 with rank 1 has cosine similarity score = 1.0000000000000004
Document 9 has the headline: stability of cylindrical and conical shells of circularcross section, with simultaneous action of axial compressionand external normal pressure .

Document 11 with rank 2 has cosine similarity score = 1.0000000000000004
Document 11 has the headline: the behaviour of non-linear systems .

Document 13 with rank 3 has cosine similarity score = 1.0000000000000004
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 21 with rank 4 has cosine similarity score = 1.0000000000000004
Document 21 has the headline: transonic flow in two dimensional and axially symmetricalnozzles .

Document 25 with rank 5 has cosine similarity score = 1.0000000000000004
Document 25 has the headline: note on creep buckling of columns .

Non-relevant documents are:
* Document = 9
* Document = 11
* Document = 21
* Document = 25

Relevant documents are:
* Document = 13

Query Q10:
are real-gas transport properties for air available over a wide range of
enthalpies and densities

Top 5 documents is:

Document 1 with rank 1 has cosine similarity score = 1.0000000000000004
Document 1 has the headline: inviscid-incompressible-flow theory of static two-dimensionalsolid jets, in proximity to the ground .

Document 6 with rank 2 has cosine similarity score = 1.0000000000000004
Document 6 has the headline: hypersonic strong viscous interaction on a flat platewith surface mass transfer .

Document 22 with rank 3 has cosine similarity score = 1.0000000000000004
Document 22 has the headline: concerning the effect of compressibility on laminarboundary layers and their separation .

Document 24 with rank 4 has cosine similarity score = 1.0000000000000004
Document 24 has the headline: influence of the leading-edge shock wave on the laminar boundary layerat hypersonic speeds .

Document 35 with rank 5 has cosine similarity score = 1.0000000000000004
Document 35 has the headline: similar solutions for the compressible boundary layeron a yawed cylinder with transpiration cooling .

Non-relevant documents are:
* Document = 1
* Document = 35

Relevant documents are:
* Document = 6
* Document = 22
* Document = 24

Query Q11:
is it possible to find an analytical,  similar solution of the strong
blast wave problem in the newtonian approximation

Top 5 documents is:

Document 139 with rank 1 has cosine similarity score = 0.9980309928653238
Document 139 has the headline: the effects of a small jet of air exhausting from the nose of a bodyof revolution in supersonic flow .

Document 317 with rank 2 has cosine similarity score = 0.9980309928653238
Document 317 has the headline: effect of distributed three-dimensional roughness andsurface cooling on boundary layer transition and lateralspread of turbulence at supersonic speeds .

Document 419 with rank 3 has cosine similarity score = 0.9980309928653238
Document 419 has the headline: pressure loads produced on a flat-plate wing by rocket jets exhaustingin a spanwise direction below the wing and perpendicular to a freestreamflow of mach number 2.0 .

Document 462 with rank 4 has cosine similarity score = 0.9980309928653238
Document 462 has the headline: effect of mach number on boundary layer transitionin the presence of pressure rise and surface roughnesson an ogive-cylinder body with cold wall conditions .

Document 789 with rank 5 has cosine similarity score = 0.9980309928653238
Document 789 has the headline: wing-flow study of pressure drag reduction at transonic speed by projectinga jet of air from the nose of a prolate spheroid of finenessratio 6 .

Non-relevant documents are:
* Document = 139
* Document = 317
* Document = 472

Relevant documents are:
* Document = 419
* Document = 789

Query Q12:
how can the aerodynamic performance of channel flow ground effect
machines be calculated

Top 5 documents is:

Document 576 with rank 1 has cosine similarity score = 1.0000000000000004
Document 576 has the headline: response of plates to a decaying and convecting randonpressure field .

Document 12 with rank 2 has cosine similarity score = 1.0000000000000002
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 14 with rank 3 has cosine similarity score = 1.0000000000000002
Document 14 has the headline: generalized heat transfer formulas and graphs .

Document 16 with rank 4 has cosine similarity score = 1.0000000000000002
Document 16 has the headline: principles of creep buckling weight-strength analysis

Document 20 with rank 5 has cosine similarity score = 1.0000000000000002
Document 20 has the headline: note on creep buckling of columns .

Non-relevant documents are:
* Document = 14
* Document = 16
* Document = 20

Relevant documents are:
* Document = 576
* Document = 12

Query Q13:
what is the basic mechanism of the transonic aileron buzz

Top 5 documents is:

Document 12 with rank 1 has cosine similarity score = 0.9973042957885019
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 15 with rank 2 has cosine similarity score = 0.9973042957885019
Document 15 has the headline: some low speed problems of high speed aircraft .

Document 55 with rank 3 has cosine similarity score = 0.9973042957885019
Document 55 has the headline: transition in the viscous wakes of blunt bodies athypersonic speeds .

Document 65 with rank 4 has cosine similarity score = 0.9973042957885019
Document 65 has the headline: an approximate solution of the supersonic blunt body                   problem for prescribed arbitrary axisymmetric shapes .                 

Document 77 with rank 5 has cosine similarity score = 0.9973042957885019
Document 77 has the headline: investigation of a retrocket exhausting from the nose of a blunt bodyinto a supersonic free stream .

Non-relevant documents are None

Relevant documents are:
* Document = 12
* Document = 15
* Document = 55
* Document = 65
* Document = 77

Query Q14:
papers on shock-sound wave interaction

Top 5 documents is:

Document 2 with rank 1 has cosine similarity score = 1.0000000000000002
Document 2 has the headline: free-flight measurements of the static and dynamic

Document 8 with rank 2 has cosine similarity score = 1.0000000000000002
Document 8 has the headline: new test techniques for a hypervelocity wind tunnel .

Document 12 with rank 3 has cosine similarity score = 1.0000000000000002
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 13 with rank 4 has cosine similarity score = 1.0000000000000002
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 14 with rank 5 has cosine similarity score = 1.0000000000000002
Document 14 has the headline: generalized heat transfer formulas and graphs .

Non-relevant documents are:
* Document = 2
* Document = 13
* Document = 14

Relevant documents are:
* Document = 8
* Document = 12

Query Q15:
material properties of photoelastic materials

Top 5 documents is:

Document 1052 with rank 1 has cosine similarity score = 0.9999877430439748
Document 1052 has the headline: theories of plastic buckling .

Document 1139 with rank 2 has cosine similarity score = 0.9999819196626009
Document 1139 has the headline: some experimental studies of panel flutter at mach1 .3.

Document 261 with rank 3 has cosine similarity score = 0.9999088055755668
Document 261 has the headline: the solution of small displacement, stability or vibrationproblems concerning a flat rectangular panel when theedges are either clamped or simply supported .

Document 1038 with rank 4 has cosine similarity score = 0.9999088055755668
Document 1038 has the headline: on some fourier transforms in the theory of non-stationaryflows .

Document 1152 with rank 5 has cosine similarity score = 0.9999088055755668
Document 1152 has the headline: review of panel flutter and effects of aerodynamic noisepart i..  panel flutter .

Non-relevant documents are:
* Document = 1052
* Document = 261

Relevant documents are:
* Document = 1139
* Document = 1038
* Document = 1152

Query Q16:
can the transverse potential flow about a body of revolution be
calculated efficiently by an electronic computer

Top 5 documents is:

Document 13 with rank 1 has cosine similarity score = 1.0000000000000002
Document 13 has the headline: experimental investigation at mach number of 3. 0 ofeffects of thermal stress and buckling on flutter characteristicsof flat single-bay panels of length-width ratio 0. 96 .

Document 538 with rank 2 has cosine similarity score = 1.0000000000000002
Document 538 has the headline: some possibilities of using gas mixtures other than inaerodynamic research .

Document 539 with rank 3 has cosine similarity score = 1.0000000000000002
Document 539 has the headline: atmosphere entries with spacecraft lift-drag ratiosmodulated to limit decelerations .

Document 576 with rank 4 has cosine similarity score = 1.0000000000000002
Document 576 has the headline: response of plates to a decaying and convecting randonpressure field .

Document 922 with rank 5 has cosine similarity score = 1.0000000000000002
Document 922 has the headline: theoretical investigation of the ablation of a glass-typeheat protection shield of varied material propertiesat the stagnation point of a re-entering irbm .

Non-relevant documents are:
* Document = 13
* Document = 538
* Document = 539
* Document = 576
* Document = 922

Relevant documents are None

Query Q17:
can the three-dimensional problem of a transverse potential flow about
a body of revolution be reduced to a two-dimensional problem

Top 5 documents is:

Document 1085 with rank 1 has cosine similarity score = 0.9943357185267147
Document 1085 has the headline: some experiments relating to the problem of simulation of hot jetengines in studies of jet effects on adjacent surfaces at a free-streammach number of 1.80 .

Document 653 with rank 2 has cosine similarity score = 0.9943295780702834
Document 653 has the headline: near noise field of a jet engine exhaust .

Document 937 with rank 3 has cosine similarity score = 0.9943295780702834
Document 937 has the headline: investigation of full scale split trailing edge wingflaps with various chords and hinge locations .

Document 7 with rank 4 has cosine similarity score = 0.9943295780702833
Document 7 has the headline: the generation of sound by aerodynamic means .

Document 52 with rank 5 has cosine similarity score = 0.9943295780702833
Document 52 has the headline: a graphical approximation for temperatures and sublimation rates atsurfaces subjected to small net and large gross heat transfer rates .

Non-relevant documents are:
* Document = 653
* Document = 937
* Document = 52

Relevant documents are:
* Document = 1085
* Document = 7

Query Q18:
are experimental pressure distributions on bodies of revolution at angle
of attack available

Top 5 documents is:

Document 891 with rank 1 has cosine similarity score = 0.9967957278559786
Document 891 has the headline: stability of rectangular plates under shear and bendingforces .

Document 1334 with rank 2 has cosine similarity score = 0.9967957278559786
Document 1334 has the headline: an investigation of the noise produced by a subsonic air jet .

Document 90 with rank 3 has cosine similarity score = 0.9967957278559785
Document 90 has the headline: further comments on the inversion of large structural matrices .

Document 109 with rank 4 has cosine similarity score = 0.9967957278559785
Document 109 has the headline: note on creep buckling of columns .

Document 137 with rank 5 has cosine similarity score = 0.9967957278559785
Document 137 has the headline: on the theory of thin elastic shells .

Non-relevant documents are:
* Document = 891
* Document = 109
* Document = 137

Relevant documents are:
* Document = 1334
* Document = 90

Query Q19:
does there exist a good basic treatment of the dynamics of re-entry
combining consideration of realistic effects with relative simplicity of
results

Top 5 documents is:

Document 12 with rank 1 has cosine similarity score = 0.9928502234549681
Document 12 has the headline: approximations for the thermodynamic and transport properties of hightemperature air .

Document 90 with rank 2 has cosine similarity score = 0.9928502234549681
Document 90 has the headline: further comments on the inversion of large structural matrices .

Document 91 with rank 3 has cosine similarity score = 0.9928502234549681
Document 91 has the headline: local heat transfer and recovery temperature on a yawedcylinder at a mach number of 4. 15 and high reynoldsnumbers .

Document 101 with rank 4 has cosine similarity score = 0.9928502234549681
Document 101 has the headline: the law of the wake in the turbulent boundary layer .

Document 137 with rank 5 has cosine similarity score = 0.9928502234549681
Document 137 has the headline: on the theory of thin elastic shells .

Non-relevant documents are:
* Document = 90
* Document = 109
* Document = 101
* Document = 137

Relevant documents are:
* Document = 12
* Document = 91

Query Q20:
has anyone formally determined the influence of joule heating,  produced
by the induced current,  in magnetohydrodynamic free convection flows
under general conditions 

Top 5 documents is:

Document 301 with rank 1 has cosine similarity score = 0.9981451805594265
Document 301 has the headline: effect of rheological behaviour on thermal stresses .

Document 710 with rank 2 has cosine similarity score = 0.9981451805594265
Document 710 has the headline: static aerodynamic characteristics of short blunt coneswith various nose and base cone angles at mach numbersof 0. 6 to 5. 5 and angles of attack to 180 .

Document 902 with rank 3 has cosine similarity score = 0.9981451805594265
Document 902 has the headline: the rolling up of the trailing vortex sheet and itseffect on the downwash behind wings .

Document 959 with rank 4 has cosine similarity score = 0.9981451805594265
Document 959 has the headline: some effects of bluntness on boundary layer transitionand heat transfer at supersonic speeds .

Document 1002 with rank 5 has cosine similarity score = 0.9981451805594265
Document 1002 has the headline: thermal effects on a transpiration cooled hemisphere .

Non-relevant documents are:
* Document = 902
* Document = 959

Relevant documents are:
* Document = 301
* Document = 710
* Document = 1002
```