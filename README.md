# MobileViT face recognition

Ovaj repozitorij je eksperimentalni projekt za ekstrakciju embeddinga lica i jednostavno prepoznavanje osoba pomoću modela `mobilevitv2_050.cvnets_in1k`.

## Što projekt radi

- Učitava unaprijed istrenirani MobileViT model iz `timm`.
- Detektira lice pomoću `facenet-pytorch` `MTCNN` modela.
- Ekstrahira embedding za lice i uspoređuje ga s centroidima spremljenima u `.npy` datotekama.
- Omogućuje generiranje centroida iz trening skupa i testiranje na zasebnim skupovima slika.

## Struktura

- [Embedding_model.py](Embedding_model.py) - učitavanje modela i transformacije.
- [utils.py](utils.py) - detekcija lica i izrada embeddinga.
- [baza_znacajki.py](baza_znacajki.py) - generiranje centroida i spremanje u `.npy` datoteke.
- [prepoznavanje_lica.py](prepoznavanje_lica.py) - testiranje prepoznavanja i live demo preko kamere.
- [klasifikacija_test.py](klasifikacija_test.py) - evaluacija na ImageNet-val skupu.
- [znacajke.py](znacajke.py) - eksperimentalna PCA analiza embeddinga.

## Potrebno

- Python 3.10+.
- PyTorch kompatibilan s tvojim CPU ili GPU okruženjem.
- Dataset organiziran kao:

```text
dataset/
  train/
    ImeOsobe/
      slika1.jpg
      slika2.jpg
  random/
    ImeOsobe/
      test1.jpg
  val/
```

Ako koristiš ImageNet evaluaciju, trebaš i folder `imagenet-val/` s klasama po ImageNet ID-u.

## Instalacija

```bash
pip install -r requirements.txt
```

Ako instaliraš PyTorch ručno zbog CUDA verzije, prvo instaliraj odgovarajuću verziju prema službenim PyTorch uputama, a zatim ostale pakete iz `requirements.txt`.

## Kako pokrenuti

Generiranje centroida:

```bash
python baza_znacajki.py
```

Testiranje na pripremljenom skupu:

```bash
python prepoznavanje_lica.py
```

Webcam demo koristi kameru računala i izlaz prekida tipkom `q`.

Evaluacija na ImageNet-val:

```bash
python klasifikacija_test.py
```

Eksperimentalna PCA analiza embeddinga:

```bash
python znacajke.py
```

## Napomene

- Prije testiranja prepoznavanja mora postojati `centroid_znacajke.npy` i `oznake.npy`.
- Ako skripta ne pronađe lice, vraća `None` i preskače tu sliku.
- Putovi u skriptama su trenutno relativni prema korijenu repozitorija.
