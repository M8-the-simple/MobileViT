# MobileViT / InsightFace prepoznavanje lica

Ovaj repozitorij sadrzi eksperimentalni pipeline za detekciju lica, izradu embeddinga i prepoznavanje osoba usporedbom embeddinga s centroidima.

Kroz kod su podrzana dva pristupa:

- `Embedding_model_insightface.py` je trenutni default i koristi InsightFace `buffalo_sc` model.
- `Embedding_model.py` je alternativni `timm` pipeline s modelom `mobilevitv2_050.cvnets_in1k`.

## Sto projekt radi

- Detektira lice pomocu `facenet-pytorch` MTCNN-a ili Haar cascadea.
- Iz svakog lica racuna embedding vektor.
- Gradi centroid po osobi iz trening skupa.
- Prepoznaje novu sliku ili video frame usporedbom embeddinga s spremljenim centroidima.
- Sadrzi dodatne skripte za ekstrakciju frameova, testiranje i vizualizaciju.

## Struktura projekta

- [utils.py](utils.py) - zajednicke pomocne funkcije za detekciju, embedding i ucitavanje centroida.
- [detectors.py](detectors.py) - omotac za Haar i MTCNN detekciju lica.
- [Embedding_model_insightface.py](Embedding_model_insightface.py) - default model za embeddinge.
- [Embedding_model.py](Embedding_model.py) - alternativa temeljena na `timm` modelu.
- [baza_znacajki.py](baza_znacajki.py) - generira i sprema centroida po osobama.
- [prepoznavanje_lica.py](prepoznavanje_lica.py) - pokrece prepoznavanje na video ulazu ili kameri.
- [frameExtractor.py](frameExtractor.py) - ekstrahira frameove iz videa i sprema ih u dataset.
- [tests.py](tests.py) - pomocne funkcije za intra i inter class testove.
- [klasifikacija_test.py](klasifikacija_test.py) - primjer evaluacije klasifikacijskog modela na ImageNet val skupu.
- [znacajke.py](znacajke.py) - eksperimentalna analiza embeddinga i PCA vizualizacija.

## Zahtjevi

- Python 3.10+.
- PyTorch instaliran prema CPU ili CUDA okruzenju.
- Paket ovisnosti iz [requirements.txt](requirements.txt).

Instalacija:

```bash
pip install -r requirements.txt
```

Ako instaliras PyTorch rucno zbog CUDA verzije, prvo instaliraj odgovarajucu PyTorch verziju, a zatim ostatak paketa iz `requirements.txt`.

## Organizacija podataka

Za generiranje centroida i prepoznavanje, trening skup treba biti organiziran ovako:

```text
dataset/
  train/
    Antonio/
      slika1.jpg
      slika2.jpg
    David/
      slika1.jpg
    Fabris/
      slika1.jpg
```

Za dodatno testiranje mozes koristiti i vlastite foldere s validacijskim ili videosnimkama. Neke skripte u repozitoriju imaju hardcoded putanje, pa ih po potrebi prilagodi lokalnom okruzenju.

## Centroidi i modeli

Spremljeni centroidi se cuvaju u:

```text
centroids/<model.name>/centroid_znacajke.npy
centroids/<model.name>/oznake.npy
```

Za InsightFace to je trenutno `centroids/buffalo_sc/`, a za `timm` model koristi se mapa s imenom modela, npr. `centroids/mobilevitv2_050.cvnets_in1k/`.

Pragovi odluke definirani su u [config.py](config.py). Ako dodes do novog modela ili novih osoba, tu je mjesto za podesavanje thresholda po osobi.

## Kako pokrenuti

Generiranje centroida iz `dataset/train`:

```bash
python baza_znacajki.py
```

Prepoznavanje lica nad video ulazom ili kamerom:

```bash
python prepoznavanje_lica.py
```

Ekstrakcija frameova iz videa u trening skup:

```bash
python frameExtractor.py
```

Testovi na podacima iz skupa:

```bash
python tests.py
```

Eksperimentalna evaluacija klasifikacije:

```bash
python klasifikacija_test.py
```

Vizualizacija embeddinga i PCA analize:

```bash
python znacajke.py
```

## Vazne napomene

- Prije prepoznavanja moraju postojati spremljeni centroidi za trenutni model.
- Ako detektor ne pronade lice, funkcije vracaju `None` i taj primjer se preskace.
- `utils.py` koristi lazy inicijalizaciju modela i detektora, pa prvi poziv moze biti sporiji.
- `prepoznavanje_lica.py` i `baza_znacajki.py` su najkorisnije ulazne tocke ako zelis brzo provjeriti cijeli pipeline.

## Brzi tijek rada

1. Organiziraj slike po osobama u `dataset/train/`.
2. Pokreni `python baza_znacajki.py` da dobijes centroidi.
3. Pokreni `python prepoznavanje_lica.py` za provjeru na videu ili kameri.
4. Po potrebi prilagodi pragove u [config.py](config.py).
