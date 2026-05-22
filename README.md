# 🇹🇷 Makemore Lesson 1 — Türkçe İsim Üretici

Bu proje, **Andrej Karpathy'nin Makemore dersinden** esinlenerek hazırlanmış bir öğrenme defteridir. Amaç, Türkçe isimlerden oluşan bir veri seti üzerinde önce klasik **bigram frekans modeli**, ardından PyTorch ile basit bir **tek katmanlı nöral ağ** kurarak yeni isimler üretmektir.

Bu çalışma bir “hazır model” projesinden çok, karakter seviyesinde dil modellemenin temel mantığını anlamaya yönelik açıklamalı bir notebook'tur. Kod hücrelerinin yanında, her adımın neden yapıldığını anlatan detaylı Markdown notları bulunur.

---

## 🎯 Projenin Amacı

Bu notebook'ta şu soruya cevap arıyoruz:

> Bir model, sadece Türkçe isimlerdeki harf geçişlerine bakarak kulağa Türkçe gibi gelen yeni isimler üretebilir mi?

Bunun için model karakterleri tek tek işler. Örneğin `deniz` ismi şu geçişlere ayrılır:

```text
. → d
d → e
e → n
n → i
i → z
z → .
```

Buradaki `.` karakteri hem kelime başlangıcını hem de kelime bitişini temsil eder.

---

## 🧠 Bu Projede Öğrenilenler

Bu defterde aşağıdaki temel konular uygulamalı olarak ele alınır:

- Türkçe karakter normalizasyonu
- Bigram mantığı
- Karakterden indekse ve indeksten karaktere dönüşüm
- Frekans matrisi oluşturma
- Olasılık dağılımı üretme
- `torch.multinomial` ile örnekleme alma
- Model smoothing
- Negative Log-Likelihood loss
- One-hot encoding
- Logits ve softmax dönüşümü
- PyTorch ile forward pass
- Backpropagation
- Gradient descent
- L2 regularization
- Eğitilmiş modelden yeni isim üretimi

---

## 📚 Kullanılan Veri Seti

Bu çalışmada Kaggle üzerindeki **Türkçe İsimler** veri seti kullanılmıştır:

[Turkish Names / Türkçe İsimler Dataset](https://www.kaggle.com/datasets/ardaorcun/turkish-names-turkce-isimler)

Veri seti üzerinde şu ön işlemler uygulanmıştır:

1. İsimler satır satır okunur.
2. Birden fazla parçadan oluşan isimler ayrıştırılır.
3. Tüm karakterler küçük harfe çevrilir.
4. Türkçe karakterler özel dönüşüm tablosuyla normalize edilir.
5. Unicode kaynaklı karakter farklılıkları giderilir.
6. Tekrarlayan isimler temizlenir.
7. İsimler alfabetik olarak sıralanır.

Özellikle Türkçe `İ` ve `I` harfleri için standart `lower()` kullanmak bazı Unicode problemlerine yol açabildiğinden özel bir normalizasyon fonksiyonu kullanılmıştır.

---

## 🧩 Proje Mantığı

Proje iki ana bölümden oluşur.

### 1. Sayma Tabanlı Bigram Modeli

İlk bölümde model herhangi bir nöral ağ kullanmaz. Sadece veri setindeki harf geçişlerini sayar.

Örneğin:

```text
a → l geçişi kaç kez olmuş?
e → m geçişi kaç kez olmuş?
. → a geçişi kaç kez olmuş?
n → . geçişi kaç kez olmuş?
```

Bu sayımlar bir frekans matrisinde tutulur. Daha sonra her satır kendi toplamına bölünerek olasılık dağılımına çevrilir.

Yani model şunu öğrenir:

```text
Bir isim a harfiyle başladıysa, sıradaki harf en yüksek ihtimalle ne olabilir?
Bir harften sonra isim bitme olasılığı ne kadardır?
Türkçe isimlerde hangi harf geçişleri daha yaygındır?
```

---

### 2. Tek Katmanlı Nöral Ağ Modeli

İkinci bölümde aynı problem PyTorch ile nöral ağ yaklaşımı kullanılarak çözülür.

Bu sefer model frekans matrisini doğrudan kullanmaz. Bunun yerine her karakteri one-hot vektöre dönüştürür ve öğrenilebilir bir ağırlık matrisiyle işler:

```python
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)
```

Bu yapı aslında çok basit bir sinir ağıdır:

```text
Karakter indeksi → One-hot vektör → Linear layer → Softmax → Sonraki karakter olasılıkları
```

Modelin amacı, gerçek veri setindeki doğru sonraki harflere daha yüksek olasılık vermeyi öğrenmektir.

---

## 🔢 Loss Mantığı

Modelin ne kadar iyi tahmin yaptığını ölçmek için **Negative Log-Likelihood** kullanılır.

Eğer model doğru harfe yüksek olasılık verirse loss düşük olur:

```text
Doğru harfin olasılığı yüksek → log değeri daha iyi → NLL düşük
```

Eğer model doğru harfe düşük olasılık verirse loss yükselir:

```text
Doğru harfin olasılığı düşük → log değeri kötü → NLL yüksek
```

Bu yüzden eğitim sürecindeki hedef şudur:

```text
Ortalama NLL loss değerini mümkün olduğunca düşürmek
```

Ek olarak modele aşırı keskin ve ezberci tahminler yaptırmamak için loss fonksiyonuna küçük bir **L2 regularization** terimi eklenir:

```python
loss = nll + 0.01 * (W**2).mean()
```

Bu terim ağırlıkların gereksiz yere büyümesini engeller ve modelin daha dengeli olmasına yardımcı olur.

---

## 🛠️ Kullanılan Teknolojiler

- Python
- PyTorch
- Matplotlib
- Unicode normalization
- Jupyter Notebook

---

## 🎲 Örnek Model Akışı

Model isim üretirken şu adımları izler:

1. Başlangıç karakteri olan `.` ile başlar.
2. Mevcut karakter için olasılık dağılımını hesaplar.
3. Bu dağılıma göre rastgele bir sonraki karakteri seçer.
4. Seçilen karakteri yeni mevcut karakter yapar.
5. Tekrar `.` karakteri gelene kadar üretime devam eder.

Basitleştirilmiş akış:

```text
. → a → l → i → n → .
```

Bu durumda modelin ürettiği isim:

```text
alin
```

Üretilen isimler her zaman gerçek bir isim olmak zorunda değildir. Ama model Türkçe isimlerdeki harf geçişlerini öğrendiği için çıktılar genellikle Türkçe isimlere benzer bir yapıda olur.

---

## 📊 Görselleştirme

Notebook içinde bigram frekans matrisi bir ısı haritası olarak görselleştirilir.

Bu görselleştirme sayesinde:

- Hangi harf geçişlerinin daha sık olduğu,
- Hangi karakterlerin isim başında daha çok kullanıldığı,
- Hangi harflerden sonra isimlerin daha sık bittiği,
- Türkçe isimlerdeki karakter örüntülerinin nasıl dağıldığı

daha kolay gözlemlenebilir.

---

## 🧪 Bu Proje Neden Önemli?

Bu proje küçük görünse de dil modellerinin temelinde yer alan birçok fikri sade bir şekilde gösterir.

Bugünkü büyük dil modelleri çok daha karmaşık mimariler kullanır; ancak temel fikir hâlâ benzerdir:

```text
Geçmiş bağlama bak → sıradaki token için olasılık dağılımı üret → en uygun çıktıyı seç
```

Bu notebook'ta bu mantık en küçük yapı taşına indirgenmiştir:

```text
Önceki karaktere bak → sonraki karakter için olasılık üret
```

Bu yüzden proje, dil modelleme mantığını öğrenmek için iyi bir başlangıç noktasıdır.

---

## 🔮 Geliştirme Fikirleri

Bu proje ileride şu şekillerde geliştirilebilir:

- Bigram yerine trigram model kurulabilir.
- Daha büyük bir Türkçe isim veri seti kullanılabilir.
- Üretilen isimler kadın/erkek isimleri olarak ayrıştırılabilir.
- Train/validation/test ayrımı eklenebilir.
- Modelin ezberleme seviyesi ölçülebilir.
- Daha derin bir nöral ağ denenebilir.
- Embedding layer eklenebilir.
- Transformer tabanlı küçük bir karakter modeli kurulabilir.
- Üretilen isimler için basit bir web arayüzü hazırlanabilir.

---

## 🙋‍♂️ Kişisel Not

Bu çalışma, Andrej Karpathy'nin Makemore dersini takip ederken konuyu daha iyi anlamak ve öğrendiklerimi kalıcı hale getirmek için hazırlanmıştır.

Amacım sadece çalışan bir kod yazmak değil; her adımda modelin arka planda ne yaptığını anlamak, not almak ve bunu açıklanabilir bir proje haline getirmekti.

Bu yüzden notebook içinde bolca açıklama, ara çıktı ve kavramsal not bulunmaktadır.

---

## ⭐ Özet

Bu repo, Türkçe isimler üzerinden karakter seviyesinde dil modellemenin temelini gösteren açıklamalı bir öğrenme projesidir.

Kısaca:

```text
Türkçe isimler → Bigram frekansları → Olasılık matrisi → Sampling → Nöral ağ → Yeni isim üretimi
```


