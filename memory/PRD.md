# FunduX - PRD (Product Requirements Document)

## Original Problem Statement
Kullanici, FunduX projesinin AR-GE portal giris formunu incelenmesini ve kodda gerekli iyilestirmelerin yapilmasini istedi.

## Project Overview
FunduX: Fundus goruntlerinden diyabetik retinopatinin erken evrede tespiti ve cok evreli derecelendirilmesi icin derin ogrenme tabanli klinik karar destek altyapisi.

## Architecture
- **Training Scripts**: 4 model (ConvNeXt-XL, AttentionUNet, DeepLabv3+ResNet50, DeepLabv3+ResNet101)
- **Inference App**: Gradio tabanli web arayuzu (app.py)
- **Data**: 18,444 fundus goruntusu (APTOS 2019, DDR, IDRiD, Messidor-2)
- **Experiment Tracking**: Comet ML
- **5 Classes**: No DR, Mild, Moderate, Severe, Proliferative

## What's Been Implemented (Jan 2026)
1. AR-GE portal giris formu detayli incelendi ve paragraf paragraf duzeltme onerileri verildi
2. Kod degisiklikleri uyguland:
   - `app.py`: Sinif isimleri duzeltildi (Class 0-4 -> No DR, Mild, Moderate, Severe, Proliferative)
   - `app.py`: Model yollari hardcoded Colab yollarindan dinamik os.path.join'e gecildi
   - Tum 4 egitim scriptine class-weighted cross-entropy loss eklendi (sinif dengesizligi icin)

## Backlog
### P0 (Critical)
- [ ] Model dosyalarini Git LFS'den indirme (toplam ~1.5GB+)
- [ ] Egitim sonuclarini dokumana ekleme (model karsilastirma tablosu)

### P1 (Important)
- [ ] CE/FDA onay sureci planlama
- [ ] Focal loss deneysel degerlendirme
- [ ] Kaynak bazli (per-dataset) performans analizi

### P2 (Nice to have)
- [ ] Web uygulamasina donusturme (React + FastAPI)
- [ ] Glokom ve AMD modulleri entegrasyonu
- [ ] KVKK uyumluluk dokumantasyonu

## User Personas
- Akademisyenler (Sanayi Bakanligi denetcileri)
- Aile hekimlikleri
- Goz uzmanlari
- Saglik tarama ekipleri
