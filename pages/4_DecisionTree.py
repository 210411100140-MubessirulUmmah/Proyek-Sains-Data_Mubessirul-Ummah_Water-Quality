import streamlit as st
import joblib
import pandas as pd
import sklearn

st.set_page_config(
    page_title="Water Quality Prediction Decision Tree",
    page_icon="🌊",
    layout="wide"
)


tab1, tab2, tab3 = st.tabs(["Data", "Prediksi", "Biodata"])

with tab1:
    st.title("🌊 Water Quality Dataset")

    st.markdown("""
    Datasest waterquality merupakan kumpulan data yang dibuat dari data imajiner kualitas air di lingkungan perkotaan. Dataset ini mencakup data kadar mikroorganisme yang terkandung di dalam air seperti kadar aluminium, ammonia, arsenic, barium, cadmium, chloramine, chromium, copper, flouride, bacteria, viruses, lead, nitrates, nitrites, mercury, perchlorate, radium, selenium, silver, dan uranium. Data tersebut berisi 21 atribut dan 7999 record, record tersebut diberi label dengan variabel kelas is_safe, yang memungkinkan klasifikasi data menggunakan nilai 1 (safe atau aman) dan 0 (not_safe atau tidak aman) dikonsumsi. data ini saya dapatkan dari Kaggel dengan link berikut : <a href='https://www.kaggle.com/datasets/mssmartypants/water-quality/'>Disini</a>. 
    
    Dataset tersebut berisi "synthetic water quality data" atau "simulated water quality data" atau dengan kata lain sample dari data yang bersifat fiktif atau imajiner yang dibuat untuk keperluan pendidikan dan latihan.
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Repository Github
    https://github.com/210411100140-MubessirulUmmah/Proyek-Sains-Data_Mubessirul-Ummah_Water-Quality
    """, unsafe_allow_html=True)

    df = pd.read_csv("waterQuality1.csv")
    st.write("Dataset Kualitas Air : ")
    st.write(df)
    st.write("Penjelasan fitur-fitur yang ada")

    st.write("""
    1.   **Aluminium** : Merupakan kandungan aluminium dalam air. berbahaya jika lebih besar dari 2,8. kandungan aluminium yang berlebihan dapat menyebabkan masalah kesehatan, terutama pada sistem saraf.
    2.   **Ammonia** : Kandungan ammonia (NH3) dalam air. Amonia adalah gas dengan bau yang tajam dan beracun dalam konsentrasi tinggi. berbahaya jika lebih besar dari 32,5. Kandungan ammonia yang tinggi dalam air dapat menyebabkan kerusakan organisme akuatik dan merusak kualitas air minum.
    2.   **Arsenic** : Kandungan arsenik dalam air. Arsenic adalah unsur kimia dalam tabel periodik dengan simbol As dan nomor atom 33. Arsenic dapat ditemukan secara alami di dalam kerak bumi dan digunakan dalam berbagai aplikasi industri, termasuk pembuatan kayu tahan air. arsenic berbahaya jika lebih besar dari 0,01. Kandungan arsenik yang tinggi dalam air minum dapat menyebabkan keracunan dan meningkatkan risiko kanker.
    2.   **Barium**: Kandungan barium dalam air. Barium adalah unsur kimia dengan simbol Ba dan nomor atom 56. Barium digunakan dalam industri minyak dan gas, serta dalam radiografi medis. berbahaya jika lebih besar dari 2. Pemaparan jangka panjang terhadap barium dapat menyebabkan kerusakan organ dalam tubuh manusia.
    2.   **Cadmium** : Kandungan kadmium dalam air. Cadmium adalah unsur kimia dengan simbol Cd dan nomor atom 48. Cadmium digunakan dalam baterai, cat, dan plastik. berbahaya jika lebih besar dari 0,005. Pemaparan cadmium dapat menyebabkan masalah kesehatan serius, termasuk kerusakan ginjal dan kanker.
    2.   **Chloramine** : Kandungan chloramine dalam air. Chloramine adalah senyawa kimia yang terbentuk dari klorin dan amonia. Ini digunakan sebagai desinfektan dalam air minum. berbahaya jika lebih besar dari 4. Paparan kloramine dalam jumlah yang tinggi dapat menyebabkan iritasi mata dan tenggorokan.
    2.   **Chromium** : Kandungan kromium dalam air. berbahaya jika lebih besar dari 0,1. Pemaparan kromium VI dapat menyebabkan kerusakan paru-paru, penyakit pernapasan, dan kanker.
    2.   **Copper** : Kandungan tembaga dalam air. Copper adalah unsur kimia dengan simbol Cu dan nomor atom 29. Copper digunakan dalam instalasi listrik, pipa, dan peralatan masak. berbahaya jika lebih besar dari 1,3. Kandungan tembaga yang berlebihan dalam air minum dapat menyebabkan gangguan pencernaan dan masalah hati.
    2.   **Fluoride** : Kandungan fluoride dalam air. Fluoride adalah ion anorganik yang penting untuk kesehatan gigi. berbahaya jika lebih besar dari 1,5. konsumsi fluoride dalam jumlah yang berlebihan dapat menyebabkan masalah kesehatan gigi dan tulang.
    2.   **Bacteria** : Indikator keberadaan bakteri dalam air. Bakteri adalah mikroorganisme yang dapat ditemukan dalam air. berbahaya jika lebih besar dari 0.
    2.   **Viruses** : Indikator keberadaan virus dalam air. berbahaya jika lebih besar dari 0
    2.   **Lead** : Kandungan timbal dalam air. Lead adalah logam berat yang dapat menyebabkan keracunan, terutama pada anak-anak. berbahaya jika lebih besar dari 0,015. Pemaparan timbal dapat menyebabkan kerusakan otak dan sistem saraf.
    2.   **Nitrates** : Kandungan nitrat dalam air. Nitrates adalah senyawa kimia yang dapat ditemukan dalam pupuk dan limbah industriberbahaya jika lebih besar dari 10. Kandungan nitrates yang tinggi dalam air dapat menyebabkan masalah kesehatan, terutama pada bayi.
    2.   **Nitrites** : Kandungan nitrit dalam air. nitrites adalah senyawa kimia yang dapat ditemukan dalam pupuk dan limbah industriberbahaya jika lebih besar dari 1. Kandungan nitrites yang tinggi dalam air dapat menyebabkan masalah kesehatan, terutama pada bayi.
    2.   **Mercury** : Kandungan merkuri dalam air. Mercury adalah logam berat yang dapat mengakumulasi dalam organisme hidup dan menyebabkan keracunan. berbahaya jika lebih besar dari 0,002. Pemaparan merkuri dapat merusak otak, ginjal, dan sistem saraf.
    2.   **Perchlorate** : Kandungan perchlorate dalam air. Perchlorate adalah senyawa kimia yang digunakan dalam bahan peledak dan propelan roket. Pemaparan perchlorate dapat mengganggu fungsi tiroid. berbahaya jika lebih besar dari 56
    2.   **Radium** : Kandungan radium dalam air. Radium adalah unsur radioaktif yang dapat ditemukan secara alami dalam tanah dan air. Paparan radium dapat meningkatkan risiko kanker. berbahaya jika lebih besar dari 5
    2.   **Selenium** : Kandungan selenium dalam air. berbahaya jika lebih besar dari 0,5. konsumsi selenium yang berlebihan dapat menyebabkan masalah kesehatan, termasuk kerusakan saraf.
    2.   **Silver** : Kandungan perak dalam air. berbahaya jika lebih besar dari 0,1. Konsumsi perak dalam jumlah yang berlebihan dapat menyebabkan argyria, kondisi di mana kulit manusia berubah menjadi warna biru keabu-abuan.
    2.   **Uranium** : Kandungan uranium dalam air. Uranium adalah unsur radioaktif yang dapat ditemukan secara alami dalam batuan dan air. Paparan uranium dapat meningkatkan risiko kanker dan masalah ginjal. berbahaya jika lebih besar dari 0,3
    2.   **Is_safe** : Kolom ini adalah label atau target variabel yang menunjukkan apakah sampel air tersebut aman untuk dikonsumsi atau tidak. class attribute {0 - not safe, 1 - safe}
    fitur di atas ini mencerminkan kandungan berbagai mikroorganisme dalam air. Dalam setiap fitur atau kandungan yang ada dalam air tersebut memiliki batasan aman. jika kandungan mikroorganisme melebihi nilai-nilai batasan ini, air dianggap tidak aman untuk konsumsi manusia.
    """,unsafe_allow_html=True)

with tab2:
    st.title("🌊 Water Quality Prediction Decision Tree")
    st.image("grafik perbandingan metode.png", caption="Grafik Perbandingan Metode")
    st.write('Dalam prediksi kali ini saya akan menggunakan model Adaboost, dimana fitur yang digunakan berjumlah 19 dengan akurasi pelatihan mencapai 98,33%.')
    fitur = joblib.load('fiturdecisiontree.pkl')
    st.write(fitur)

    aluminium = st.number_input("Kandungan Aluminium : ")
    ammonia = st.number_input("Kandungan Ammonia : ")
    arsenic = st.number_input("Kandungan Arsenic : ")
    barium = st.number_input("Kandungan Barium : ")
    cadmium = st.number_input("Kandungan Cadmium : ")
    chloramine = st.number_input("Kandungan Chloramine : ")
    chromium = st.number_input("Kandungan Chromium : ")
    copper = st.number_input("Kandungan Tembaga : ")
    flouride = st.number_input("Kandungan Flourida : ")
    bacteria = st.number_input("Kandungan Bakteri : ")
    viruses = st.number_input("Kandungan Virus : ")
    nitrates = st.number_input("Kandungan Nitrat : ")
    nitrites = st.number_input("Kandungan Nitrit : ")
    mercury = st.number_input("Kandungan Mercuri : ")
    perchlorate = st.number_input("Kandungan Perchlorate : ")
    radium = st.number_input("Kandungan Radium : ")
    selenium = st.number_input("Kandungan Selenium : ")
    silver = st.number_input("Kandungan Perak : ")
    uranium = st.number_input("Kandungan Uranium : ")


    results = []
    data = {'aluminium' : aluminium,
            'ammonia' : ammonia,
            'arsenic' : arsenic,
            'barium' : barium,
            'cadmium' : cadmium,
            'chloramine' : chloramine,
            'chromium' : chromium,
            'copper' : copper,
            'flouride' : flouride,
            'bacteria' : bacteria,
            'viruses' : viruses,
            'nitrates' : nitrates,
            'nitrites' : nitrites,
            'mercury' : mercury,
            'perchlorate' : perchlorate,
            'radium' : radium,
            'selenium' : selenium,
            'silver' : silver,
            'uranium' : uranium}


    results.append(data)
    data_implementasi = pd.DataFrame(results)


    if st.button("Cek Prediksi"):
        scaler = joblib.load('saclerdecisiontree.pkl')
        data_uji_scaled = scaler.transform(data_implementasi)
        st.write('Data Inputan',data_implementasi)
        st.write('Data Normalisasi',data_uji_scaled)

        # Load the model
        dtmodel = joblib.load('modeldecisiontree.pkl')

        # Make predictions
        prediksi = dtmodel.predict(data_uji_scaled)
        if (prediksi == 0):
            st.write('Hasil Prediksi : not safe')
        elif (prediksi == 1):
            st.write('Hasil Prediksi : safe')


with tab3:
    st.markdown("""
    <h1>Biodata</h1>
    """, unsafe_allow_html=True)
    st.write('''
    Nama  : Mubessirul Ummah

    NIM   : 210411100140

    Kelas : Proyek Sains Data B
    ''')