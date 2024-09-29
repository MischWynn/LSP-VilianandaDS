import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3
import joblib


model = joblib.load("housing.sav")
df = pd.read_csv('housing.csv')
df ['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)
df.drop(['ocean_proximity'],axis=1,inplace=True)

data_understanding = "<h2>Pemahaman Data</h2> Didapatkan data dengan mencakup variabel sebagai berikut:<li>longitude: Ukuran seberapa jauh barat sebuah rumah; nilai yang lebih tinggi menunjukkan lebih jauh ke barat</li> <li>latitude: Ukuran seberapa jauh utara sebuah rumah; nilai yang lebih tinggi menunjukkan lebih jauh ke utara</li> <li>housingMedianAge: Usia median rumah dalam satu blok; angka yang lebih rendah menunjukkan bangunan yang lebih baru</li> <li>totalRooms: Jumlah total kamar dalam satu blok</li><li>totalBedrooms: Jumlah total kamar tidur dalam satu blok</li> <li>population: Jumlah total penduduk yang tinggal dalam satu blok</li><li>households: Jumlah total rumah tangga, sekelompok orang yang tinggal dalam satu unit rumah, untuk satu blok</li><li>medianIncome: Pendapatan median untuk rumah tangga dalam satu blok rumah (diukur dalam puluhan ribu Dolar AS)</li><li>medianHouseValue: Nilai median rumah untuk rumah tangga dalam satu blok (diukur dalam Dolar AS)</li><li>oceanProximity: Lokasi rumah terhadap laut/samudra</li>"

X = df.drop('median_income',axis=1)
y = df['median_income']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
y_pred=model.predict(X_test)
accuracy = round(model.score(X_test, y_test)*100, 2) 



st.title('Portofolio')
st.markdown(data_understanding, unsafe_allow_html=True)
st.dataframe(df)
st.header("Model Accuracy")
st.subheader(f"{accuracy}%")

st.title('Heatmap of Dataset')
heatmap = plt.figure() 
sns.heatmap(df.corr(),annot=True)
st.pyplot(heatmap)

st.title('Distribution of Median income')
distribution = plt.figure() 
sns.set(style = 'whitegrid')
sns.histplot(df['median_income'], kde=True)
plt.title('Distribution of Median Income')
plt.ylabel('Frequency of Value')
plt.ticklabel_format(style = 'plain')
distribution = mpld3.fig_to_html(distribution)
components.html(distribution, height=600)

st.title("Prediction Result")
st.write("Model ini menggunakan data yang berjumlah 20 ribu dan dibagi menjadi 14 ribu data untuk training dan 6 ribu data untuk testing")
scatter_pred = plt.figure() 
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test, color='red')
plt.xlabel('Test Data')
plt.ylabel('Pred Data')
plt.title('')
st.pyplot(scatter_pred)
st.write("Titik-titik biru mewakili pasangan data uji dan hasil prediksi, sementara garis merah lurus adalah garis referensi yang menunjukkan di mana prediksi seharusnya jatuh jika model prediksi benar-benar akurat (y = x). Dari plot ini, tampak bahwa sebagian besar titik mendekati garis merah, menandakan bahwa prediksi model cukup baik, meskipun ada beberapa pencilan (outliers) yang cukup jauh dari garis, terutama di area atas dan kanan grafik.")
