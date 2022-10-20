'''
	Contoh Deloyment untuk Domain Data Science (DS)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import load

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None
model = load('model.pkl')

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk variabel input atau features (X) ke model
	input_harapan 		= 0
	input_pengeluaran 	= 0
	input_rerata 		= 0
	input_usia  		= 0
	
	if request.method=='POST':
		# Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
		input_harapan 		= float(request.form['harapan'])
		input_pengeluaran 	= float(request.form['pengeluaran'])
		input_rerata 		= float(request.form['rerata'])
		input_usia 			= float(request.form['usia'])
		
		# Prediksi kelas atau spesies bunga iris berdasarkan data pengukuran yg diberikan pengguna
		df_test = pd.DataFrame(data={
			"Harapan_Lama_Sekolah" : [input_harapan],
			"Pengeluaran_Perkapita"  : [input_pengeluaran],
			"Rerata_Lama_Sekolah" : [input_rerata],
			"Usia_Harapan_Hidup"  : [input_usia]
		})

		hasil_prediksi = model.predict(df_test[0:1])[0]

		# Set Path untuk gambar hasil prediksi
		if hasil_prediksi == 'Low':
			gambar_prediksi = '/static/images/low.png'

		elif hasil_prediksi == 'Normal':
			gambar_prediksi = '/static/images/right.png'

		elif hasil_prediksi == 'High':
			gambar_prediksi = '/static/images/upperRight.png'

		else:
			gambar_prediksi = '/static/images/up.png'
		
		# Return hasil prediksi dengan format JSON
		return jsonify({
			"prediksi": hasil_prediksi,
			"gambar_prediksi" : gambar_prediksi
		})

# =[Main]========================================

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	# model = load('model_iris_dt.model')

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)