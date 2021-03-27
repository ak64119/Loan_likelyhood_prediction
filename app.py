from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np
import pandas as pd
# for WOE and scorecard function
import utilities
from utilities.woe import woe_conversion, woe_graph, woe_analysis, mono_bin, char_bin
from utilities.scorecard import scorecard

app = Flask(__name__)
mdl_values = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
	d = None
	if request.method == 'POST':
		print('POST received')
		d = request.form.to_dict()
	else:
		print('GET received')
		d = request.args.to_dict()

	print("Dataframe format required for Machine Learning prediction")
	df = pd.DataFrame([d.values()], columns=d.keys())

	#change datatype of column
	df = df.astype({'Age': np.float,'Prv_Loan_Flag': np.float, 'Avg_amt_CA_txn': np.float, 
		'Num_txns': np.float, 'Lst_txn_amt': np.float})

	feat_woe = woe_conversion(df,mdl_values[1])
	# Generate credit score for all observations, and generate score table
	X_score_tab, X_scored = scorecard(feat_woe, mdl_values[0], mdl_values[1], 600, 50, 20)
	score = X_scored[['total_score']]
	df = pd.concat([df, score], axis=1)

	output = np.where(df.total_score>536.00, 'Very Low Likelihood',
                           np.where(df.total_score>479.00,'Low Likelihood',
                                   np.where(df.total_score>455.00,'Medium Likelihood', 
                                           np.where(df.total_score>430.00,'High Likelihood',
                                                    'Very High Likelihood'))))

	return render_template('home.html', prediction_text="This customer has {} of taking a loan.".format(output[0]))


if __name__ == '__main__':
	app.run(debug=True)