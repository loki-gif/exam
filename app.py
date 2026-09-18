from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import re

app = Flask(__name__)

# Load model and preprocessing objects
with open('model.pkl', 'rb') as f:
	model = pickle.load(f)

# Some projects saved a scaler and training columns separately
scaler = None
try:
	with open('standard.pkl', 'rb') as f:
		scaler = pickle.load(f)
except FileNotFoundError:
	scaler = None

with open('training_columns.pkl', 'rb') as f:
	training_columns = list(pickle.load(f))

# Identify numeric base features (these are expected as raw numeric inputs)
NUMERIC_FEATURES = [
	'employees', 'net_income', 'market_cap',
	'legal_units_count', 'direct_subsidiaries_count', 'max_hierarchy_depth'
]

# Build categorical groups from training column names by splitting off the last "_suffix"
# Only keep groups that have multiple one-hot columns (so single numeric-like columns
# such as 'net_income' won't be shown as a dropdown).
_temp_groups = {}
pattern = re.compile(r"(.+?)_([^_]+)$")
for col in training_columns:
	m = pattern.match(col)
	if m:
		prefix, suffix = m.group(1), m.group(2)
		_temp_groups.setdefault(prefix, []).append((col, suffix))

# Keep only prefixes that have more than one column (true categorical one-hot groups)
grouped = {p: vals for p, vals in _temp_groups.items() if len(vals) > 1}

@app.route('/', methods=['GET'])
def index():
	# Build form options from grouped prefixes
	options = {prefix: [s for (_, s) in vals] for prefix, vals in grouped.items()}
	return render_template('index.html', numeric_features=NUMERIC_FEATURES, options=options)


def build_input_dataframe(form):
	# Start with zeros for all training columns
	data = {c: 0.0 for c in training_columns}

	# Fill numeric features
	for nf in NUMERIC_FEATURES:
		if nf in training_columns:
			val = form.get(nf, '')
			try:
				data[nf] = float(val) if val != '' else 0.0
			except Exception:
				raise ValueError(f'Numeric value required for {nf}')

	# Fill categorical one-hot groups
	for prefix, vals in grouped.items():
		sel = form.get(prefix)
		if sel:
			# find the exact column name matching this selection
			colname = f"{prefix}_{sel}"
			if colname in data:
				data[colname] = 1.0
			else:
				# If selection not found, leave all zeros (do not create new columns)
				pass

	# Build DataFrame with exact training column order
	X = pd.DataFrame([data], columns=training_columns)
	X = X.astype(float)
	return X


@app.route('/predict', methods=['POST'])
def predict():
	try:
		X = build_input_dataframe(request.form)
		# Apply scaler if present and compatible. Two supported cases:
		# 1) scaler was fit on all model features -> transform whole X
		# 2) scaler was fit only on numeric features -> transform numeric subset and
		#    reassemble features in original order
		if scaler is not None:
			try:
				if hasattr(scaler, 'mean_') and len(scaler.mean_) == X.shape[1]:
					X_scaled = scaler.transform(X)
				else:
					# maybe scaler was fit only on numeric columns
					numeric_cols = [c for c in NUMERIC_FEATURES if c in training_columns]
					if hasattr(scaler, 'mean_') and len(scaler.mean_) == len(numeric_cols) and len(numeric_cols) > 0:
						# scale numeric subset
						X_num = X[numeric_cols]
						X_num_s = scaler.transform(X_num)
						# reassemble full feature array in training_columns order
						row = []
						for col in training_columns:
							if col in numeric_cols:
								# pop next scaled numeric
								row.append(None)  # placeholder
							else:
								row.append(float(X.iloc[0][col]))
						# replace numeric placeholders with scaled values in order
						it = iter(X_num_s[0])
						for i, col in enumerate(training_columns):
							if col in numeric_cols:
								row[i] = float(next(it))
						X_scaled = [row]
					else:
						return render_template('result.html', error='Scaler and model feature mismatch.')
			except Exception as e:
				return render_template('result.html', error=f'Scaler transform failed: {e}')
		else:
			X_scaled = X.values

		# Model expects 2D array
		pred = model.predict(X_scaled)

		# Regression model
		result = {
			'prediction': float(pred[0])
		}

		return render_template('result.html', result=result)

	except Exception as e:
		return render_template('result.html', error=str(e))


if __name__ == '__main__':
	app.run(debug=True)

