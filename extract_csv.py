import pandas as pd

dfL12_h6 = pd.read_csv('task2_output/task2_model_predictions_l12_h6.csv')
l12_h6_bilstm = dfL12_h6[dfL12_h6['model'] == 'BiLSTM+Attention']
l12_h6_bilstm.to_csv('frontend_data/l12_h6_bilstm.csv', index=False)

dfL18_h3 = pd.read_csv('task2_output/task2_model_predictions_l18_h3.csv')
l18_h3_bilstm = dfL18_h3[dfL18_h3['model'] == 'BiLSTM+Attention']
l18_h3_bilstm.to_csv('frontend_data/l18_h3_bilstm.csv', index=False)

dfL12_h9 = pd.read_csv('task2_output/task2_model_predictions_l12_h9.csv')
l12_h9_bilstm = dfL12_h9[dfL12_h9['model'] == 'BiLSTM+Attention']
l12_h9_bilstm.to_csv('frontend_data/l12_h9_bilstm.csv', index=False)

dfL12_h12 = pd.read_csv('task2_output/task2_model_predictions_l12_h12.csv')
l12_h12_bilstm = dfL12_h12[dfL12_h12['model'] == 'BiLSTM+Attention']
l12_h12_bilstm.to_csv('frontend_data/l12_h12_bilstm.csv', index=False)
