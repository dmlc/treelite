import pandas
import matplotlib.pyplot as plt

def fetch(csvname):
  df = pandas.read_csv(csvname, header=0, index_col=0)

  treelite = df[df['package'] == 'treelite']
  xgboost = df[df['package'] == 'XGBoost']

  return treelite, xgboost

plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(6, 4))
treelite, xgboost = fetch('higgs.csv')

plt.plot(treelite['batchsize'], treelite['throughput'], 'o-',
         color='red', linewidth=4, label='Treelite')
plt.plot(xgboost['batchsize'], xgboost['throughput'], 'o-',
         color='blue', linewidth=4, label='XGBoost')

plt.title('higgs')
plt.xscale('log')
plt.xlabel('Batch size')
plt.ylabel('Throughput (lines/sec)')
plt.legend(loc='best')
plt.tight_layout()