import matplotlib.pyplot as plt
import pandas

def load_bench(df):
  df2 = df.groupby(['engine', 'batchsize'])\
          .agg({'elapsed_time': ['median', 'count']})
  df2['throughput', 'median'] \
    = df2.index.get_level_values('batchsize') / df2['elapsed_time', 'median']
  return df2

def plot_bench(df, title, label='', engine='Treelite', color='r'):
  df2 = df.loc[engine]
  handle = plt.plot(df2.index, df2['throughput', 'median'], '-o',
                    linewidth=3, color=color, label=label)
  plt.xscale('log')
  plt.xlabel('Batch size')
  plt.ylabel('Throughput (rows/sec)')
  plt.title(title)
  plt.tight_layout()
  return handle[0]

plt.rcParams.update({'font.size':14})
plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)
l1 = plot_bench(load_bench(pandas.read_csv('allstate-treelite.csv')), 'allstate',
                      color='r')
l3 = plot_bench(load_bench(pandas.read_csv('allstate-xgb.csv')), 'allstate',
                      color='k', engine='XGBoost')
plt.ylim(ymin=0)

plt.subplot(1, 3, 2)
plot_bench(load_bench(pandas.read_csv('higgs-treelite.csv')), 'higgs',
                      color='r')
plot_bench(load_bench(pandas.read_csv('higgs-xgb.csv')), 'higgs',
                      color='k', engine='XGBoost')
plt.ylim(ymin=0)

plt.subplot(1, 3, 3)
plot_bench(load_bench(pandas.read_csv('yahoo-treelite.csv')), 'yahoo',
                      color='r')
plot_bench(load_bench(pandas.read_csv('yahoo-xgb.csv')), 'yahoo',
                      color='k', engine='XGBoost')
plt.ylim(ymin=0)

plt.figlegend((l1, l3),
              ('Treelite', 'XGBoost'),
              loc='lower center', ncol=2,
              bbox_to_anchor=[0.5, -0.03], frameon=False)
plt.tight_layout()
