import pandas as pd
import config
import matplotlib
import numpy as np
import ast

def getTracks(tracks):
    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
               ('track', 'genres'), ('track', 'genres_all'),
               ('track', 'genres_top')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
               ('album', 'date_created'), ('album', 'date_released'),
               ('artist', 'date_created'), ('artist', 'active_year_begin'),
               ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(
        'category', categories=SUBSETS, ordered=True)

    COLUMNS = [('track', 'license'), ('artist', 'bio'),
               ('album', 'type'), ('album', 'information')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')


    return tracks

# tracks = getTracks(pd.read_csv(config.tracks_file, index_col=0, header=[0, 1]))
genres = pd.read_csv(config.genres_file, index_col=0)
features = pd.read_csv(config.features_file, index_col=0, header=[0,1,2])
echonest = pd.read_csv(config.echonest_file, index_col=0, header=[0,1,2])

# np.testing.assert_array_equal(features.index, tracks.index)
# assert echonest.index.isin(tracks.index).all()

print("Tracks:, Genres: {}, Features: {}, Echonest:{} "
      .format(
      # tracks.shape,
      genres.shape,
      features.shape,
      echonest.shape))


# print('{} tracks, {} artists, {} albums, {} genres'.format(
#     len(tracks), len(tracks['artist', 'id'].unique()),
#     len(tracks['album', 'id'].unique()),
#     sum(genres['#tracks'] > 0)))
# mean_duration = tracks['track', 'duration'].mean()
# print('track duration: {:.0f} days total, {:.0f} seconds average'.format(
#     sum(tracks['track', 'duration']) / 3600 / 24,
#     mean_duration))

# Same as above, formated for the paper.
# plt.figure(figsize=(5, 4))
# d = tracks['album'].drop_duplicates('id')
# d = pd.Series(1, index=d['date_released'])
# d = d.resample('A').sum().fillna(0)
# b = d.index >= pd.to_datetime(1990, format='%Y')
# b &= d.index <= pd.to_datetime(2017, format='%Y')
# d[b].plot(color='k')
# plt.xlabel('release year')
# plt.ylabel('#albums')
# plt.tight_layout()
# plt.savefig('album_release_year.pdf')
#
# d.index.min().year, d.index.max().year
#
# for effect in ['artist', 'album']:
#     d = tracks[effect, 'id'].value_counts()
#     ipd.display(d.head(5))
#     p = sns.distplot(d[(d.values < 50) & (d.values >= 0)], kde=False)
#     p.set_xlabel('#tracks per ' + effect);
#     p.set_ylabel('#' + effect + 's');