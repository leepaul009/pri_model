
### numpy tips
```
np.set_printoptions(precision=1)
numpy.set_printoptions(threshold=sys.maxsize)

import sys
np.set_printoptions(threshold=sys.maxsize)

# use '.item()' uploaded dict object
# ex. a_dict = dict_obj.item()

np.savez(f, x=a_dict)
data = np.load(f, allow_pickle=True)
a_dict = data['x'].item()


# using numpy.vectorize.
import numpy as np
x = np.array([1, 2, 3, 4, 5])
squarer = lambda t: t ** 2
vfunc = np.vectorize(squarer)
vfunc(x)
# Output : array([ 1,  4,  9, 16, 25])



np.mean(arr, axis=0)
np.mean(arr, axis=1)



arr = np.empty((0,3), int)
arr = np.append(arr, np.array([[10,20,30]]), axis=0)
```

### pandas tips
```
df['col'].apply(lambda x: map_local[x])

# use dates as key, thus new_series like a map
new_series = pd.Series(data['numerical_column'].values , index=data['dates'])


# Below are some quick examples

# Apply a lambda function to each column
df2 = df.apply(lambda x : x + 10)

# Using Dataframe.apply() and lambda function
df["A"] = df["A"].apply(lambda x: x-2)

# Apply function NumPy.square() to square the values of two rows 
#'A'and'B
df2 = df.apply(lambda x: np.square(x) if x.name in ['A','B'] else x)

# Using DataFrame.map() to Single Column
df['A'] = df['A'].map(lambda A: A/2.)

# Using DataFrame.assign() and Lambda
df2 = df.assign(B=lambda df: df.B/2)


df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)

# get pandas index
index = madf.index[madf['nucleic_acid_type_new'] == 'dsDNA']









```












































groups
```
big var                                                 small var
--------------------------------------------------    ----------------------------------------------------------------
big var                     big var                   small var                           small var
norm                        long                      norm                                long
-------------------------   ------------------------  --------------------------------    -----------------------------
big var       big var 
norm          norm
b-dist        s-dist
------------- -----------   ------------  ----------- ---------------  ----------------    -------------  --------------



```






