# Miscellaneous

# Jupyter Cheatsheet
- tab: allows you to complete variable names or list functions of a package within code cell
- shift + tab: lets you see function documentation, variable values
- (shift + tab)x2: allows you to see more in-depth documentation
- markdown + latex: $ or $$ and can insert latex
- enter: enter edit mode
- escape: enter command mode
- h: show all commands
- a: create cell above
- b: create cell below
- y: change to code cell
- m: change to markdown cell
- l: toggle line numbers
- d + d: delete cell
- escape + s : saves notebook
- shift + ctrl + p: enter command palette
- arrow keys: move around cells
- x: cut
- c: copy
- v: paste
- space: scroll down
- shift + space: scroll up
- %timeit function(): allows you to time function
- %%timeit: allows you to time entire cell (at top of cell)
- %pdb: turn on interactive debugger (q: turns it off)
- ([additional](http://ipython.readthedocs.io/en/stable/interactive/magics.html) magic commands)
- View > Cell Toolbar > Slideshow: bring up slide cell menu
- jupyter nbconvert notebook.ipynb --to slides: convert to slideshow from file
- jupyter nbconvert notebook.ipynb --to slides --post serve: convert to slideshow and immediately see

# Numpy Cheatsheet
- Numpy array shapes :
  - `(rows, columns)`
  - `(layers, rows, columns)`
- `x = v[None, :]` : add new dimension to array
- Numpy tricks :
  - ```
    if p.shape = (a,), q.shape = (a,b)
    np.dot(p,q).shape = (b,)
    ```
  - ```
    if p = (a,) and q = (b,)
    then p * q[:,None] = (a,) * (b,)[:,None] = (a,) * (b,1) = (b,a)
    ```
  - [Sample code implementing tricks](assets/code/bike-sharing-dataset/)

# Keras
- define model layers, compile loss optimizer metrics, and train on data

# Tensorflow
- different from keras, yet conceptually the same, with more boilerplate code
- `tf.Variable` : for holding values that will change when training (weights, biases)
- `tf.placeholder` : for holding values that will not change when training, but vary (hyperparameters, inputs to the model), when running `sess` must set all placeholder values using the `feed_dict={}`
- `tf.Session()` : use `sess` to first initialize all variables, then train model over epochs and batches by feeding input data to optimizer (optimizer, loss functions built into tf), and it will update the variables (weights, biases) for you
- crafting the model architecture has more detail compared to keras (but is similar to Sequential models), you have to use `tf.add`, `tf.matmul` and additionally features like `tf.nn.relu` or `tf.nn.dropout`
- saving and loading models is similar to keras
