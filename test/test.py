from tqdm import tqdm

if __name__ == '__main__':
    py_expected = []
    py_pred = []
    #cpp_pred = []
    err_array = []

    test = train_csv.sample(n=1000)

    for index, rows in tqdm(test.iterrows()):
        row = rows.copy()
        row = row.to_frame()
        row = row.transpose()
        expected = row.pop(target)
        expected = expected.iloc[0]
        py_expected.append(expected)
        #row.pop('id')
        res = model.predict(row, verbose=0)
        res = res[0][0]
        py_pred.append(res)
        err = abs(res - expected)
        #err = res - expected
        err_array.append(err)
        argv = row.values.tolist()[0]
        argv = list(map(str, argv))
        argv = " ".join(argv)
        #cpp_predict = os.popen(f"./build/nn {argv}").read()
        #cpp_pred.append(float(cpp_predict))
        #print(f"Predicted: {res}, cpp predicted: {cpp_predict} and Expected: {expected}")

    fig, axs = plt.subplots(2, 1)
    x_ax = range(len(py_expected))
    axs[0].plot(x_ax, py_expected, label='Python DB (Expected value)')
    axs[0].plot(x_ax, py_pred, label='C++ Predicted value')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title('C++ Predicted values versus Python DataBase Expected values')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Value')
    axs[1].hist(err_array, bins=100)
    axs[1].grid()
    axs[1].set_title('Errors distribution')
    axs[1].set_xlabel('Error')
    axs[1].set_ylabel('Volume')
    plt.show()