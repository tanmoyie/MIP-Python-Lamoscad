list_of_tuples = data_preparation.generate_input_data()
with open(f'Outputs/input_data({date_time}).pickle', 'wb') as f:
    pickle.dump(list_of_tuples, f)
# reading pickle file
input_data2 = pickle.load(open(f"Outputs/input_data({date_time}).pickle", "rb"))

