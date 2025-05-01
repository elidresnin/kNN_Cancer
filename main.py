from CancerMatcher import CancerMatcher
from cancer import Cancer
import pandas as pd

class CancerMatcherRunner:
    @staticmethod
    def populate_array_of_test_data(test_data_file_path):
        test_data = []

        try:
            data = pd.read_csv(test_data_file_path)
            test_data: list[Cancer] = [
                Cancer(row["id"], row["diagnosis"], row)  # row is a Series
                 for _, row in data.iterrows()
            ]
            return test_data

        except FileNotFoundError:
            print(f"Error: Test file '{test_data_file_path}' not found")
            return []
        except ValueError as e:
            print(f"Error in file format: {str(e)}")
            return []

    @staticmethod
    def main():

        # Read digits from an input file
        data_collection = CancerMatcher("train.csv")

        # Get test digits
        test_data = CancerMatcherRunner.populate_array_of_test_data("test.csv")
        # # Test Activity 2
        # print("Activity 2 - Read digits from an input file")
        #
        # print("done")
        #
        # # Testing Activity 3
        # print("Activity 3 - Compare two digits")
        # firstCancer = data_collection.cases[1]
        # secondCancer = data_collection.cases[5]
        # firstCancer.set_similarity(secondCancer)
        # print(firstCancer)
        # print(secondCancer)
        #
        #
        #
        # # Testing Activity 4
        # print("Activity 4 - Find most similar")
        # data_collection.compute_similarity(test_data[1])
        # # print(firstCancer)
        # print(data_collection.most_similar())
        #
        # # print("Activity 5 - Find kNN")
        # k = 5
        # kNN = data_collection.find_k_most_similar(k)
        # print("kNN digit's label is " + str(data_collection.k_nearest_neighbors(k)))
        # print(firstCancer)
        # print(kNN)
        # print(data_collection.k_nearest_neighbors(3))



        k = 3
        count = 0

        sim_correct_predictions = 0
        knn_correct_predictions = 0
        wknn_correct_predictions = 0

        total_tests = len(test_data)

        for case in test_data:
            data_collection.compute_similarity(case)
            sim_predicted_label = data_collection.most_similar()
            knn_predicted_label = data_collection.k_nearest_neighbors(k)
            wknn_predicted_label = data_collection.weighted_k_nearest_neighbors(k)
            true_label = case.get_label()
            print("--------")
            if sim_predicted_label.get_label() == true_label:
                sim_correct_predictions += 1
            else:
                print("most sim guessed label: " + str(sim_predicted_label.get_label())+ "; correct label: " + str(true_label))

            if knn_predicted_label == true_label:
                knn_correct_predictions += 1
            else:
                print("knn guessed label: " + str(knn_predicted_label) + "; correct label: " + str(true_label))

            if wknn_predicted_label == true_label:
                wknn_correct_predictions += 1
            else:
                print("wknn guessed label: " + str(wknn_predicted_label) + "; correct label: " + str(true_label))
            count +=1
            print(str(count) + "/" + str(total_tests))



        print("------------------------------------------------------------------------")
        print(f"Most Similar Accuracy: {(sim_correct_predictions / total_tests):.2%}")
        print(f"kNN Accuracy using k = {k}: {(knn_correct_predictions / total_tests):.2%}")
        print(f"Weighted kNN most similar Accuracy using k = {k}: {(wknn_correct_predictions / total_tests):.2%}")

if __name__ == "__main__":
    CancerMatcherRunner.main()