from CancerMatcher import CancerMatcher
from cancer import Cancer
from PrintHelper import PrintHelper

class DigitMatcherRunner:
    @staticmethod
    def populate_array_of_test_data(test_data_file_path):

        test_data = []

        try:
            with open(test_data_file_path, 'r') as f:
                for line in f:
                    # Split the line into pixel values
                    values = line.strip().split(',')
                    label = int(values[0])
                    # Convert pixel values to integers (0 or 1)
                    data = [0 if int(x) == 0 else 1 for x in values]

                    # Create a Digit object with a dummy label (-1) since test data has no labels
                    test_digit = Cancer(label, data)
                    test_data.append(test_digit)

            return test_data

        except FileNotFoundError:
            print(f"Error: Test file '{test_data_file_path}' not found")
            return []
        except ValueError as e:
            print(f"Error in file format: {str(e)}")
            return []

    @staticmethod
    def main():
        # Test Activity 2
        print("Activity 2 - Read digits from an input file")
        # digit_collection = DigitMatcher("small_train.csv")
        # print(digit_collection.get_digits()[0])

        # Testing Activity 3
        print("Activity 3 - Compare two digits")
        # firstDigit = digit_collection.get_digits()[1]
        # secondDigit = digit_collection.get_digits()[5]
        # firstDigit.set_similarity(secondDigit)
        #print(firstDigit)
        #print(secondDigit)

        # Get test digits
        # test_digits = DigitMatcherRunner.populate_array_of_test_digits("small_test.csv")
        # testDigit = test_digits[10]

        # Testing Activity 4
        print("Activity 4 - Find most similar")
        # digit_collection.compute_similarity(testDigit)
        # print(firstDigit)
        # print(digit_collection.most_similar())

        print("Activity 5 - Find kNN")
        k = 5
        # kNN = digit_collection.find_k_most_similar(k)
        # print("kNN digit's label is " + str(digit_collection.k_nearest_neighbors(k)))
        # PrintHelper.print(firstDigit, kNN)
        # print(digit_collection.k_nearest_neighbors(3))


        print("Activity 6 - Weighted kNN")
        # print("Weighted kNN digit's label is " + str(digit_collection.weighted_k_nearest_neighbors(k)))
        # PrintHelper.print(firstDigit, kNN)

        print("\nActivity 7 - Test Accuracy")

        # Read digits from an input file
        digit_collection = DigitMatcher("train.csv")

        # Get test digits
        test_digits = DigitMatcherRunner.populate_array_of_test_digits("test.csv")

        k = 8

        sim_correct_predictions = 0
        knn_correct_predictions = 0
        wknn_correct_predictions = 0

        total_tests = len(test_digits)

        for test_digit in test_digits:
            digit_collection.compute_similarity(test_digit)
            sim_predicted_label = digit_collection.most_similar()
            knn_predicted_label = digit_collection.k_nearest_neighbors(k)
            wknn_predicted_label = digit_collection.weighted_k_nearest_neighbors(k)
            true_label = test_digit.get_label()
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


        print("------------------------------------------------------------------------")
        print(f"Most Similar Accuracy: {(sim_correct_predictions / total_tests):.2%}")
        print(f"kNN Accuracy using k = {k}: {(knn_correct_predictions / total_tests):.2%}")
        print(f"Weighted kNN most similar Accuracy using k = {k}: {(wknn_correct_predictions / total_tests):.2%}")

if __name__ == "__main__":
    DigitMatcherRunner.main()