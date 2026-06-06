"""Menu-driven console application for the project."""

from src.evaluate_model import evaluate_model
from src.predict import predict_student_performance
from src.train_model import train_model


def show_menu() -> None:
    """Display the main menu."""

    print("\nStudent Performance Prediction System")
    print("=" * 42)
    print("1. Train Model")
    print("2. Evaluate Model")
    print("3. Predict Student Performance")
    print("4. Exit")


def main() -> None:
    """Run the menu until the user chooses to exit."""

    while True:
        show_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            evaluate_model()
        elif choice == "3":
            predict_student_performance()
        elif choice == "4":
            print("Thank you for using the system.")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")


if __name__ == "__main__":
    main()
