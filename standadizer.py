def get_standardized_input():
    print("--- Heart Disease Risk Assessment Input Tool ---")
    standardized_data = {}

    # 1. Age (Numeric)
    while True:
        try:
            age = float(input("Enter Age (in years): "))
            if age > 0:
                standardized_data["Age"] = age
                break
            print("Age must be a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 2. Sex (Binary)
    while True:
        sex = input("Enter Sex (male/female): ").strip().lower()
        if sex in ['male', 'm']:
            standardized_data["Sex"] = 1
            break
        elif sex in ['female', 'f']:
            standardized_data["Sex"] = 0
            break
        print("Invalid input. Please enter 'male' or 'female'.")

    # 3. Chest Pain Type (Nominal: 1 to 4)
    print("\nChest Pain Types:")
    print("1: Typical Angina\n2: Atypical Angina\n3: Non-Anginal Pain\n4: Asymptomatic")
    while True:
        cp = input("Select Chest Pain Type (1-4): ").strip()
        if cp in ['1', '2', '3', '4']:
            standardized_data["chest pain type"] = int(cp)
            break
        print("Invalid choice. Please enter a number from 1 to 4.")

    # 4. Resting Blood Pressure (Numeric)
    while True:
        try:
            rbp = float(input("\nEnter Resting Blood Pressure (in mm Hg): "))
            if rbp > 0:
                standardized_data["resting bp s"] = rbp
                break
            print("Blood pressure must be a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 5. Serum Cholesterol (Numeric)
    while True:
        try:
            chol = float(input("Enter Serum Cholesterol (in mg/dl): "))
            if chol > 0:
                standardized_data["cholesterol"] = chol
                break
            print("Cholesterol must be a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 6. Fasting Blood Sugar (Binary based on threshold)
    while True:
        try:
            fbs = float(input("Enter Fasting Blood Sugar (in mg/dl): "))
            # Automatically converts to 1 if > 120 mg/dl, else 0 per documentation
            standardized_data["fasting blood sugar"] = 1 if fbs > 120 else 0
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 7. Resting Electrocardiogram Results (Nominal: 0 to 2)
    print("\nResting ECG Results:")
    print("0: Normal")
    print("1: ST-T wave abnormality")
    print("2: Left ventricular hypertrophy")
    while True:
        ecg = input("Select Resting ECG Result (0-2): ").strip()
        if ecg in ['0', '1', '2']:
            standardized_data["resting ecg"] = int(ecg)
            break
        print("Invalid choice. Please enter 0, 1, or 2.")

    # 8. Maximum Heart Rate Achieved (Numeric)
    while True:
        try:
            max_hr = float(input("\nEnter Maximum Heart Rate Achieved (71-202): "))
            if 71 <= max_hr <= 202:
                standardized_data["max heart rate"] = max_hr
                break
            print("Warning: Value is outside standard documentation bounds (71-202), but accepted.")
            standardized_data["max heart rate"] = max_hr
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 9. Exercise Induced Angina (Binary)
    while True:
        ex_angina = input("Exercise induced angina? (yes/no): ").strip().lower()
        if ex_angina in ['yes', 'y']:
            standardized_data["exercise angina"] = 1
            break
        elif ex_angina in ['no', 'n']:
            standardized_data["exercise angina"] = 0
            break
        print("Invalid input. Please enter 'yes' or 'no'.")

    # 10. Oldpeak (Numeric depression)
    while True:
        try:
            oldpeak = float(input("Enter Oldpeak (ST depression value): "))
            standardized_data["oldpeak"] = oldpeak
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 11. Slope of the peak exercise ST segment (Nominal: 1 to 3 mapped to 0-2 index layout or direct documentation numbers)
    print("\nSlope of Peak Exercise ST Segment:")
    print("1: Upsloping\n2: Flat\n3: Downsloping")
    while True:
        slope = input("Select ST Slope (1-3): ").strip()
        if slope in ['1', '2', '3']:
            # Your sample data rows use 0, 1, 2. Let's map doc values (1,2,3) to match your dataset structure (0,1,2)
            # If your model specifically expects 0,1,2:
            standardized_data["ST slope"] = int(slope) - 1 
            break
        print("Invalid choice. Please enter 1, 2, or 3.")

    return standardized_data

# --- Execution Example ---
if __name__ == "__main__":
    user_features = get_standardized_input()
    
    print("\n" + "="*40)
    print("STANDARDIZED DICTIONARY FOR MODEL PREDICTION:")
    print("="*40)
    import pprint
    pprint.pprint(user_features)