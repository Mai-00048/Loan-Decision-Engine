
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# Uploading Model
model = tf.keras.models.load_model("model.h5")

# --------------------------------------------------------- CALCULATION & LOGIC ----------------------------------------------#
def calculate_profit(probability, threshold=0.5):
    if probability < threshold:
        #Loan Approved: Profit of 30 OMR.
        return 30
    else:
        # Loan Rejected: Random Loss Between 50 OMR and 1000 OMR.
        return -np.random.uniform(50, 1000)

# Tests Different thresholds To Find The One That Yields The Highest Profit.
def optimize_threshold(predictions):
    thresholds = np.linspace(0, 1, 100)
    profits = []
    for threshold in thresholds:
        profit = sum(calculate_profit(p, threshold) for p in predictions)
        profits.append(profit)
    best_threshold = thresholds[np.argmax(profits)]
    return best_threshold, max(profits)

# Display The Result
def show_prediction(prediction, threshold):
    probability = prediction[0][0]
    if probability > threshold:
        risk_level = f"High Risk: {probability*100:.2f}% - Loan Rejected"
        risk_label.config(foreground="red")  #Change To Red
    else:
        risk_level = f"Low Risk: {probability*100:.2f}% - Loan Approved"
        risk_label.config(foreground="green")  #Change To Green
    risk_label.config(text=risk_level)

# Update The Threshold And Net Profit
def update_metrics(predictions):
    best_threshold, max_profit = optimize_threshold(predictions)
    threshold_label.config(text=f"Optimal Threshold: {best_threshold:.2f}")
    profit_label.config(text=f"Maximum Net Profit: {max_profit:.2f} OMR")

# Prediction
def predict_default():
    try:
        age = int(entries[0].get())
        income = float(entries[1].get())
        loan_amount = float(entries[2].get())
        credit_score = int(entries[3].get())
        months_employed = int(entries[4].get())
        num_credit_lines = int(entries[5].get())
        interest_rate = float(entries[6].get())
        loan_term = int(entries[7].get())
        dti_ratio = float(entries[8].get())
        education = education_var.get()
        employment_type = employment_type_var.get()
        marital_status = marital_status_var.get()
        has_mortgage = 1 if has_mortgage_var.get() == "Yes" else 0
        has_dependents = 1 if has_dependents_var.get() == "Yes" else 0
        loan_purpose = loan_purpose_var.get()
        has_cosigner = 1 if has_cosigner_var.get() == "Yes" else 0

        user_data = pd.DataFrame({
            "Age": [age],
            "Income": [income],
            "LoanAmount": [loan_amount],
            "CreditScore": [credit_score],
            "MonthsEmployed": [months_employed],
            "NumCreditLines": [num_credit_lines],
            "InterestRate": [interest_rate],
            "LoanTerm": [loan_term],
            "DTIRatio": [dti_ratio],
            "Education": [education],
            "EmploymentType": [employment_type],
            "MaritalStatus": [marital_status],
            "HasMortgage": [has_mortgage],
            "HasDependents": [has_dependents],
            "LoanPurpose": [loan_purpose],
            "HasCoSigner": [has_cosigner]
        })

        categorical_columns = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
        user_data = pd.get_dummies(user_data, columns=categorical_columns)
        user_data = user_data.astype(np.float32)

        prediction = model.predict(user_data)
        default_probability = prediction[0][0]

        # Calculate Net Profit 
        predictions = [default_probability] 
        update_metrics(predictions)

        # Update Diagram and Display Results
        update_plot(default_probability)
        show_prediction(prediction, threshold=0.5)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ------------------------------------------------------ GRAPHICAL INTERFACE ---------------------------------------------------#
def update_plot(probability):
    ax.clear()

    categories = ["Low Risk", "High Risk"]
    values = [1 - probability, probability]
    colors = ["#4CAF50", "#F44336"]

    angle = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    x = np.cos(angle)
    y = np.sin(angle)
    z = np.zeros_like(x)
    dx = np.ones_like(x) * 0.1
    dy = np.ones_like(y) * 0.1
    dz = values

    ax.bar3d(x, y, z, dx, dy, dz, color=colors, zsort='average')

    ax.set_title("Default Risk Probability")
    canvas.draw()

# --------------------------------------------------- GUI LAYOUT ---------------------------------------------------------#
root = tk.Tk()
root.title("Loan Default Prediction App")
root.geometry("800x630")
root.resizable(False, False)  
root.geometry("800x630+{}+{}".format(int(root.winfo_screenwidth() / 2 - 400), int(root.winfo_screenheight() / 2 - 315)))  

root.configure(bg="#f5f5f5")

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(side="left", fill="both", expand=True)

separator = ttk.Separator(root, orient="vertical")
separator.pack(side="left", fill="y", padx=10)

right_frame = ttk.Frame(root, padding="10")
right_frame.pack(side="right", fill="y")

fig = plt.figure(figsize=(4, 4), dpi=100)
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

update_plot(0.5)

labels = [
    "Age", "Income (OMR)", "Loan Amount (OMR)", "Credit Score",
    "Months Employed", "Number of Credit Lines", "Interest Rate (%)",
    "Loan Term (months)", "DTI Ratio", "Education", "Employment Type",
    "Marital Status", "Has Mortgage", "Has Dependents", "Loan Purpose", "Has Co-Signer"
]
entries = []
row_num = 0

for label in labels[:9]:
    ttk.Label(main_frame, text=f"{label}:", anchor="w").grid(row=row_num, column=0, sticky="w", pady=5)
    entry = ttk.Entry(main_frame, width=30)
    entry.grid(row=row_num, column=1, pady=5)
    entries.append(entry)
    row_num += 1

education_var = tk.StringVar(value="Select Education")
employment_type_var = tk.StringVar(value="Select Employment Type")
marital_status_var = tk.StringVar(value="Select Marital Status")
loan_purpose_var = tk.StringVar(value="Select Loan Purpose")
has_mortgage_var = tk.StringVar(value="No")
has_dependents_var = tk.StringVar(value="No")
has_cosigner_var = tk.StringVar(value="No")

dropdowns = [
    (education_var, ["High School", "Bachelor's", "Master's", "Ph.D."]),
    (employment_type_var, ["Full-Time", "Part-Time", "Self-Employed"]),
    (marital_status_var, ["Single", "Married", "Divorced"]),
    (loan_purpose_var, ["Home", "Debt Consolidation", "Education", "Other"])
]
for var, options in dropdowns:
    ttk.Label(main_frame, text=f"{labels[row_num]}:", anchor="w").grid(row=row_num, column=0, sticky="w", pady=5)
    dropdown = ttk.Combobox(main_frame, textvariable=var, values=options, state="readonly", width=28)
    dropdown.grid(row=row_num, column=1, pady=5)
    row_num += 1

check_vars = [
    (has_mortgage_var, "Has Mortgage"),
    (has_dependents_var, "Has Dependents"),
    (has_cosigner_var, "Has Co-Signer")
]
for var, label in check_vars:
    ttk.Label(main_frame, text=f"{label}:", anchor="w").grid(row=row_num, column=0, sticky="w", pady=5)
    ttk.Checkbutton(main_frame, text="Yes", variable=var, onvalue="Yes", offvalue="No").grid(row=row_num, column=1, pady=5)
    row_num += 1

def clear_inputs():
    # Clear All Entries
    for entry in entries:
        entry.delete(0, tk.END)
    
    # Reset all dropdowns and checkboxes to default values
    education_var.set("Select Education")
    employment_type_var.set("Select Employment Type")
    marital_status_var.set("Select Marital Status")
    loan_purpose_var.set("Select Loan Purpose")
    has_mortgage_var.set("No")
    has_dependents_var.set("No")
    has_cosigner_var.set("No")
    
    # Reset the labels and risk color
    risk_label.config(foreground="black")
    threshold_label.config(text="Optimal Threshold: Not Calculated")
    profit_label.config(text="Maximum Net Profit: Not Calculated OMR")
    risk_label.config(text="Risk Level: Not Calculated")
    
    # Reset the plot to default state (50% probability for both categories)
    update_plot(0.5)

metrics_frame = ttk.Frame(right_frame, padding="10")
metrics_frame.pack(fill="x", pady=10)

threshold_label = ttk.Label(metrics_frame, text="Optimal Threshold: Not Calculated", font=("Helvetica", 12))
threshold_label.pack(pady=(0, 5))   

 
profit_label = ttk.Label(metrics_frame, text="     Maximum Net Profit: Not Calculated OMR", font=("Helvetica", 12))
profit_label.pack(pady=(0, 5))   
 
risk_label = ttk.Label(metrics_frame, text="Risk Level: Not Calculated", font=("Helvetica", 12), anchor="center", padding=10)
risk_label.pack(pady=(0, 5)) 
  
button_frame = ttk.Frame(right_frame)   
button_frame.pack(side="bottom", pady=10)

predict_button = ttk.Button(button_frame, text="Predict", command=predict_default, width=20, padding=(10, 5))
predict_button.pack(side="left", padx=10)  

clear_button = ttk.Button(button_frame, text="Clear", command=clear_inputs, width=20, padding=(10, 5))
clear_button.pack(side="left", padx=10)  


root.mainloop()
