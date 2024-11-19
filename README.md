# Loan Approval System

This project implements a **Loan Approval System** for a financial institution, where the goal is to maximize **net profit** by making intelligent loan approval decisions based on the predicted probability of default.

## Problem Description

- The institution charges a flat fee of **30 OMR** for each approved loan.
- However, there’s a risk: if a loan applicant defaults, the institution loses an amount between **50 OMR and 1000 OMR**.
- The challenge is to approve loans based on the predicted probability of default, optimizing for net profit by adjusting the approval threshold.

## Approach

1. **Predictive Model**: A machine learning model predicts the likelihood of loan default for each applicant.
2. **Loan Decision**: 
   - Approve the loan if the predicted probability of default is below a certain threshold.
   - Reject the loan if the probability is above that threshold.
3. **Net Profit Calculation**:
   - Earn **30 OMR** for each approved loan.
   - If the applicant defaults, the institution incurs a loss between **50 OMR and 1000 OMR**.
4. **Optimize for Net Profit**: The threshold for loan approval is adjusted to maximize overall net profit.

## Submission

- The Python code for data preprocessing, model training, and evaluation is available in the repository.
- The **final model** is selected based on its ability to maximize net profit.

## Screenshots

Below are two screenshots of the system in action when predicting loan approvals and calculating the net profit.

![2](https://github.com/user-attachments/assets/dd71a2f6-5c31-4dff-9108-355de8d81bac)

![لقطة شاشة 2024-11-19 095214](https://github.com/user-attachments/assets/31dc883b-1a6b-4549-93d6-5754f9848330)



For more details and to access the code, visit the repository at: [GitHub Repository URL](#).
